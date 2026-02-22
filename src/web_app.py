#!/usr/bin/env python3
"""Minimal web app to upload an image and get the cropped result."""

from __future__ import annotations

import json
import os
import hashlib
import pathlib
import platform
import subprocess
import threading
import time
import uuid
from io import BytesIO

from flask import Flask, Response, jsonify, render_template_string, request, send_file, stream_with_context
import PIL
import pytesseract

from web_cropper import crop_uploaded_image_bytes

app = Flask(__name__)
JOBS: dict[str, dict] = {}
JOBS_LOCK = threading.Lock()

INDEX_HTML = """<!doctype html>
<html lang="he" dir="rtl">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>ממשק בדיקת קרופר</title>
  <style>
    :root { --bg: #f7f7ef; --card: #ffffff; --ink: #202021; --accent: #0e7a52; --line: #d7d7cc; }
    body { margin: 0; font-family: "Segoe UI", Tahoma, sans-serif; color: var(--ink);
           background: radial-gradient(circle at top right, #e8f5ea, var(--bg)); }
    main { max-width: 860px; margin: 2rem auto; padding: 0 1rem; }
    .card { background: var(--card); border: 1px solid var(--line); border-radius: 12px; padding: 1rem; }
    h1 { margin-top: 0; }
    .row { display: flex; gap: 1rem; flex-wrap: wrap; align-items: center; }
    button { border: 0; border-radius: 10px; padding: 0.7rem 1rem; color: #fff; background: var(--accent); cursor: pointer; }
    button:disabled { opacity: 0.55; cursor: not-allowed; }
    #status { min-height: 1.25rem; }
    img { max-width: 100%; height: auto; border: 1px solid var(--line); border-radius: 10px; display: none; }
    .preview-grid { display: grid; grid-template-columns: 1fr; gap: 1rem; margin-top: 1rem; }
    @media (min-width: 800px) { .preview-grid { grid-template-columns: 1fr 1fr; } }
  </style>
</head>
<body>
  <main>
    <div class="card">
      <h1>ממשק בדיקת קרופר</h1>
      <p>העלו תמונה וקבלו חיתוך אוטומטי מבוסס OCR.</p>
      <div class="row">
        <input id="fileInput" type="file" accept="image/*">
        <button id="cropBtn">העלאה וחיתוך</button>
        <a id="downloadBtn" href="#" download="cropped.png" style="display:none;">הורדת התמונה החתוכה</a>
      </div>
      <p id="status"></p>
      <pre id="timingBox" style="display:block; background:#f3f3ea; border:1px solid #d7d7cc; border-radius:10px; padding:0.8rem; overflow:auto; white-space:pre-wrap;"></pre>
      <div class="preview-grid">
        <div>
          <h3>תמונה מקורית</h3>
          <img id="original" alt="original image">
        </div>
        <div>
          <h3>תמונה חתוכה</h3>
          <img id="cropped" alt="cropped image">
        </div>
      </div>
    </div>
  </main>
  <script>
    const fileInput = document.getElementById("fileInput");
    const cropBtn = document.getElementById("cropBtn");
    const statusEl = document.getElementById("status");
    const timingBox = document.getElementById("timingBox");
    const originalEl = document.getElementById("original");
    const croppedEl = document.getElementById("cropped");
    const downloadBtn = document.getElementById("downloadBtn");

    cropBtn.addEventListener("click", async () => {
      const file = fileInput.files[0];
      if (!file) {
        statusEl.textContent = "יש לבחור תמונה קודם.";
        return;
      }
      const originalName = file.name.replace(/[.][^/.]+$/, "");
      const downloadName = `${originalName}_cropped.jpg`;
      originalEl.src = URL.createObjectURL(file);
      originalEl.style.display = "block";
      croppedEl.style.display = "none";
      downloadBtn.style.display = "none";
      timingBox.textContent = "";
      statusEl.textContent = "Processing...";
      cropBtn.disabled = true;
      try {
        const formData = new FormData();
        formData.append("image", file);
        const startRes = await fetch("/api/crop", { method: "POST", body: formData });
        if (!startRes.ok) {
          const contentType = startRes.headers.get("content-type") || "";
          let message = `שגיאת שרת (${startRes.status})`;
          if (contentType.includes("application/json")) {
            const data = await startRes.json();
            message = data.error || message;
          } else {
            const bodyText = await startRes.text();
            if (bodyText) message = `${message}: ${bodyText.slice(0, 120)}`;
          }
          throw new Error(message);
        }
        const startData = await startRes.json();
        const jobId = startData.job_id;
        timingBox.textContent += "started\\n";
        let elapsedTotal = null;
        let prevStepSec = null;
        let prevStepLabel = null;
        const MIN_SPAN_SEC = 0.01;
        const HIDDEN_SPAN_LABELS = new Set(["job_queued", "job_started"]);

        const es = new EventSource(`/api/jobs/${jobId}/events`);
        await new Promise((resolve, reject) => {
          es.onmessage = (ev) => {
            try {
              const payload = JSON.parse(ev.data);
              if (payload.type === "log" && payload.message) {
                timingBox.textContent += payload.message + "\\n";
                const m = payload.message.match(/^\\[\\+(\\d+(?:\\.\\d+)?)s\\]/);
                if (m) {
                  const sec = Number(m[1]);
                  if (Number.isFinite(sec)) {
                    elapsedTotal = sec;
                    const label = payload.message.replace(/^\\[\\+\\d+(?:\\.\\d+)?s\\]\\s*/, "");
                    if (prevStepSec !== null && prevStepLabel) {
                      const span = sec - prevStepSec;
                      if (span >= MIN_SPAN_SEC && !HIDDEN_SPAN_LABELS.has(prevStepLabel)) {
                        timingBox.textContent += `span ${prevStepLabel} ${span.toFixed(3)}s\\n`;
                      }
                    }
                    prevStepSec = sec;
                    prevStepLabel = label;
                  }
                }
                timingBox.scrollTop = timingBox.scrollHeight;
              }
              if (payload.type === "done") {
                es.close();
                resolve();
              }
              if (payload.type === "error") {
                es.close();
                reject(new Error(payload.message || "שגיאת עיבוד"));
              }
            } catch (err) {
              es.close();
              reject(err);
            }
          };
          es.onerror = () => {
            es.close();
            reject(new Error("נותק חיבור הלוגים לשרת"));
          };
        });

        const res = await fetch(`/api/jobs/${jobId}/result`);
        if (!res.ok) {
          const contentType = res.headers.get("content-type") || "";
          let message = `שגיאת שרת (${res.status})`;
          if (contentType.includes("application/json")) {
            const data = await res.json();
            message = data.error || message;
          } else {
            const bodyText = await res.text();
            if (bodyText) {
              message = `${message}: ${bodyText.slice(0, 120)}`;
            }
          }
          throw new Error(message);
        }
        const blob = await res.blob();
        const croppedUrl = URL.createObjectURL(blob);
        croppedEl.src = croppedUrl;
        croppedEl.style.display = "block";
        downloadBtn.href = croppedUrl;
        downloadBtn.download = downloadName;
        downloadBtn.style.display = "inline-block";
        if (elapsedTotal !== null) {
          timingBox.textContent += `elapsed_total ${elapsedTotal.toFixed(3)}s\\n`;
        }
        statusEl.textContent = "Done.";
      } catch (err) {
        statusEl.textContent = "Error: " + err.message;
        timingBox.textContent += `error ${err.message}\\n`;
      } finally {
        cropBtn.disabled = false;
      }
    });
  </script>
</body>
</html>
"""


@app.get("/")
def index():
    return render_template_string(INDEX_HTML)


@app.post("/api/crop")
def crop_api():
    if "image" not in request.files:
        return jsonify({"error": "Missing image file field: image"}), 400
    uploaded = request.files["image"]
    if not uploaded.filename:
        return jsonify({"error": "No file selected"}), 400
    raw = uploaded.read()
    if not raw:
        return jsonify({"error": "Uploaded file is empty"}), 400
    job_id = uuid.uuid4().hex
    with JOBS_LOCK:
        JOBS[job_id] = {
            "status": "queued",
            "logs": ["[+0.000s] job_queued"],
            "result": None,
            "mime_type": None,
            "meta": None,
            "error": None,
        }

    def worker() -> None:
        t0 = time.perf_counter()

        def elapsed_prefix() -> str:
            return f"[+{(time.perf_counter() - t0):.3f}s]"

        def progress(message: str) -> None:
            with JOBS_LOCK:
                job = JOBS.get(job_id)
                if job is not None:
                    job["logs"].append(f"{elapsed_prefix()} {message}")

        with JOBS_LOCK:
            if job_id in JOBS:
                JOBS[job_id]["status"] = "running"
                JOBS[job_id]["logs"].append(f"{elapsed_prefix()} job_started")
        try:
            cropped_bytes, mime_type, meta = crop_uploaded_image_bytes(raw, progress_cb=progress)
            with JOBS_LOCK:
                job = JOBS.get(job_id)
                if job is not None:
                    job["status"] = "done"
                    job["result"] = cropped_bytes
                    job["mime_type"] = mime_type
                    job["meta"] = meta
                    job["logs"].append(f"{elapsed_prefix()} job_finished")
        except Exception as exc:
            with JOBS_LOCK:
                job = JOBS.get(job_id)
                if job is not None:
                    job["status"] = "error"
                    job["error"] = str(exc)
                    job["logs"].append(f"{elapsed_prefix()} error {exc}")

    threading.Thread(target=worker, daemon=True).start()
    return jsonify({"job_id": job_id})


@app.get("/api/jobs/<job_id>/events")
def crop_job_events(job_id: str):
    def event_stream():
        sent = 0
        while True:
            with JOBS_LOCK:
                job = JOBS.get(job_id)
                if job is None:
                    payload = {"type": "error", "message": "Job not found"}
                    yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
                    return
                logs = list(job["logs"])
                status = job["status"]
                error = job["error"]
            while sent < len(logs):
                payload = {"type": "log", "message": logs[sent]}
                yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
                sent += 1
            if status == "done":
                yield f"data: {json.dumps({'type': 'done'}, ensure_ascii=False)}\n\n"
                return
            if status == "error":
                payload = {"type": "error", "message": error or "Processing failed"}
                yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
                return
            yield ": keepalive\n\n"
            time.sleep(0.4)

    return Response(stream_with_context(event_stream()), mimetype="text/event-stream")


@app.get("/api/jobs/<job_id>/result")
def crop_job_result(job_id: str):
    with JOBS_LOCK:
        job = JOBS.get(job_id)
    if job is None:
        return jsonify({"error": "Job not found"}), 404
    if job["status"] == "error":
        return jsonify({"error": job["error"] or "Processing failed"}), 500
    if job["status"] != "done" or not job["result"]:
        return jsonify({"error": "Result is not ready"}), 425

    response = send_file(
        BytesIO(job["result"]),
        mimetype=job["mime_type"] or "image/jpeg",
        as_attachment=True,
        download_name="cropped.jpg",
    )
    meta = job.get("meta") or {}
    timing = meta.get("timing", {})
    timing_detail = meta.get("timing_detail", {})
    response.headers["X-Correction-Mode"] = str(meta.get("correction_mode", ""))
    response.headers["X-Crop-Area"] = str(meta.get("crop_area", ""))
    response.headers["X-Px-Per-Char"] = f"{float(meta.get('px_per_char', 0.0)):.6f}"
    size = meta.get("size", {})
    response.headers["X-Output-Size"] = f"{size.get('width', 0)}x{size.get('height', 0)}"
    response.headers["X-Timing-OCR1"] = f"{float(timing.get('ocr1', 0.0)):.6f}"
    response.headers["X-Timing-Layout"] = f"{float(timing.get('layout', 0.0)):.6f}"
    response.headers["X-Timing-Crop"] = f"{float(timing.get('crop', 0.0)):.6f}"
    response.headers["X-Timing-OCR2"] = f"{float(timing.get('ocr2', 0.0)):.6f}"
    response.headers["X-Timing-Debug"] = f"{float(timing.get('debug', 0.0)):.6f}"
    response.headers["X-Timing-Crop-Finalize"] = f"{float(timing.get('crop_finalize', 0.0)):.6f}"
    response.headers["X-Timing-Post-Crop-Stripes"] = f"{float(timing.get('post_crop_stripes', 0.0)):.6f}"
    for key, value in timing_detail.items():
        header_key = "X-Timing-Detail-" + str(key).replace("_", "-")
        response.headers[header_key] = f"{float(value):.6f}"
    return response


@app.get("/api/runtime")
def runtime_info():
    tessdata_prefix = os.environ.get("TESSDATA_PREFIX", "")
    heb_path = pathlib.Path(tessdata_prefix) / "tessdata" / "heb.traineddata" if tessdata_prefix else None
    heb_sha256 = None
    heb_size = None
    if heb_path is not None and heb_path.exists() and heb_path.is_file():
        try:
            data = heb_path.read_bytes()
            heb_sha256 = hashlib.sha256(data).hexdigest()
            heb_size = len(data)
        except Exception:
            heb_sha256 = None
            heb_size = None
    try:
        version_out = subprocess.check_output(["tesseract", "--version"], text=True)
        tesseract_version = version_out.splitlines()[0].strip() if version_out else ""
    except Exception as exc:
        tesseract_version = f"error: {exc}"
    return jsonify(
        {
            "python": platform.python_version(),
            "pillow": PIL.__version__,
            "pytesseract": pytesseract.__version__,
            "tesseract": tesseract_version,
            "tessdata_prefix": tessdata_prefix,
            "heb_traineddata_path": str(heb_path) if heb_path is not None else "",
            "heb_traineddata_size": heb_size,
            "heb_traineddata_sha256": heb_sha256,
        }
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", "8000")), debug=False)
