#!/usr/bin/env python3
"""Minimal web app to upload an image and get the cropped result."""

from __future__ import annotations

import json
import os
import threading
import time
import uuid
from io import BytesIO

from flask import Flask, Response, jsonify, render_template_string, request, send_file, stream_with_context

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
      <pre id="logBox" style="display:none; background:#f3f3ea; border:1px solid #d7d7cc; border-radius:10px; padding:0.8rem; overflow:auto;"></pre>
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
    const originalEl = document.getElementById("original");
    const croppedEl = document.getElementById("cropped");
    const downloadBtn = document.getElementById("downloadBtn");
    const logBox = document.getElementById("logBox");

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
      logBox.style.display = "none";
      logBox.textContent = "";
      statusEl.textContent = "מעבד...";
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
        logBox.style.display = "block";
        logBox.textContent = "";

        const es = new EventSource(`/api/jobs/${jobId}/events`);
        await new Promise((resolve, reject) => {
          es.onmessage = (ev) => {
            try {
              const payload = JSON.parse(ev.data);
              if (payload.type === "log" && payload.message) {
                logBox.textContent += payload.message + "\\n";
                logBox.scrollTop = logBox.scrollHeight;
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
        const logLines = [];
        const correctionMode = res.headers.get("X-Correction-Mode");
        const cropArea = res.headers.get("X-Crop-Area");
        const pxPerChar = res.headers.get("X-Px-Per-Char");
        const outputSize = res.headers.get("X-Output-Size");
        if (correctionMode) logLines.push(`מצב תיקון: ${correctionMode}`);
        if (outputSize) logLines.push(`גודל פלט: ${outputSize}`);
        if (cropArea) logLines.push(`שטח חיתוך: ${cropArea}`);
        if (pxPerChar) logLines.push(`פיקסלים לתו: ${Number(pxPerChar).toFixed(1)}`);
        const timingKeys = [
          ["X-Timing-OCR1", "ocr1"],
          ["X-Timing-Layout", "layout"],
          ["X-Timing-Crop", "crop"],
          ["X-Timing-OCR2", "ocr2"],
          ["X-Timing-Debug", "debug"]
        ];
        let timingSum = 0;
        for (const [headerName, label] of timingKeys) {
          const raw = res.headers.get(headerName);
          if (!raw) continue;
          const v = Number(raw);
          if (Number.isFinite(v)) {
            timingSum += v;
            logLines.push(`זמן ${label}: ${v.toFixed(3)}s`);
          }
        }
        if (timingSum > 0) {
          logLines.push(`סה"כ: ${timingSum.toFixed(3)}s`);
        }
        if (logLines.length > 0) {
          logBox.textContent = logLines.join("\\n");
          logBox.style.display = "block";
        }
        statusEl.textContent = "הסתיים.";
      } catch (err) {
        statusEl.textContent = "שגיאה: " + err.message;
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
            "logs": [],
            "result": None,
            "mime_type": None,
            "meta": None,
            "error": None,
        }

    def worker() -> None:
        def progress(message: str) -> None:
            with JOBS_LOCK:
                job = JOBS.get(job_id)
                if job is not None:
                    job["logs"].append(message)

        with JOBS_LOCK:
            if job_id in JOBS:
                JOBS[job_id]["status"] = "running"
        try:
            cropped_bytes, mime_type, meta = crop_uploaded_image_bytes(raw, progress_cb=progress)
            with JOBS_LOCK:
                job = JOBS.get(job_id)
                if job is not None:
                    job["status"] = "done"
                    job["result"] = cropped_bytes
                    job["mime_type"] = mime_type
                    job["meta"] = meta
        except Exception as exc:
            with JOBS_LOCK:
                job = JOBS.get(job_id)
                if job is not None:
                    job["status"] = "error"
                    job["error"] = str(exc)
                    job["logs"].append(f"ERROR: {exc}")

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
    return response


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", "8000")), debug=False)
