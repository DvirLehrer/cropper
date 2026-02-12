#!/usr/bin/env python3
"""Minimal web app to upload an image and get the cropped result."""

from __future__ import annotations

import os
from io import BytesIO

from flask import Flask, jsonify, render_template_string, request, send_file

from web_cropper import crop_uploaded_image_bytes

app = Flask(__name__)

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
      statusEl.textContent = "מעבד...";
      cropBtn.disabled = true;
      try {
        const formData = new FormData();
        formData.append("image", file);
        const res = await fetch("/api/crop", { method: "POST", body: formData });
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
    try:
        cropped_bytes, mime_type = crop_uploaded_image_bytes(raw)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500
    return send_file(
        BytesIO(cropped_bytes),
        mimetype=mime_type,
        as_attachment=True,
        download_name="cropped.jpg",
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", "8000")), debug=False)
