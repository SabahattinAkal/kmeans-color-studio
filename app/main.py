"""Small FastAPI demo for interactive image quantization."""

from __future__ import annotations

import base64

import cv2
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import HTMLResponse
import numpy as np

from kmeans_color_studio.core import quantize_image


app = FastAPI(title="K-Means Color Studio", version="1.0.0")


HTML = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>K-Means Color Studio</title><style>
body{font:16px system-ui;max-width:900px;margin:48px auto;padding:0 20px;background:#10151f;color:#eef2ff}
.card{background:#192231;padding:24px;border-radius:18px}input,button{margin:8px;padding:10px}
button{background:#6d5dfc;color:white;border:0;border-radius:9px}img{max-width:100%;margin-top:20px;border-radius:12px}
.swatch{display:inline-block;width:90px;height:54px;margin:8px;border-radius:8px;padding:6px;color:white;text-shadow:0 1px 3px #000}
</style></head><body><div class="card"><h1>K-Means Color Studio</h1>
<p>Upload an image, choose 2–32 colors, and export a reproducible dominant palette.</p>
<input id="file" type="file" accept="image/*"><input id="colors" type="number" min="2" max="32" value="5">
<select id="space"><option value="rgb">RGB</option><option value="lab">CIELAB</option></select>
<button onclick="run()">Quantize</button><div id="palette"></div><img id="result"></div>
<script>async function run(){const f=document.querySelector('#file').files[0];if(!f)return alert('Choose an image');
const body=new FormData();body.append('file',f);const k=document.querySelector('#colors').value;const s=document.querySelector('#space').value;
const r=await fetch('/api/quantize?colors='+k+'&color_space='+s,{method:'POST',body});const d=await r.json();if(!r.ok)return alert(d.detail);
document.querySelector('#result').src='data:image/png;base64,'+d.image_base64;
document.querySelector('#palette').innerHTML=d.colors.map(c=>`<span class="swatch" style="background:${c.hex}">${c.hex}</span>`).join('');}</script>
</body></html>"""


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return HTML


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/quantize")
async def quantize(
    file: UploadFile = File(...),
    colors: int = Query(5, ge=2, le=32),
    color_space: str = Query("rgb", pattern="^(rgb|lab)$"),
) -> dict[str, object]:
    payload = await file.read()
    if len(payload) > 20 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Image must be smaller than 20 MB")
    bgr = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
    if bgr is None:
        raise HTTPException(status_code=400, detail="Unsupported or invalid image")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    result = quantize_image(rgb, clusters=colors, color_space=color_space)
    ok, encoded = cv2.imencode(".png", cv2.cvtColor(result.image, cv2.COLOR_RGB2BGR))
    if not ok:
        raise HTTPException(status_code=500, detail="Could not encode result")
    return {
        "colors": result.colors,
        "color_space": result.color_space,
        "mean_squared_error": round(result.mean_squared_error, 4),
        "elapsed_seconds": round(result.elapsed_seconds, 4),
        "image_base64": base64.b64encode(encoded).decode("ascii"),
    }
