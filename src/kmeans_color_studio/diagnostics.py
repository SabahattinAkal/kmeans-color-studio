"""Classical color diagnostics consolidated from the 2022 image exercises."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from .core import build_comparison, save_rgb


def _image(value: np.ndarray) -> np.ndarray:
    image = np.asarray(value)
    if image.ndim != 3 or image.shape[2] != 3 or image.size == 0:
        raise ValueError("image must have shape (height, width, 3) in RGB order")
    return np.clip(image, 0, 255).astype(np.uint8)


def channel_histograms(image: np.ndarray) -> np.ndarray:
    rgb = _image(image)
    return np.vstack([np.bincount(rgb[:, :, channel].ravel(), minlength=256) for channel in range(3)])


def channel_montage(image: np.ndarray) -> np.ndarray:
    rgb = _image(image)
    views = []
    for channel in range(3):
        view = np.zeros_like(rgb)
        view[:, :, channel] = rgb[:, :, channel]
        views.append(view)
    gap = np.full((rgb.shape[0], 8, 3), 248, dtype=np.uint8)
    return np.hstack([views[0], gap, views[1], gap, views[2]])


def grayscale_and_equalized(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    gray = cv2.cvtColor(_image(image), cv2.COLOR_RGB2GRAY)
    equalized = cv2.equalizeHist(gray)
    return gray, equalized


def reduce_bit_depth(image: np.ndarray, bits_per_channel: int) -> np.ndarray:
    if not 1 <= bits_per_channel <= 8:
        raise ValueError("bits_per_channel must be between 1 and 8")
    rgb = _image(image)
    levels = 2**bits_per_channel
    quantized = np.rint(rgb.astype(float) * (levels - 1) / 255.0) * 255.0 / (levels - 1)
    return np.clip(np.rint(quantized), 0, 255).astype(np.uint8)


def _entropy(gray: np.ndarray) -> float:
    counts = np.bincount(gray.ravel(), minlength=256).astype(float)
    probabilities = counts[counts > 0] / counts.sum()
    return float(-np.sum(probabilities * np.log2(probabilities)))


def render_histogram_svg(path: str | Path, histograms: np.ndarray) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    width, height, margin = 900, 420, 54
    maximum = max(float(histograms.max()), 1.0)
    colors = ("#ef4444", "#22c55e", "#3b82f6")
    lines = []
    for histogram, color in zip(histograms, colors, strict=True):
        points = []
        for index, count in enumerate(histogram):
            x = margin + index / 255 * (width - 2 * margin)
            y = height - margin - float(count) / maximum * (height - 2 * margin)
            points.append(f"{x:.1f},{y:.1f}")
        lines.append(f'<polyline fill="none" stroke="{color}" stroke-width="2" points="{" ".join(points)}"/>')
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect width="100%" height="100%" fill="#ffffff"/>
<text x="{width / 2}" y="30" text-anchor="middle" font-family="Arial" font-size="22">RGB channel histograms</text>
<line x1="{margin}" y1="{height-margin}" x2="{width-margin}" y2="{height-margin}" stroke="#334155"/>
<line x1="{margin}" y1="{margin}" x2="{margin}" y2="{height-margin}" stroke="#334155"/>
<text x="{width/2}" y="{height-12}" text-anchor="middle" font-family="Arial" font-size="14">intensity (0–255)</text>
{''.join(lines)}
</svg>'''
    destination.write_text(svg, encoding="utf-8")
    return destination


def export_diagnostics(output: str | Path, image: np.ndarray, *, bits_per_channel: int = 2) -> dict[str, object]:
    target = Path(output)
    target.mkdir(parents=True, exist_ok=True)
    rgb = _image(image)
    gray, equalized = grayscale_and_equalized(rgb)
    reduced = reduce_bit_depth(rgb, bits_per_channel)
    histograms = channel_histograms(rgb)
    save_rgb(target / "rgb-channels.png", channel_montage(rgb))
    save_rgb(target / "grayscale.png", np.repeat(gray[:, :, None], 3, axis=2))
    save_rgb(target / "equalized-grayscale.png", np.repeat(equalized[:, :, None], 3, axis=2))
    save_rgb(target / "bit-depth-comparison.png", build_comparison(rgb, reduced))
    render_histogram_svg(target / "rgb-histograms.svg", histograms)
    report = {
        "shape": list(rgb.shape),
        "channel_mean_rgb": np.mean(rgb, axis=(0, 1)).round(4).tolist(),
        "channel_std_rgb": np.std(rgb, axis=(0, 1)).round(4).tolist(),
        "grayscale_entropy_bits": round(_entropy(gray), 6),
        "equalized_entropy_bits": round(_entropy(equalized), 6),
        "grayscale_std": round(float(np.std(gray)), 6),
        "equalized_std": round(float(np.std(equalized)), 6),
        "bits_per_channel": bits_per_channel,
        "levels_per_channel": 2**bits_per_channel,
        "unique_reduced_colors": int(len(np.unique(reduced.reshape(-1, 3), axis=0))),
    }
    (target / "diagnostics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
