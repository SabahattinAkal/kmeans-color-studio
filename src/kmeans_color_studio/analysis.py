"""Automatic palette-size analysis and dependency-free SVG reports."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import html
import json
import math
from pathlib import Path
from typing import Literal

import cv2
import numpy as np

from .core import quantize_image


@dataclass(frozen=True)
class ClusterCandidate:
    clusters: int
    rgb_mse: float
    fit_distortion: float
    elapsed_seconds: float


@dataclass(frozen=True)
class ClusterAnalysis:
    selected_clusters: int
    color_space: str
    candidates: tuple[ClusterCandidate, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "selected_clusters": self.selected_clusters,
            "color_space": self.color_space,
            "selection_method": "maximum distance from the log-distortion endpoint line",
            "candidates": [asdict(candidate) for candidate in self.candidates],
        }


def _analysis_thumbnail(image: np.ndarray, max_pixels: int) -> np.ndarray:
    height, width = image.shape[:2]
    if height * width <= max_pixels:
        return image
    scale = math.sqrt(max_pixels / (height * width))
    return cv2.resize(image, (max(2, round(width * scale)), max(2, round(height * scale))), interpolation=cv2.INTER_AREA)


def _select_elbow(candidates: list[ClusterCandidate]) -> int:
    if len(candidates) < 3:
        return candidates[-1].clusters
    x = np.asarray([item.clusters for item in candidates], dtype=float)
    y = np.log(np.maximum([item.fit_distortion for item in candidates], np.finfo(float).eps))
    x = (x - x.min()) / max(np.ptp(x), np.finfo(float).eps)
    y = (y - y.min()) / max(np.ptp(y), np.finfo(float).eps)
    start = np.array([x[0], y[0]])
    end = np.array([x[-1], y[-1]])
    direction = end - start
    distances = np.abs(direction[0] * (start[1] - y) - (start[0] - x) * direction[1]) / np.linalg.norm(direction)
    distances[[0, -1]] = -1
    return candidates[int(np.argmax(distances))].clusters


def analyze_cluster_range(
    image: np.ndarray,
    *,
    minimum: int = 2,
    maximum: int = 10,
    color_space: Literal["rgb", "lab"] = "lab",
    seed: int = 42,
    max_analysis_pixels: int = 120_000,
) -> ClusterAnalysis:
    if not 2 <= minimum < maximum <= 32:
        raise ValueError("cluster range must satisfy 2 <= minimum < maximum <= 32")
    thumbnail = _analysis_thumbnail(image, max_analysis_pixels)
    candidates: list[ClusterCandidate] = []
    for clusters in range(minimum, maximum + 1):
        result = quantize_image(
            thumbnail,
            clusters,
            seed=seed,
            max_samples=max_analysis_pixels,
            attempts=3,
            color_space=color_space,
        )
        candidates.append(
            ClusterCandidate(
                clusters=clusters,
                rgb_mse=round(result.mean_squared_error, 6),
                fit_distortion=round(result.fit_distortion, 6),
                elapsed_seconds=round(result.elapsed_seconds, 6),
            )
        )
    return ClusterAnalysis(
        selected_clusters=_select_elbow(candidates),
        color_space=color_space,
        candidates=tuple(candidates),
    )


def save_analysis_json(path: str | Path, analysis: ClusterAnalysis) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(analysis.to_dict(), indent=2), encoding="utf-8")
    return destination


def render_elbow_svg(path: str | Path, analysis: ClusterAnalysis) -> Path:
    width, height = 900, 500
    left, right, top, bottom = 84, 32, 64, 72
    plot_width, plot_height = width - left - right, height - top - bottom
    values = np.asarray([item.fit_distortion for item in analysis.candidates], dtype=float)
    minimum, maximum = float(values.min()), float(values.max())
    span = max(maximum - minimum, np.finfo(float).eps)
    points: list[str] = []
    circles: list[str] = []
    for index, candidate in enumerate(analysis.candidates):
        x = left + index / max(1, len(analysis.candidates) - 1) * plot_width
        y = top + (maximum - candidate.fit_distortion) / span * plot_height
        points.append(f"{x:.1f},{y:.1f}")
        selected = candidate.clusters == analysis.selected_clusters
        color = "#f59e0b" if selected else "#38bdf8"
        radius = 8 if selected else 5
        circles.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{radius}" fill="{color}"/>')
        circles.append(f'<text x="{x:.1f}" y="{height - 34}" text-anchor="middle" font-size="13">{candidate.clusters}</text>')
    title = html.escape(f"Automatic K analysis — {analysis.color_space.upper()} — selected K={analysis.selected_clusters}")
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#0f172a"/>',
        '<style>text{font-family:system-ui,sans-serif;fill:#e2e8f0}</style>',
        f'<text x="{width / 2}" y="32" text-anchor="middle" font-size="20" font-weight="700">{title}</text>',
        f'<line x1="{left}" y1="{top + plot_height}" x2="{width-right}" y2="{top + plot_height}" stroke="#64748b"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_height}" stroke="#64748b"/>',
        f'<polyline points="{" ".join(points)}" fill="none" stroke="#38bdf8" stroke-width="3"/>',
        *circles,
        f'<text x="{width / 2}" y="{height - 8}" text-anchor="middle" font-size="14">Palette size (K)</text>',
        f'<text x="20" y="{height / 2}" transform="rotate(-90 20 {height / 2})" text-anchor="middle" font-size="14">Fit distortion</text>',
        '</svg>',
    ]
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(svg), encoding="utf-8")
    return destination
