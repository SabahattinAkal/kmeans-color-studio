"""Image color quantization and palette export utilities."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import time

import cv2
import numpy as np


@dataclass(frozen=True)
class QuantizationResult:
    """Result of a deterministic K-Means color quantization run."""

    image: np.ndarray
    palette: np.ndarray
    counts: np.ndarray
    mean_squared_error: float
    elapsed_seconds: float

    @property
    def colors(self) -> list[dict[str, object]]:
        total = int(self.counts.sum())
        items: list[dict[str, object]] = []
        for rgb, count in zip(self.palette, self.counts, strict=True):
            r, g, b = (int(channel) for channel in rgb)
            items.append(
                {
                    "rgb": [r, g, b],
                    "hex": f"#{r:02X}{g:02X}{b:02X}",
                    "pixel_count": int(count),
                    "fraction": round(int(count) / total, 6),
                }
            )
        return items


def _validate_image(image: np.ndarray, clusters: int) -> np.ndarray:
    if image is None or not isinstance(image, np.ndarray):
        raise ValueError("image must be a NumPy array")
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("image must have shape (height, width, 3) in RGB order")
    if image.size == 0:
        raise ValueError("image cannot be empty")
    if not 2 <= clusters <= 32:
        raise ValueError("clusters must be between 2 and 32")
    if image.shape[0] * image.shape[1] < clusters:
        raise ValueError("image must contain at least as many pixels as clusters")
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(image)


def quantize_image(
    image: np.ndarray,
    clusters: int = 5,
    *,
    seed: int = 42,
    max_samples: int = 100_000,
    attempts: int = 5,
) -> QuantizationResult:
    """Reduce an RGB image to ``clusters`` colors.

    K-Means is fitted on a reproducible random pixel sample for large images, then
    every pixel is assigned to its nearest center. Palette entries are sorted by
    frequency so the first item is always the dominant color.
    """

    image = _validate_image(image, clusters)
    if max_samples < clusters:
        raise ValueError("max_samples must be at least the number of clusters")
    if attempts < 1:
        raise ValueError("attempts must be positive")

    started = time.perf_counter()
    pixels = image.reshape(-1, 3).astype(np.float32)
    rng = np.random.default_rng(seed)
    if len(pixels) > max_samples:
        indices = rng.choice(len(pixels), size=max_samples, replace=False)
        fit_pixels = pixels[indices]
    else:
        fit_pixels = pixels

    cv2.setRNGSeed(int(seed))
    criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        100,
        0.2,
    )
    _, _, centers = cv2.kmeans(
        fit_pixels,
        clusters,
        None,
        criteria,
        attempts,
        cv2.KMEANS_PP_CENTERS,
    )

    # Assign in chunks to keep memory bounded for multi-megapixel images.
    labels = np.empty(len(pixels), dtype=np.int32)
    error_sum = 0.0
    chunk_size = 250_000
    for start in range(0, len(pixels), chunk_size):
        chunk = pixels[start : start + chunk_size]
        distances = np.sum((chunk[:, None, :] - centers[None, :, :]) ** 2, axis=2)
        chunk_labels = np.argmin(distances, axis=1)
        labels[start : start + len(chunk)] = chunk_labels
        error_sum += float(distances[np.arange(len(chunk)), chunk_labels].sum())

    counts = np.bincount(labels, minlength=clusters)
    order = np.argsort(-counts, kind="stable")
    remap = np.empty(clusters, dtype=np.int32)
    remap[order] = np.arange(clusters)
    sorted_labels = remap[labels]
    palette = np.clip(np.rint(centers[order]), 0, 255).astype(np.uint8)
    quantized = palette[sorted_labels].reshape(image.shape)

    return QuantizationResult(
        image=quantized,
        palette=palette,
        counts=counts[order].astype(np.int64),
        mean_squared_error=error_sum / len(pixels),
        elapsed_seconds=time.perf_counter() - started,
    )


def load_rgb(path: str | Path) -> np.ndarray:
    source = Path(path)
    image = cv2.imread(str(source), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"could not read image: {source}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def save_rgb(path: str | Path, image: np.ndarray) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(destination), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    if not ok:
        raise OSError(f"could not write image: {destination}")
    return destination


def build_comparison(original: np.ndarray, quantized: np.ndarray) -> np.ndarray:
    if original.shape != quantized.shape:
        raise ValueError("original and quantized images must have the same shape")
    header_height = 42
    gap = 12
    height, width = original.shape[:2]
    canvas = np.full((height + header_height, width * 2 + gap, 3), 248, dtype=np.uint8)
    canvas[header_height:, :width] = original
    canvas[header_height:, width + gap :] = quantized
    cv2.putText(canvas, "ORIGINAL", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (35, 35, 35), 2)
    cv2.putText(
        canvas,
        "QUANTIZED",
        (width + gap + 12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (35, 35, 35),
        2,
    )
    return canvas


def export_palette_json(path: str | Path, result: QuantizationResult) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "colors": result.colors,
        "mean_squared_error": round(result.mean_squared_error, 4),
        "elapsed_seconds": round(result.elapsed_seconds, 6),
    }
    destination.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return destination


def export_palette_css(path: str | Path, result: QuantizationResult) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    variables = "\n".join(
        f"  --dominant-color-{index}: {color['hex']};"
        for index, color in enumerate(result.colors, start=1)
    )
    destination.write_text(f":root {{\n{variables}\n}}\n", encoding="utf-8")
    return destination
