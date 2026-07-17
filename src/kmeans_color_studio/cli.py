"""Command-line interface for K-Means Color Studio."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .core import (
    build_comparison,
    export_palette_css,
    export_palette_json,
    load_rgb,
    quantize_image,
    save_rgb,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="kmeans-color-studio",
        description="Quantize an image and export its dominant color palette.",
    )
    parser.add_argument("image", type=Path, help="input image path")
    parser.add_argument("--colors", "-k", type=int, default=5, help="palette size (2-32)")
    parser.add_argument("--output", "-o", type=Path, default=Path("output"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-samples", type=int, default=100_000)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    original = load_rgb(args.image)
    result = quantize_image(
        original,
        clusters=args.colors,
        seed=args.seed,
        max_samples=args.max_samples,
    )
    args.output.mkdir(parents=True, exist_ok=True)
    save_rgb(args.output / "quantized.png", result.image)
    save_rgb(args.output / "comparison.png", build_comparison(original, result.image))
    export_palette_json(args.output / "palette.json", result)
    export_palette_css(args.output / "palette.css", result)
    report = {
        "input": str(args.image),
        "output": str(args.output),
        "colors": result.colors,
        "mean_squared_error": round(result.mean_squared_error, 4),
        "elapsed_seconds": round(result.elapsed_seconds, 4),
    }
    print(json.dumps(report, indent=2))
    return 0
