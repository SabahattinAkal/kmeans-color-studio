"""Command-line interface for K-Means Color Studio."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .analysis import analyze_cluster_range, render_elbow_svg, save_analysis_json
from .core import (
    build_comparison,
    export_palette_css,
    export_palette_json,
    load_rgb,
    quantize_image,
    save_rgb,
)
from .diagnostics import export_diagnostics


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="kmeans-color-studio",
        description="Quantize an image and export its dominant color palette.",
    )
    parser.add_argument("image", type=Path, help="input image path")
    parser.add_argument("--colors", "-k", default="5", help="palette size (2-32) or 'auto'")
    parser.add_argument("--color-space", choices=["rgb", "lab"], default="rgb")
    parser.add_argument("--min-colors", type=int, default=2, help="minimum K for --colors auto")
    parser.add_argument("--max-colors", type=int, default=10, help="maximum K for --colors auto")
    parser.add_argument("--output", "-o", type=Path, default=Path("output"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-samples", type=int, default=100_000)
    parser.add_argument("--diagnostics", action="store_true", help="export RGB, histogram, grayscale, equalization, and bit-depth diagnostics")
    parser.add_argument("--bits-per-channel", type=int, default=2, help="bit-depth preview used with --diagnostics")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    original = load_rgb(args.image)
    analysis = None
    if str(args.colors).lower() == "auto":
        analysis = analyze_cluster_range(
            original,
            minimum=args.min_colors,
            maximum=args.max_colors,
            color_space=args.color_space,
            seed=args.seed,
        )
        clusters = analysis.selected_clusters
    else:
        try:
            clusters = int(args.colors)
        except ValueError as exc:
            raise ValueError("--colors must be an integer or 'auto'") from exc
    result = quantize_image(
        original,
        clusters=clusters,
        seed=args.seed,
        max_samples=args.max_samples,
        color_space=args.color_space,
    )
    args.output.mkdir(parents=True, exist_ok=True)
    save_rgb(args.output / "quantized.png", result.image)
    save_rgb(args.output / "comparison.png", build_comparison(original, result.image))
    export_palette_json(args.output / "palette.json", result)
    export_palette_css(args.output / "palette.css", result)
    if analysis is not None:
        save_analysis_json(args.output / "cluster_analysis.json", analysis)
        render_elbow_svg(args.output / "cluster_analysis.svg", analysis)
    diagnostics = export_diagnostics(args.output / "diagnostics", original, bits_per_channel=args.bits_per_channel) if args.diagnostics else None
    report = {
        "input": str(args.image),
        "output": str(args.output),
        "clusters": clusters,
        "color_space": result.color_space,
        "colors": result.colors,
        "mean_squared_error": round(result.mean_squared_error, 4),
        "fit_distortion": round(result.fit_distortion, 4),
        "elapsed_seconds": round(result.elapsed_seconds, 4),
        "diagnostics": diagnostics,
    }
    print(json.dumps(report, indent=2))
    return 0
