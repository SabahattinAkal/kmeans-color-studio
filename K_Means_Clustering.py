"""Backward-compatible entry point for the rebuilt project."""

from kmeans_color_studio.cli import main


if __name__ == "__main__":
    raise SystemExit(main())
