"""Public API for K-Means Color Studio."""

from .core import QuantizationResult, build_comparison, quantize_image

__all__ = ["QuantizationResult", "build_comparison", "quantize_image"]
__version__ = "1.0.0"
