"""Public API for K-Means Color Studio."""

from .core import QuantizationResult, build_comparison, quantize_image
from .analysis import ClusterAnalysis, analyze_cluster_range

__all__ = ["ClusterAnalysis", "QuantizationResult", "analyze_cluster_range", "build_comparison", "quantize_image"]
__version__ = "1.1.0"
