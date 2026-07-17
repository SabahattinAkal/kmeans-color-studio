import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from kmeans_color_studio.analysis import analyze_cluster_range, render_elbow_svg
from kmeans_color_studio.core import export_palette_css, export_palette_json, quantize_image
from kmeans_color_studio.diagnostics import channel_histograms, grayscale_and_equalized, reduce_bit_depth


class QuantizationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.image = np.zeros((24, 24, 3), dtype=np.uint8)
        self.image[:, :8] = [240, 20, 20]
        self.image[:, 8:16] = [20, 240, 20]
        self.image[:, 16:] = [20, 20, 240]

    def test_quantizes_known_three_color_image(self) -> None:
        result = quantize_image(self.image, clusters=3, seed=7)
        self.assertEqual(result.image.shape, self.image.shape)
        self.assertEqual(result.palette.shape, (3, 3))
        self.assertEqual(int(result.counts.sum()), self.image.shape[0] * self.image.shape[1])
        self.assertLess(result.mean_squared_error, 0.01)
        self.assertEqual(len(np.unique(result.image.reshape(-1, 3), axis=0)), 3)

    def test_is_reproducible(self) -> None:
        first = quantize_image(self.image, clusters=3, seed=42)
        second = quantize_image(self.image, clusters=3, seed=42)
        np.testing.assert_array_equal(first.image, second.image)
        np.testing.assert_array_equal(first.palette, second.palette)

    def test_lab_color_space_preserves_shape(self) -> None:
        result = quantize_image(self.image, clusters=3, seed=42, color_space="lab")
        self.assertEqual(result.image.shape, self.image.shape)
        self.assertEqual(result.color_space, "lab")
        self.assertEqual(result.clusters, 3)
        self.assertTrue(np.isfinite(result.fit_distortion))

    def test_automatic_cluster_analysis_and_svg(self) -> None:
        analysis = analyze_cluster_range(self.image, minimum=2, maximum=5, seed=42)
        self.assertIn(analysis.selected_clusters, range(2, 6))
        self.assertEqual(len(analysis.candidates), 4)
        with tempfile.TemporaryDirectory() as directory:
            path = render_elbow_svg(Path(directory) / "elbow.svg", analysis)
            self.assertIn("Automatic K analysis", path.read_text(encoding="utf-8"))

    def test_rejects_invalid_cluster_count(self) -> None:
        with self.assertRaises(ValueError):
            quantize_image(self.image, clusters=1)

    def test_exports_json_and_css(self) -> None:
        result = quantize_image(self.image, clusters=3)
        with tempfile.TemporaryDirectory() as directory:
            json_path = export_palette_json(Path(directory) / "palette.json", result)
            css_path = export_palette_css(Path(directory) / "palette.css", result)
            payload = json.loads(json_path.read_text(encoding="utf-8"))
            self.assertEqual(len(payload["colors"]), 3)
            self.assertIn("--dominant-color-1", css_path.read_text(encoding="utf-8"))

    def test_classical_image_diagnostics(self) -> None:
        histograms = channel_histograms(self.image)
        self.assertEqual(histograms.shape, (3, 256))
        self.assertEqual(int(histograms[0].sum()), 24 * 24)
        gray, equalized = grayscale_and_equalized(self.image)
        self.assertEqual(gray.shape, (24, 24))
        self.assertEqual(equalized.shape, gray.shape)
        reduced = reduce_bit_depth(self.image, bits_per_channel=2)
        self.assertEqual(reduced.shape, self.image.shape)
        self.assertTrue(set(np.unique(reduced)).issubset({0, 85, 170, 255}))


if __name__ == "__main__":
    unittest.main()
