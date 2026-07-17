import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from kmeans_color_studio.core import export_palette_css, export_palette_json, quantize_image


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


if __name__ == "__main__":
    unittest.main()
