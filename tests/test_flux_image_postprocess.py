import sys
import unittest
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import flux_image_postprocess as img_post  # noqa: E402
from flux_image_postprocess import (  # noqa: E402
    TuzKleinEditComposite,
    _auto_threshold_mad,
    _fill_holes,
    _grow_mask,
    _keep_largest_islands,
    _merge_custom_mask,
)


class FluxImagePostprocessTests(unittest.TestCase):
    def _image(self, value, size=32):
        arr = np.full((1, size, size, 3), value, dtype=np.float32)
        return torch.from_numpy(arr)

    def _mask(self, array):
        return torch.from_numpy(array.astype(np.float32))

    def test_fill_holes_fills_center_void(self):
        mask = np.zeros((9, 9), dtype=np.float32)
        mask[2:7, 2:7] = 1.0
        mask[4, 4] = 0.0
        filled = _fill_holes(mask)
        self.assertGreater(filled[4, 4], 0.5)

    def test_keep_largest_islands_keeps_primary_component(self):
        mask = np.zeros((12, 12), dtype=np.float32)
        mask[1:7, 1:7] = 1.0
        mask[9:11, 9:11] = 1.0
        kept = _keep_largest_islands(mask, 1)
        self.assertGreater(kept[3, 3], 0.5)
        self.assertEqual(float(kept[9:11, 9:11].sum()), 0.0)

    def test_grow_mask_expands_positive_and_shrinks_negative(self):
        mask = np.zeros((9, 9), dtype=np.float32)
        mask[4, 4] = 1.0
        grown = _grow_mask(mask, 1)
        shrunk = _grow_mask(np.pad(np.ones((5, 5), dtype=np.float32), 2), -1)
        self.assertGreater(grown.sum(), mask.sum())
        self.assertLess(shrunk.sum(), 9 * 9)

    def test_auto_threshold_uses_valid_region(self):
        diff = np.vstack([
            np.ones((12, 12), dtype=np.float32),
            np.full((12, 12), 50.0, dtype=np.float32),
        ])
        valid = np.vstack([
            np.ones((12, 12), dtype=np.float32),
            np.zeros((12, 12), dtype=np.float32),
        ])
        threshold = _auto_threshold_mad(diff, valid)
        self.assertGreater(threshold, 0.0)
        self.assertLess(threshold, 20.0)

    def test_merge_custom_mask_modes(self):
        base = np.array([[0.0, 1.0], [0.5, 0.5]], dtype=np.float32)
        custom = np.array([[1.0, 0.0], [1.0, 0.0]], dtype=np.float32)
        self.assertTrue(np.allclose(_merge_custom_mask(base, custom, "replace"), custom))
        self.assertTrue(np.allclose(_merge_custom_mask(base, custom, "add"), np.maximum(base, custom)))
        self.assertTrue(np.allclose(_merge_custom_mask(base, custom, "subtract"), np.clip(base * (1.0 - custom), 0.0, 1.0)))

    def test_node_returns_near_empty_mask_for_identical_images(self):
        node = TuzKleinEditComposite()
        image = self._image(0.0)
        composited, mask, report, debug_gallery = node.run(
            generated_image=image,
            original_image=image,
            delta_e_threshold=-1.0,
            flow_quality="medium",
            use_occlusion=False,
            occlusion_threshold=-1.0,
            noise_removal_pct=0.0,
            close_radius_pct=0.0,
            fill_holes=False,
            fill_borders=True,
            max_islands=0,
            grow_mask_pct=0.0,
            feather_pct=0.0,
            color_match_blend=0.0,
            poisson_blend_edges=False,
            custom_mask=None,
            custom_mask_mode="replace",
            enable_debug=False,
        )
        self.assertLess(float(mask.mean().item()), 1e-4)
        self.assertTrue(torch.allclose(composited, image, atol=1e-4))
        self.assertIsInstance(report, str)
        self.assertEqual(tuple(debug_gallery.shape), (1, 64, 64, 3))

    def test_node_replace_custom_mask_uses_supplied_region(self):
        node = TuzKleinEditComposite()
        original = self._image(0.0, size=24)
        generated = self._image(1.0, size=24)
        custom_mask = np.zeros((1, 24, 24), dtype=np.float32)
        custom_mask[0, 8:16, 8:16] = 1.0

        composited, mask, _, _ = node.run(
            generated_image=generated,
            original_image=original,
            delta_e_threshold=-1.0,
            flow_quality="medium",
            use_occlusion=False,
            occlusion_threshold=-1.0,
            noise_removal_pct=0.0,
            close_radius_pct=0.0,
            fill_holes=False,
            fill_borders=False,
            max_islands=0,
            grow_mask_pct=0.0,
            feather_pct=0.0,
            color_match_blend=0.0,
            poisson_blend_edges=False,
            custom_mask=self._mask(custom_mask),
            custom_mask_mode="replace",
            enable_debug=False,
        )

        self.assertGreater(float(mask[0, 12, 12].item()), 0.5)
        self.assertLess(float(mask[0, 2, 2].item()), 0.1)
        self.assertGreater(float(composited[0, 12, 12, 0].item()), 0.5)
        self.assertLess(float(composited[0, 2, 2, 0].item()), 0.1)

    def test_missing_cv2_raises_clear_error(self):
        old = img_post.cv2
        img_post.cv2 = None
        try:
            with self.assertRaises(RuntimeError):
                img_post._require_cv2()
        finally:
            img_post.cv2 = old


if __name__ == "__main__":
    unittest.main()
