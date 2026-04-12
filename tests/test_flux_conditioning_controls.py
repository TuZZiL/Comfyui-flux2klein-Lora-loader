import copy
import sys
import unittest
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from conditioning_reference import apply_masked_reference_mix, gaussian_blur_per_channel, mix_reference_latent, rebalance_reference_appearance  # noqa: E402
from flux_conditioning_controls import (  # noqa: E402
    Flux2KleinColorAnchor,
    Flux2KleinMaskRefController,
    Flux2KleinRefLatentController,
    Flux2KleinStructureLock,
    Flux2KleinTextRefBalance,
    _apply_mask_to_reference_latent,
    _reference_token_span,
)


class FakeModel:
    def __init__(self):
        self.model_options = {}
        self.attn_patch = None

    def clone(self):
        return copy.deepcopy(self)

    def set_model_attn1_patch(self, fn):
        self.attn_patch = fn


class FluxConditioningControlsTests(unittest.TestCase):
    def test_reference_token_span_maps_selected_reference(self):
        span = _reference_token_span({"reference_image_num_tokens": [2, 3, 4]}, 1)
        self.assertEqual(span["seq_start"], -7)
        self.assertEqual(span["seq_end"], -4)
        self.assertEqual(span["num_ref_tokens"], 3)

    def test_mask_helper_only_affects_selected_channel_band(self):
        ref = torch.ones(1, 128, 2, 2)
        mask = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])

        modified = _apply_mask_to_reference_latent(
            ref,
            mask,
            strength=1.0,
            invert_mask=False,
            feather=0,
            channel_mode="low",
        )

        self.assertTrue(torch.allclose(modified[:, 64:, :, :], ref[:, 64:, :, :]))
        self.assertEqual(float(modified[:, :64, 0, 1].sum().item()), 0.0)

    def test_mask_controller_updates_reference_latents(self):
        node = Flux2KleinMaskRefController()
        conditioning = [
            (
                torch.zeros(1, 4, 8),
                {"reference_latents": [torch.ones(1, 128, 2, 2)]},
            )
        ]
        mask = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])

        (out,) = node.apply_mask(
            conditioning,
            mask,
            strength=1.0,
            invert_mask=False,
            feather=0,
            channel_mode="high",
        )

        updated = out[0][1]["reference_latents"][0]
        self.assertTrue(torch.allclose(updated[:, :64, :, :], torch.ones(1, 64, 2, 2)))
        self.assertEqual(float(updated[:, 64:, 0, 1].sum().item()), 0.0)

    def test_ref_latent_controller_scales_only_selected_reference_tokens(self):
        node = Flux2KleinRefLatentController()
        model = FakeModel()
        conditioning = [
            (
                torch.zeros(1, 8, 4),
                {"reference_latents": [torch.ones(1, 128, 2, 2)]},
            )
        ]

        model_out, _ = node.control(
            model,
            conditioning,
            strength=2.0,
            reference_index=1,
        )

        patch = model_out.attn_patch
        self.assertIsNotNone(patch)
        q = torch.zeros(1, 1, 5, 4)
        k = torch.ones(1, 1, 5, 4)
        v = torch.ones(1, 1, 5, 4)
        patched = patch(q, k, v, extra_options={"reference_image_num_tokens": [2, 3], "block_index": 0})
        self.assertTrue(torch.allclose(patched["k"][:, :, :2, :], torch.ones(1, 1, 2, 4)))
        self.assertTrue(torch.allclose(patched["k"][:, :, 2:, :], torch.full((1, 1, 3, 4), 2.0)))

    def test_ref_latent_controller_can_target_all_reference_spans(self):
        node = Flux2KleinRefLatentController()
        model = FakeModel()
        conditioning = [
            (
                torch.zeros(1, 8, 4),
                {"reference_latents": [torch.ones(1, 128, 2, 2), torch.ones(1, 128, 2, 2) * 2]},
            )
        ]

        model_out, _ = node.control(
            model,
            conditioning,
            strength=1.5,
            reference_index=-1,
        )

        patch = model_out.attn_patch
        q = torch.zeros(1, 1, 5, 4)
        k = torch.ones(1, 1, 5, 4)
        v = torch.ones(1, 1, 5, 4)
        patched = patch(q, k, v, extra_options={"reference_image_num_tokens": [2, 3], "block_index": 0})
        self.assertTrue(torch.allclose(patched["k"][:, :, :2, :], torch.full((1, 1, 2, 4), 1.5)))
        self.assertTrue(torch.allclose(patched["k"][:, :, 2:, :], torch.full((1, 1, 3, 4), 1.5)))

    def test_ref_latent_controller_rebalances_selected_reference_only(self):
        node = Flux2KleinRefLatentController()
        model = FakeModel()
        ref_a = torch.ones(1, 128, 4, 4)
        ref_b = torch.arange(1, 1 + 1 * 128 * 4 * 4, dtype=torch.float32).reshape(1, 128, 4, 4)
        conditioning = [
            (
                torch.zeros(1, 8, 4),
                {"reference_latents": [ref_a.clone(), ref_b.clone()]},
            )
        ]

        _, out_cond = node.control(
            model,
            conditioning,
            strength=1.0,
            reference_index=1,
            appearance_scale=1.5,
            detail_scale=0.25,
            blur_radius=2,
            channel_mask_start=0,
            channel_mask_end=64,
        )

        updated = out_cond[0][1]["reference_latents"]
        self.assertTrue(torch.allclose(updated[0], ref_a))
        self.assertFalse(torch.allclose(updated[1], ref_b))

    def test_text_ref_balance_scales_text_and_reference_regions(self):
        node = Flux2KleinTextRefBalance()
        model = FakeModel()
        conditioning = [(
            torch.zeros(1, 8, 4),
            {"reference_latents": [torch.ones(1, 128, 2, 2)]},
        )]

        model_out, _ = node.balance_streams(model, conditioning, balance=0.25)
        patch = model_out.attn_patch
        q = torch.zeros(1, 1, 4, 4)
        k = torch.ones(1, 1, 4, 4)
        v = torch.ones(1, 1, 4, 4)
        patched = patch(q, k, v, extra_options={"img_slice": [2, 4], "reference_image_num_tokens": [2], "block_index": 1})
        self.assertTrue(torch.allclose(patched["k"][:, :, :2, :], torch.full((1, 1, 2, 4), 0.5)))
        self.assertTrue(torch.allclose(patched["k"][:, :, 2:, :], torch.ones(1, 1, 2, 4)))

    def test_text_ref_balance_latent_mix_updates_reference_latents(self):
        node = Flux2KleinTextRefBalance()
        model = FakeModel()
        conditioning = [(
            torch.zeros(1, 8, 4),
            {"reference_latents": [torch.ones(1, 128, 2, 2), torch.ones(1, 128, 2, 2) * 2]},
        )]

        model_out, out_cond = node.balance_streams(
            model,
            conditioning,
            balance=1.0,
            balance_mode="latent_mix",
            target_reference_index=-1,
            replace_mode="zeros",
        )

        self.assertIsNotNone(model_out.attn_patch)
        updated = out_cond[0][1]["reference_latents"]
        self.assertTrue(torch.count_nonzero(updated[0]) == 0)
        self.assertTrue(torch.count_nonzero(updated[1]) == 0)

    def test_mask_controller_can_mix_reference_latents(self):
        node = Flux2KleinMaskRefController()
        conditioning = [
            (
                torch.zeros(1, 4, 8),
                {"reference_latents": [torch.ones(1, 4, 2, 2), torch.ones(1, 4, 2, 2) * 2]},
            )
        ]
        mask = torch.ones(1, 2, 2)

        (out,) = node.apply_mask(
            conditioning,
            mask,
            strength=1.0,
            mask_action="mix",
            reference_keep=0.0,
            replace_mode="zeros",
            target_reference_index=-1,
            channel_mode="all",
        )

        updated = out[0][1]["reference_latents"]
        self.assertTrue(torch.count_nonzero(updated[0]) == 0)
        self.assertTrue(torch.count_nonzero(updated[1]) == 0)

    def test_reference_helpers_support_mix_and_rebalance(self):
        ref = torch.arange(1, 1 + 1 * 4 * 4 * 4, dtype=torch.float32).reshape(1, 4, 4, 4)
        mixed = mix_reference_latent(
            ref,
            reference_keep=0.0,
            replace_mode="channel_mean",
            channel_start=0,
            channel_end=4,
            spatial_fade="none",
            spatial_fade_strength=0.0,
        )
        balanced = rebalance_reference_appearance(
            ref,
            appearance_scale=1.25,
            detail_scale=0.25,
            blur_radius=2,
            channel_start=0,
            channel_end=4,
        )
        masked = apply_masked_reference_mix(
            ref,
            torch.ones(1, 4, 4),
            strength=1.0,
            reference_keep=0.0,
            replace_mode="zeros",
            invert_mask=False,
            feather=0,
            channel_mode="all",
        )

        self.assertEqual(tuple(mixed.shape), tuple(ref.shape))
        self.assertEqual(tuple(balanced.shape), tuple(ref.shape))
        self.assertEqual(tuple(masked.shape), tuple(ref.shape))
        self.assertFalse(torch.allclose(mixed, ref))
        self.assertFalse(torch.allclose(balanced, ref))
        self.assertTrue(torch.count_nonzero(masked) == 0)

    def test_color_anchor_adds_sampler_hook(self):
        node = Flux2KleinColorAnchor()
        model = FakeModel()
        conditioning = [(
            torch.zeros(1, 8, 4),
            {"reference_latents": [torch.full((1, 128, 2, 2), 3.0)]},
        )]

        model_out = node.apply(model, conditioning, strength=0.5, ramp_curve=1.0)[0]
        hooks = model_out.model_options.get("sampler_post_cfg_function", [])
        self.assertEqual(len(hooks), 1)

        denoised = torch.zeros(1, 128, 2, 2)
        result = hooks[0]({"denoised": denoised, "sigma": torch.tensor(1.0)})
        self.assertGreater(float(result.mean().item()), 0.0)

    def test_color_anchor_can_average_multiple_references(self):
        node = Flux2KleinColorAnchor()
        model = FakeModel()
        conditioning = [(
            torch.zeros(1, 8, 4),
            {"reference_latents": [torch.zeros(1, 128, 2, 2), torch.ones(1, 128, 2, 2)]},
        )]

        model_out = node.apply(model, conditioning, strength=0.5, ramp_curve=1.0, ref_index=-1)[0]
        hooks = model_out.model_options.get("sampler_post_cfg_function", [])
        self.assertEqual(len(hooks), 1)
        result = hooks[0]({"denoised": torch.zeros(1, 128, 2, 2), "sigma": torch.tensor(1.0)})
        self.assertGreater(float(result.mean().item()), 0.0)
        self.assertLess(float(result.mean().item()), 1.0)

    def test_structure_lock_moves_low_frequency_toward_reference(self):
        node = Flux2KleinStructureLock()
        model = FakeModel()
        ref = torch.tensor(
            [
                [
                    [
                        [0.0, 0.0, 1.0, 1.0],
                        [0.0, 0.0, 1.0, 1.0],
                        [0.5, 0.5, 1.5, 1.5],
                        [0.5, 0.5, 1.5, 1.5],
                    ],
                    [
                        [0.0, 0.0, 1.0, 1.0],
                        [0.0, 0.0, 1.0, 1.0],
                        [0.5, 0.5, 1.5, 1.5],
                        [0.5, 0.5, 1.5, 1.5],
                    ],
                    [
                        [0.0, 0.0, 1.0, 1.0],
                        [0.0, 0.0, 1.0, 1.0],
                        [0.5, 0.5, 1.5, 1.5],
                        [0.5, 0.5, 1.5, 1.5],
                    ],
                    [
                        [0.0, 0.0, 1.0, 1.0],
                        [0.0, 0.0, 1.0, 1.0],
                        [0.5, 0.5, 1.5, 1.5],
                        [0.5, 0.5, 1.5, 1.5],
                    ],
                ]
            ],
            dtype=torch.float32,
        )
        conditioning = [(
            torch.zeros(1, 8, 4),
            {"reference_latents": [ref.clone()]},
        )]

        model_out, _ = node.apply(
            model,
            conditioning,
            strength=1.0,
            blur_radius=1,
            ramp_start=0.0,
            ramp_end=1.0,
        )

        hooks = model_out.model_options.get("sampler_post_cfg_function", [])
        self.assertEqual(len(hooks), 1)

        denoised = torch.zeros_like(ref)
        result = hooks[0]({"denoised": denoised, "sigma": torch.tensor(1.0)})
        ref_low = gaussian_blur_per_channel(ref, 1)
        result_low = gaussian_blur_per_channel(result, 1)
        zero_low = gaussian_blur_per_channel(denoised, 1)

        ref_dist = torch.mean(torch.abs(result_low - ref_low)).item()
        zero_dist = torch.mean(torch.abs(zero_low - ref_low)).item()
        self.assertLess(ref_dist, zero_dist)

    def test_structure_lock_respects_mask_and_invert_mask(self):
        node = Flux2KleinStructureLock()
        model = FakeModel()
        ref = torch.ones(1, 4, 4, 4)
        conditioning = [(
            torch.zeros(1, 8, 4),
            {"reference_latents": [ref.clone()]},
        )]
        mask = torch.tensor([[[1.0, 1.0, 0.0, 0.0], [1.0, 1.0, 0.0, 0.0], [1.0, 1.0, 0.0, 0.0], [1.0, 1.0, 0.0, 0.0]]])

        model_out, _ = node.apply(
            model,
            conditioning,
            strength=1.0,
            blur_radius=1,
            ramp_start=0.0,
            ramp_end=1.0,
            mask=mask,
            invert_mask=False,
        )
        hook = model_out.model_options["sampler_post_cfg_function"][0]
        result = hook({"denoised": torch.zeros_like(ref), "sigma": torch.tensor(1.0)})
        left = result[:, :, :, :2].abs().mean().item()
        right = result[:, :, :, 2:].abs().mean().item()
        self.assertGreater(left, right)

        model_out_inv, _ = node.apply(
            model,
            conditioning,
            strength=1.0,
            blur_radius=1,
            ramp_start=0.0,
            ramp_end=1.0,
            mask=mask,
            invert_mask=True,
        )
        hook_inv = model_out_inv.model_options["sampler_post_cfg_function"][0]
        result_inv = hook_inv({"denoised": torch.zeros_like(ref), "sigma": torch.tensor(1.0)})
        left_inv = result_inv[:, :, :, :2].abs().mean().item()
        right_inv = result_inv[:, :, :, 2:].abs().mean().item()
        self.assertLess(left_inv, right_inv)


if __name__ == "__main__":
    unittest.main()
