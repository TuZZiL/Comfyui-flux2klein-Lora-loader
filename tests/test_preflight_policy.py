import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from preflight_policy import build_loader_hint, build_multi_advice, build_single_advice  # noqa: E402


def make_analysis(db_img=1.0, db_txt=1.0, sb_values=None):
    sb_values = sb_values or [1.0] * 24
    return {
        "db": {i: {"img": db_img, "txt": db_txt} for i in range(8)},
        "sb": {i: sb_values[i] for i in range(24)},
        "rank": 16,
        "alpha": None,
        "layer_stats": [],
    }


def make_compat(matched=12, total=12, incomplete=0):
    return {
        "total_modules": total,
        "matched_modules": matched,
        "skipped_modules": max(total - matched, 0),
        "incomplete_modules": incomplete,
        "sample_skipped": [],
        "sample_incomplete": [],
        "matched_module_bases": [],
        "skipped_module_bases": [],
        "complete_module_bases": [],
    }


class PreflightPolicyTests(unittest.TestCase):
    def test_late_heavy_profile_prefers_body_preservation(self):
        sb = [0.75] * 8 + [0.95] * 8 + [1.75] * 8
        advice = build_single_advice(
            make_analysis(db_img=0.95, db_txt=0.95, sb_values=sb),
            make_compat(),
            use_case="Edit",
            source_name="late.safetensors",
        )
        self.assertEqual(advice["recommended_edit_mode"], "Preserve Body")
        self.assertLess(advice["recommended_strength"], 0.9)
        self.assertIn("late-heavy", advice["profile_tags"])

    def test_style_heavy_profile_prefers_style_only(self):
        sb = [0.92] * 24
        advice = build_single_advice(
            make_analysis(db_img=1.35, db_txt=0.95, sb_values=sb),
            make_compat(),
            use_case="Edit",
            source_name="style.safetensors",
        )
        self.assertEqual(advice["recommended_edit_mode"], "Style Only")
        self.assertGreater(advice["recommended_strength"], 0.9)
        self.assertIn("style-heavy", advice["profile_tags"])

    def test_partial_compatibility_produces_warning(self):
        advice = build_single_advice(
            make_analysis(),
            make_compat(matched=5, total=12, incomplete=2),
            use_case="Edit",
            source_name="partial.safetensors",
        )
        self.assertEqual(advice["compat_status"], "partial")
        self.assertIn("partial", advice["report"])
        self.assertTrue(any("incomplete" in warning.lower() for warning in advice["warnings"]))

    def test_loader_hint_returns_style_verdict_and_apply_payload(self):
        sb = [0.92] * 24
        hint = build_loader_hint(
            make_analysis(db_img=1.35, db_txt=0.95, sb_values=sb),
            use_case="Edit",
            source_name="style.safetensors",
        )
        self.assertIn("Style-dominant", hint["verdict"])
        self.assertEqual(hint["recommended_edit_mode"], "Style Only")
        self.assertEqual(hint["apply"]["edit_mode"], "Style Only")
        self.assertGreaterEqual(hint["apply"]["protection"], 0.35)

    def test_multi_overlap_scales_active_slots_and_preserves_inactive_slots(self):
        slot_a = {
            "enabled": True,
            "lora": "a.safetensors",
            "strength": 1.2,
            "use_case": "Edit",
            "edit_mode": "Raw",
            "balance": 0.5,
        }
        slot_b = {
            "enabled": True,
            "lora": "b.safetensors",
            "strength": 1.1,
            "use_case": "Edit",
            "edit_mode": "Raw",
            "balance": 0.5,
        }
        slot_c = {
            "enabled": False,
            "lora": "c.safetensors",
            "strength": 0.8,
            "use_case": "Edit",
            "edit_mode": "Raw",
            "balance": 0.5,
        }
        entries = [
            {
                "index": 0,
                "slot": slot_a,
                "active": True,
                "advice": build_single_advice(make_analysis(sb_values=[0.75] * 8 + [0.95] * 8 + [1.6] * 8), make_compat(), source_name="a.safetensors"),
            },
            {
                "index": 1,
                "slot": slot_b,
                "active": True,
                "advice": build_single_advice(make_analysis(sb_values=[0.8] * 8 + [1.0] * 8 + [1.55] * 8), make_compat(), source_name="b.safetensors"),
            },
            {
                "index": 2,
                "slot": slot_c,
                "active": False,
                "advice": {},
            },
        ]

        multi = build_multi_advice(entries, use_case="Edit", source_name="multi")
        recommended = multi["recommended_slots"]

        self.assertEqual(multi["active_slot_count"], 2)
        self.assertEqual(recommended[2]["strength"], 0.8)
        self.assertLess(recommended[0]["strength"], slot_a["strength"])
        self.assertLess(recommended[1]["strength"], slot_b["strength"])
        self.assertIn("stacked", multi["report"])


if __name__ == "__main__":
    unittest.main()

