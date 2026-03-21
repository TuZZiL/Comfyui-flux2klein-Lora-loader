"""
Strength scheduling profiles for per-step LoRA control.

Each schedule defines control points (start_percent, strength_multiplier).
These are interpolated into HookKeyframes for ComfyUI's native hook system.

start_percent: 0.0 = first step, 1.0 = last step
strength_multiplier: 0.0 = LoRA off, 1.0 = full LoRA
"""

import comfy.hooks

SCHEDULE_PROFILES = {
    "Constant": [
        (0.0, 1.0),
    ],
    "Fade Out": [
        (0.0, 1.0),
        (0.33, 0.7),
        (0.66, 0.3),
        (1.0, 0.0),
    ],
    "Fade In": [
        (0.0, 0.0),
        (0.33, 0.3),
        (0.66, 0.7),
        (1.0, 1.0),
    ],
    "Strong Start": [
        (0.0, 1.0),
        (0.25, 0.5),
        (0.50, 0.2),
        (0.75, 0.0),
    ],
    "Pulse": [
        (0.0, 0.3),
        (0.25, 1.0),
        (0.75, 1.0),
        (1.0, 0.3),
    ],
}

SCHEDULE_NAMES = list(SCHEDULE_PROFILES.keys())


def _lerp(a, b, t):
    """Linear interpolation between a and b at t (0-1)."""
    return a + (b - a) * t


def build_keyframes(schedule_name, num_keyframes=5):
    """
    Build a HookKeyframeGroup from a schedule profile.

    Interpolates the profile's control points into num_keyframes evenly-spaced
    keyframes for smooth transitions during sampling.

    Returns: comfy.hooks.HookKeyframeGroup
    """
    profile = SCHEDULE_PROFILES.get(schedule_name)
    if not profile:
        profile = SCHEDULE_PROFILES["Constant"]

    # For Constant, just one keyframe is enough
    if len(profile) == 1:
        kf_group = comfy.hooks.HookKeyframeGroup()
        kf_group.add(comfy.hooks.HookKeyframe(
            strength=profile[0][1], start_percent=0.0, guarantee_steps=1
        ))
        return kf_group

    # Interpolate profile control points into num_keyframes evenly-spaced points
    kf_group = comfy.hooks.HookKeyframeGroup()

    for i in range(num_keyframes):
        pct = i / max(1, num_keyframes - 1)

        # Find surrounding control points
        strength = profile[-1][1]  # default to last
        for j in range(len(profile) - 1):
            p0, s0 = profile[j]
            p1, s1 = profile[j + 1]
            if p0 <= pct <= p1:
                t = (pct - p0) / (p1 - p0) if p1 > p0 else 0
                strength = _lerp(s0, s1, t)
                break

        kf_group.add(comfy.hooks.HookKeyframe(
            strength=round(strength, 4),
            start_percent=round(pct, 4),
            guarantee_steps=1,
        ))

    return kf_group
