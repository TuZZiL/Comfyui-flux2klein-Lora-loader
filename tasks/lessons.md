# Lessons

- Keep the preflight advisor schedule-free. The advisor should stay focused on compatibility, edit mode, balance, and starter strength only.
- Put advisory heuristics in a pure module first. That keeps the logic testable without ComfyUI runtime dependencies.
- When UX labels are already serialized in saved ComfyUI graphs, improve tooltips and validation first; postpone breaking renames to a later migration phase.
- Hidden state widgets like `slot_data` should preserve their original widget type when ComfyUI needs to serialize them into `widgets_values`; only purely decorative widgets should be converted to `converted-widget`.
- When a user-facing dial is semantically inverted, flip the math, the tooltip, and the log language together. Renaming only the label leaves the UI misleading.
- For companion conditioning docs, describe the actual surface being changed (`conditioning`, `model`, or `reference_latents`), then give starting values and a concrete workflow example. Parameter tables alone are not enough.
