# Project Report

## 2026-04-15
- context: Start of session.
- done: Initialized project memory file.
- next: Capture concrete task context when work starts.
- done: Investigated Flux Multi state-loss after workflow JSON transfer between machines.
- resolved: Added persistence fallback for Multi slot state in `node.properties.slot_data_json` and restore logic in `js/flux_lora_multi.js` (`onConfigure` + `onSerialize`).
- context: Root risk is hidden-widget (`slot_data`) serialization inconsistency across ComfyUI installs; fallback now keeps selected LoRAs/values recoverable.
- done: Added anatomy controls to `FluxLoraMulti` expanded cards (`anatomy_profile`, `anatomy_strength`, `anatomy_strict_zero`, custom JSON editor for `Custom` profile).
- done: Added compact anatomy status hint per slot and profile-list loading from `FluxLoraLoader` object info with local fallback values.
- resolved: Multi UI now exposes anatomy slot fields already supported by backend `slot_data` contract.
