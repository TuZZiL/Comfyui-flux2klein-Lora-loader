# Project Report

## 2026-04-24
- context: Reviewed codebase math/heuristics for LoRA layer scaling, edit/anatomy interpolation, conditioning reference mixing, schedules, and edit compositing.
- done: Identified main improvement candidate: mask reference `scale` and `mix` modes currently interpret mask polarity inconsistently.
- done: Improved `ColorAnchor` ramp behavior so the first sampler callback is a no-op until sigma/step progress exists; added safer variance weighting and channel-count handling.
- ideas: Consider adding robust norm normalization for AutoStrength/preflight and smoothing schedule/keyframe curves only if real samples show instability.
- resolved: `python -m py_compile flux_conditioning_controls.py conditioning_common.py` and pure policy/anatomy/preflight tests passed; conditioning tests remain blocked by missing `torch`.

## 2026-04-19
- context: Applied requested consistency pass for edit-mode naming, strength ranges, and anatomy profile semantics.
- done: Fixed anatomy `strict_zero` discontinuity so targeted blocks now fade with `anatomy_strength` (`multiplier = 1 - strength`) instead of always dropping to hard zero.
- done: Clarified docs/tooltips that `edit_mode` is the semantic/identity protection layer while `anatomy_profile` is a body/structure overlay; documented that both stack and should be combined only when needed.
- done: Added fast inline LoRA advisory payload (`build_loader_hint`) and Comfy route `/tuz_flux/loader_hint` for lightweight analyze/apply suggestions without the separate Preflight node.
- done: Added inline `Analyze` / `Apply` suggestion UI to single Loader badge row and per-slot expanded actions in Multi.
- resolved: Validation passed for advisory backend with `python -m unittest tests.test_preflight_policy tests.test_anatomy_profiles` and `python -m py_compile preflight_policy.py flux_preflight_advisor.py`.
- resolved: Added/updated anatomy tests to cover interpolated `strict_zero` behavior and validated with `python -m unittest tests.test_anatomy_profiles`.
- done: Switched canonical no-protection edit mode from `None` to `Raw` across loader pipeline, UI lists, docs, and tests.
- done: Kept backward compatibility for legacy workflows by accepting `None` as a runtime alias of `Raw`.
- done: Unified LoRA strength ranges to `-3..+3` in Loader, Scheduled, Multi UI, and Composer UI; added backend clamping in Python slot normalization/entrypoints.
- done: Updated anatomy naming for variant A (`Balanced Structure`) and kept `Balanced Identity` alias support for old slot payloads.
- done: Smoothed `Preserve Body` early single-block transition to remove abrupt jump around sb3/sb4.
- done: Corrected auto-protection formula comment to match real output range.
- resolved: Targeted validation passed: `python -m unittest tests.test_lora_compat tests.test_composer_policy tests.test_preflight_policy tests.test_anatomy_profiles tests.test_node_json_contracts` (48 tests).
- blocked: Full suite run is limited in this environment because optional deps (`torch`, `numpy`) are missing for two test modules.

## 2026-04-17
- context: Implemented Auto v2 for `TUZ FLUX LoRA Loader`.
- done: Added Loader-only `auto_bias` and `auto_tune` controls with backward-compatible Auto behavior.
- done: Added Auto decision metadata flow (`reason_code`, `reason_label`, metrics, bias/tune values).
- done: Added `flux_lora.auto_decision` UI event and compact Auto decision badge in the graph widget.
- done: Updated EN/UA docs for the new Auto controls and badge behavior.
- done: Added unit tests for neutral parity, bias/tune shifts, invalid fallback, metadata contract, and manual-mode invariance.
- resolved: Validation passed with `python -m unittest discover -s tests -p "test_*.py"` (`69 tests`).

## 2026-04-16
- context: Reviewed README clarity for end users.
- done: Assessed README structure, onboarding flow, and terminology density.
- done: Implemented beginner-friendly README refresh in `README.md` and `README_UA.md`.
- resolved: Added start-here flow, screenshot, glossary, beginner recipes, troubleshooting, and optional advanced reference labels.
- resolved: Clarified loader `protection` vs `balance` and the separate `Text/Ref Balance` control.
- done: Fixed `Structure Lock` device mismatch by moving reference latent onto the sampler tensor device before blending.
- resolved: `apply_structure_lock` now works with CUDA `denoised` tensors and CPU-origin reference latents.
- next: If desired, do a final read-through on GitHub-rendered markdown for spacing/visual balance.

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
- resolved: Fixed visual misalignment in Multi anatomy row by unifying `Strict zero` and `Custom JSON` geometry/typography (same height and baseline rhythm).
- done: Performed focused UI quality review for `FluxLoraMulti` card widget (`js/flux_lora_multi.js`) with usability/clarity/risk findings and improvement priorities.
- done: Rebranded project metadata for Comfy Manager submission to `TUZ FluxKlein Toolkit` (`pyproject` DisplayName + normalized package name + README titles).
- done: Updated repository/documentation/issue URLs and install clone commands to renamed repo `https://github.com/TuZZiL/tuz-fluxklein-toolkit`.
- done: Aligned Loader docs to new `protection` naming with explicit legacy `balance` compatibility notes (EN/UA).
- resolved: Unified protection field naming across Loader/Multi/Scheduled (`protection` primary, legacy `balance` accepted for backward compatibility in Python + Multi JS slot contract).
