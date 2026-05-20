# Android Evidence Pack 20260515 Recovery Value

- source pack: `/storage/emulated/0/Pictures/Screenshots/evidence`
- imported corpus pack id: `android_evidence_pack_20260515`
- role: `historical_forensic_lane`

## Why this pack matters

- It preserves a real intermediate truth chain from the earlier Mesa/Rusticl/KGSL recovery line.
- It proves that the old project did not fail at trivial API bring-up; it reached device enumeration, context creation, program build, and `GPU_COMMAND` submission.
- It also records why that line was formally closed at the time: transfer visibility and compute execution proof did not become production-safe.

## Hard recovery conclusions from this pack

- `release-truth-lock.json` captures a previous stable truth state where `vendor OpenCL` was treated as the only production backend and `Rusticl` was closed as research-only.
- `rusticl-build-evidence.json` proves a successful Mesa/Rusticl build against Mesa `26.1.0`, including explicit KGSL API rebases.
- `mesa-turnip-build-evidence.json` proves a successful Turnip build and records which KGSL files were manually rebased.
- `patch-42R-final-gpu-submit.json` shows the project reached accepted GPU submission through KGSL for Rusticl.
- `patch-43R-decision-gate-close.json` shows why that path was not considered production-ready: kernel submission success was not equal to visible compute execution.
- `device-identity.json` preserves an older device/access truth: KGSL accessible from Termux, DRM render node present but not accessible.

## How this changes current recovery work

- We must not repeat old false simplifications such as "GPU submit success means Rusticl is done".
- We must distinguish:
  - build success
  - device enumeration
  - queue submission
  - memory visibility
  - end-to-end parity
- The imported pack is historical evidence, not current production truth.
- Current custom-stack-first policy remains correct, but historical `vendor OpenCL` truth must stay visible because it explains previous engineering decisions.

## Practical use in the current project

- Run every search, scan, comparison, donor lookup, and file-to-file recovery pass through the TurboQuant/corpus lane first.
- Treat direct shell search and direct web search as corroboration-only after the first TurboQuant/corpus pass.
- Use `release-truth-lock.json` and `rusticl-kgsl-final-decision.json` as guardrails when evaluating new Mesa 26.2+ recovery claims.
- Use `mesa-turnip-build-evidence.json` and `rusticl-build-evidence.json` to recover exact old build assumptions and KGSL/API rebases.
- Use `patch-40R` to `patch-43R` artifacts as a staged failure/learning chain for the rebuilt driver mesh.
- Do not overwrite current forensics with this pack; keep it as a historical lane parallel to:
  - `forensics/adb-driver-truth.json`
  - `forensics/mesa/driver-mesh-recovery-ready.json`

## Current stance after import

1. Historical evidence confirms there was much more than `0001-0003`.
2. Historical evidence also confirms some previous conclusions were tied to Mesa `26.1.0` + older build assumptions.
3. Fresh recovery against Mesa `26.2.0-devel` must preserve old lessons, not old backend truth.
