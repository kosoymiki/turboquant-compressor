# TurboQuant Driver Debug Forensics Playbook

This is the standalone version of the corpus driver debug contract.

## Required Flow

1. primary truth mesh retrieval
2. TurboQuant preflight and retrieval
3. corpus and prior evidence review
4. local Android forensic trace review
5. web corroboration
6. source or runtime change
7. donor comparison
8. post-change forensic replay

## TQ Gate

No plain read or edit is allowed before TurboQuant classification and corpus retrieval.

Required order:

1. TurboQuant classification of the target question, file, or artifact
2. corpus retrieval or context-pack retrieval
3. corroborating read
4. edit or experiment

## Truth Priority

- old working runtime evidence packs
- current live device traces
- official implementation contracts and specs
- fresh external donors and workload references
- books and curated corpus references

## Required Evidence

- primary truth mesh manifests and artifacts
- TurboQuant read/edit justification packet
- ADB truth
- bugreport
- dumpsys
- logcat
- getprop
- package inventory
- mounts
- services
- selinux
- process state
- device nodes
- runtime packs
- donor release unpack
- comparative runtime report

## Comparative Rule

- `Turnip/Vulkan`: compare old working Turnip/Freedreno reports first, then proprietary donor releases for Vulkan packaging, loader, and linkage behavior.
- `Rusticl/OpenCL`: compare old working `android_evidence_pack_20260515` artifacts first, then compute-focused donors and our own `mesa_rusticl_kgsl` runtime traces.

## Primary Truth Mesh

- `/data/data/com.termux/files/home/corpus-control-plane/raw/forensics/android_evidence_pack_20260515`
- `/data/data/com.termux/files/home/corpus-control-plane/mcp/turboquant/forensics`
- `/data/data/com.termux/files/home/corpus-control-plane/raw/books/opencl`

## Non-Negotiable

No direct patching from intuition alone.
No web-first debugging.
No build-first workflow before forensic review and source-zero closure.
