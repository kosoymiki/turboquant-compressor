# RenderDoc GPU Debug Corpus 2026-05-22

## Scope

New donor corpus ingested from:

- `/storage/emulated/0/Download/renderdoc-gpu-debug-portable.tar.gz`

Canonical ingest artifact:

- `forensics/renderdoc-gpu-debug-corpus-ingest.json`

## Archive Truth

- archive size: `50,396,556` bytes
- archive sha256: `d541ee3f952efeb7c26ab70ef39e75dde03a69735070212083fffa2ef3ca2ac2`
- extracted root: `/data/data/com.termux/files/home/.cache/turboquant/renderdoc-gpu-debug-portable/renderdoc-gpu-debug`
- extracted size: `68,604,458` bytes
- extracted files: `140`

Extension split:

- `.json`: `82`
- `.pdf`: `20`
- `.html`: `19`
- `.py`: `7`
- `.md`: `3`
- `.txt`: `3`
- `.sh`: `2`
- `.c`: `1`
- `.h`: `1`

## Authority Layer

The bundle already carries its own explicit authority rule:

`Official specs > kernel source > Mesa source > device evidence > academic papers > community reports`

That matches the TQ+Mesa forensic contract and is accepted as compatible donor policy.

Primary local authority entries:

- `docs/research/CORPUS_MASTER.md`
- `docs/research/source-index.json`
- `docs/research/book-corpus-index.json`

## MCP / Tool Surface

Legacy MCP server root:

- `mcp_server/server.py`
- `mcp_server/rdc_runner.py`
- `mcp_server/recipes.py`

Exact tool entries discovered:

- `rdc_session`
- `rdc_overview`
- `rdc_draws`
- `rdc_events`
- `rdc_pipeline`
- `rdc_shader`
- `rdc_export`
- `rdc_pixel`
- `rdc_diff`
- `rdc_resources`
- `rdc_shader_edit`
- `rdc_capture`
- `rdc_command`
- `adb_gpu_kgsl`
- `adb_gpu_sync`
- `adb_gpu_vulkan`
- `adb_gpu_opencl`
- `adb_gpu_mesa`
- `adb_gpu_perf`
- `adb_gpu_capture`
- `gpu_corpus_search`

Recipe entries discovered:

- `INVISIBLE_OBJECT`
- `COMPUTE_WRONG_OUTPUT`
- `SYNC_STALE_DATA`
- `VISUAL_ARTIFACT`
- `PERFORMANCE`
- `COMPARE_FRAMES`
- `DEBUG_PIXEL`

Engineering meaning:

- this is not only a RenderDoc wrapper
- it is an older full-stack GPU-debug orchestrator spanning RenderDoc, ADB, KGSL, Vulkan, OpenCL, Mesa, sync, and performance

## Corpus Payload

Curated source index:

- sources total: `53`
- `P0`: `40`
- `P1`: `13`

High-value categories:

- `gpu_api_spec`: `8`
- `kernel_driver`: `2`
- `android_graphics`: `2`
- `mesa_driver`: `2`
- `opencl_research`: `4`
- `gpu_architecture`: `4`

Book corpus payload:

- extracted books total: `91`
- GPU-relevant extracted books: `53`
- extracted text bytes: `40,639,488`

## Frontier Donor Relevance

### `mesa_command_buffer`

Accepted donor sources:

- `opencl-spec`
- `vulkan-spec`
- `linux-drm-docs`
- `mesa-rusticl-docs`
- `mesa-freedreno-docs`
- `xdc-freedreno-android`

Accepted local donor entries:

- `references/commands-quick-ref.md`
- `docs/research/source-index.json`
- `docs/research/CORPUS_MASTER.md`
- `docs/research/corpus/html/vkCmdDispatch.html`
- `docs/research/corpus/html/vk_synchronization.html`
- `docs/research/corpus/pdf/OpenCL_API_3.0.pdf`
- `docs/research/corpus/pdf/vkspec_1.4.pdf`

Engineering use:

- command recording / replay semantics
- queue-submit / sync dependency modeling
- call-surface mapping for command buffer entrypoints
- donor expectations for mutable replay and synchronization behavior

### `mesa_system_svm`

Accepted donor sources:

- `opencl-spec`
- `linux-drm-docs`
- `android-sync-framework`
- `kgsl-uapi`
- `mesa-rusticl-docs`
- `qualcomm-adreno-opencl`

Accepted local donor entries:

- `docs/research/corpus/c/msm_kgsl.h`
- `docs/research/corpus/c/kgsl_sync.c`
- `docs/research/corpus/html/kernel_dmabuf.html`
- `docs/research/corpus/html/kernel_syncfile.html`
- `docs/research/corpus/html/android_sync_framework.html`
- `docs/research/corpus/pdf/OpenCL_API_3.0.pdf`

Engineering use:

- lower capability boundaries between VM-backed coarse SVM and true fine/system SVM
- KGSL-side memory/sync substrate context
- donor comparison against vendor OpenCL and foreign Mesa drivers

## Decision

This bundle is accepted as a first-class donor corpus for:

- `mesa_command_buffer`
- `mesa_system_svm`
- broader GPU debug / sync / KGSL / Vulkan / OpenCL forensics

It is not production truth by itself.

It is donor truth plus research corpus, and must remain below:

- official Khronos specs
- kernel source / docs
- local Mesa source
- live device evidence
