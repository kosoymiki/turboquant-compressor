# Frontier Donor Model 2026-05-22

This is the canonical multi-lane donor model for the current TQ + Mesa frontier.

It exists to prevent lane-by-lane improvisation and to force every implementation campaign to start from an explicit donor map.

Canonical generator:

```sh
cd /data/data/com.termux/files/home/tmp_turboquant
node scripts/generate-frontier-donor-model.mjs
```

Artifact:

- [frontier-donor-model-2026-05-22.json](/data/data/com.termux/files/home/tmp_turboquant/forensics/frontier-donor-model-2026-05-22.json:1)

## Rule

Every active lane must have:

- local source donors
- local forensic donors
- external implementation donors
- external authority/spec donors
- contrast/negative donors when useful

No lane should start as a narrow bug chase without this donor model.

## Current High-Value Verdicts

`mesa_system_svm`
- strongest external donor: `PoCL`
- strongest local donor: `Rusticl + Freedreno/KGSL userptr path`
- strongest contrast donor: `clvk`
- strongest authority donor: OpenCL API spec

`mesa_command_buffer`
- strongest local donor: Rusticl `api/icd/core` lane
- strongest authority donor: Khronos command-buffer refpages
- RenderDoc donor bundle remains relevant for command recording, replay, and debugging discipline

`P1-P3 frontier queue`
- canonical queue contract now lives in [P1_P3_FRONTIER_QUEUE_2026-05-22.md](/data/data/com.termux/files/home/tmp_turboquant/docs/P1_P3_FRONTIER_QUEUE_2026-05-22.md:1)
- donor-backed backlog artifact now lives in `forensics/p1-p3-frontier-backlog-2026-05-22.json`
- every post-`P0` cluster must carry a donor mesh before implementation work starts

## Minimum Corpus Contract

The donor model must remain larger than a single repo or a single spec.

For current frontier work this means:

- local Mesa source
- local TQ source
- local forensics
- RenderDoc GPU-debug corpus
- GitHub donor repos
- official Khronos / kernel / Android docs

That donor spread is required for a “closed” engineering claim.
