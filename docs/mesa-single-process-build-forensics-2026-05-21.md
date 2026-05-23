# Mesa Single-Process Build Forensics 2026-05-21

Primary anchors:

- `/storage/emulated/0/wrapper/tz.txt`
- `docs/frontier-tracing-contract-2026-05-21.md`
- `docs/tz-frontier-capability-matrix-2026-05-21.md`
- local Mesa build dir: `/data/data/com.termux/files/home/mesa-upstream-26.2-devel/build-tq-zero`

## Rule

For `P0-P3` closure work, build cleanup is not allowed to run ad hoc.

Before each new Mesa/Rusticl build pass:

- corpus anchors must be reloaded
- official web corroboration must be checked for the active lane
- build-state warnings must be classified as:
  - build graph / ninja state
  - compile diagnostics
  - runtime capability drift

## Current build-state facts

Observed on single-process build start:

- `ninja: warning: premature end of file; recovering`

Local build-dir facts:

- `build.ninja` ends with a normal newline
- `.ninja_log` is present and parseable
- `.ninja_deps` was modified at the same time as the interrupted build pass
- the warning therefore must not be attributed to Rusticl source text by default

## Official-source corroboration

`ninja` upstream/Android source shows the exact warning is emitted when the parsed state file is shorter than the actual file and recovery truncates to the last full record.

Corroborating source:

- Android `external/ninja` diff with recovery path:
  - `*err = "premature end of file";`
  - truncate to `parsed_file_size`
- Ninja manual:
  - dependency and depfile state is tracked outside the textual `build.ninja`

Interpretation for this tree:

- the warning is consistent with interrupted or concurrently modified Ninja state
- until disproven, the primary suspect is `.ninja_deps` or another Ninja state database, not the generated build graph text

## Clean-log policy

The next build pass is only considered clean if:

- the build starts without the `premature end of file` warning
- there are no compiler warnings in touched Rusticl/TurboQuant frontier files
- there are no frontier overclaims that bypass source-gate or live-artifact rules

## Next-pass procedure

1. Preserve the current forensic finding in corpus.
2. Clear or rotate only the Ninja state files that are allowed to be regenerated.
3. Re-run the build in one process.
4. Triage compile diagnostics before any new frontier edits.
5. Re-run frontier evidence only after the build log is clean.
