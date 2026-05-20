# Corpus Verification Rule

This repository treats corpus material as a lead, not as final truth.

## Mandatory Rule

Every claim derived from corpus material must be re-verified before it is used for:

- code changes
- release documentation
- benchmark interpretation
- runtime policy decisions
- public performance or scientific claims

## Accepted Second Source

A corpus-derived claim is only usable after confirmation by at least one of:

1. live repo behavior
2. committed benchmark or forensic evidence
3. a primary external source

Primary external sources include:

- official documentation
- original research papers
- upstream source repositories

## Conflict Resolution

- If corpus disagrees with live code, prefer live code.
- If corpus disagrees with committed evidence, prefer the newer and better-scoped evidence.
- If corpus disagrees with a primary source, prefer the primary source.
- If none of the above are decisive, mark the claim unresolved and do not publish it as release truth.

## Practical Workflow

1. Extract the claim from corpus material.
2. Find the exact file, artifact, or runtime path in this repo that should prove or falsify it.
3. If the repo is insufficient, fetch a primary external source.
4. Record the claim as one of:
   - verified
   - contradicted
   - stale
   - unresolved
5. Only then use it in code, docs, or release gating.

## Current Verified Examples

- GitHub release-facing repo guidance must come from GitHub Docs, not repo folklore.
- SemVer version bump policy must come from SemVer itself, not remembered convention.
- TurboQuant/QJL claims must be checked against the original papers and current repo behavior.
