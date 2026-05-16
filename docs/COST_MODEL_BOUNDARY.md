# Cost Model Boundary

## Executive Summary

TurboQuant Compressor does not reduce Anthropic or Claude Code token billing directly.

Claude Code prompt caching is controlled by Anthropic API / Claude Code behavior.
TurboQuant-style compression applies to local vector stores or self-hosted inference runtimes where the operator controls KV-cache/vector representation.

## Feature Request #19436 Implementation

This project implements feature request #19436 as **Cost-Aware Brain** - an external planning and retrieval layer, not as internal Claude Code TTL modification.

The Cost-Aware Brain:
- Analyzes Claude Code usage patterns via ccusage or manual input
- Classifies context blocks by volatility (static/session/dynamic/context_pack_candidate)
- Builds compressed local context packs for large stable content
- Provides cache hygiene recommendations
- Does NOT modify Claude Code server-side prompt caching

## Three Layers

### Layer A — Claude Code / Anthropic API Prompt Caching

Relevant concepts:
- cache_control
- cache TTL
- cache write tokens
- cache read tokens
- stable prompt prefixes
- context size
- tool/MCP overhead

### Layer B — Local MCP Vector Compression

This project lives here.
It compresses local vectors/context packs and serves relevant chunks through MCP.

### Layer C — Self-Hosted Inference KV-Cache Compression

TurboQuant KV-cache compression belongs here.
This requires control of the inference runtime, such as vLLM/SGLang/custom serving.

## Non-Goals

- Does not inject TurboQuant into Anthropic-hosted Claude.
- Does not alter Claude Code server-side prompt caching.
- Does not guarantee Claude Code token savings.
- Does not implement paper-exact TurboQuant.
- Does not store QJL payload in LEVEL_0.