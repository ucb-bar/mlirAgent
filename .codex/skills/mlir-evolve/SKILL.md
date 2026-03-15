---
name: mlir-evolve
description: MLIR/LLVM/IREE workflow for the mlirEvolve repo. Use when working in this repo to build or configure toolchains, create MLIR dialects, analyze/debug passes (provenance/rlm), generate kernels, run HW-in-the-loop perf tests, manage the Neo4j/LanceDB knowledge graph, or integrate XPU backends (GPU/NPU).
---

# MLIR Evolve

## Overview
Use this skill to navigate and extend the mlirEvolve toolchain and research workflows in a consistent, Codex-friendly way.

## Quick Start
- Read `references/repo-map.md` for a map of key directories and current tools.
- Reuse existing wrappers in `src/mlirAgent/tools/*.py` before creating new ones.
- When adding a tool, follow the conventions in `references/tool-conventions.md` and place it under the matching domain folder in `src/mlirAgent/tools/`.

## Core Capabilities

### 1) Build + Toolchain Management
- Use `src/mlirAgent/tools/build.py` for Ninja/CMake builds.
- When adding new build helpers, place them in `src/mlirAgent/tools/building/`.
- Prefer Config-driven paths and flags (see `references/infra.md`).

### 2) Dialect + TableGen Workflows
- Create dialect tooling under `src/mlirAgent/tools/dialects/`.
- Use `iree-tblgen --gen-dialect-json` style flows for structural extraction.
- Track ingestion plans in `references/graph-ingestion.md` and keep outputs machine-readable (JSON).

### 3) Pass Analysis + Provenance
- Prefer the structural tracer in `src/mlirAgent/tools/provenance.py`.
- Use `src/mlirAgent/tools/trace_provenance.py` only when bindings are unavailable.
- For RLM-based analyses, keep prompts and result parsing deterministic and logged.

### 4) Heuristics + Autocomp Kernels
- Heuristics live under `src/mlirAgent/tools/heuristics/`.
- Kernel creation tools live under `src/mlirAgent/tools/kernels/`.
- Keep heuristics declarative and separate from execution where possible.

### 5) HW-in-the-loop + Performance
- HW harness tools live under `src/mlirAgent/tools/HW-in-the-loop/`.
- Performance tooling lives under `src/mlirAgent/tools/perf/`.
- Use SSH via env-configured targets; do not hardcode credentials or IPs.

### 6) XPU Backends (GPU/NPU/etc.)
- Put backend integration tools in `src/mlirAgent/tools/XPU/`.
- Keep backend enablement steps explicit and reversible.

## Conventions
- Follow the tool interface and file layout in `references/tool-conventions.md`.
- Use `src/mlirAgent/config.py` for paths and external endpoints (Neo4j, LanceDB).
- Write outputs to `data/` or `data/artifacts/` to keep experiments reproducible.

## References
- `references/repo-map.md`
- `references/tool-conventions.md`
- `references/infra.md`
- `references/graph-ingestion.md`
