---
name: mlir-evolve
description: MLIR/LLVM/IREE workflow for the mlirAgent repo. Use when working in this repo to build or configure toolchains, analyze/debug passes (provenance/rlm), run evolutionary optimization, manage the Neo4j/LanceDB knowledge graph, or mine compiler recipes.
---

# mlirAgent

## Overview
Use this skill to navigate and extend the mlirAgent toolchain and research workflows.

## Quick Start
- Reuse existing wrappers in `src/mlirAgent/tools/*.py` before creating new ones.
- Use `src/mlirAgent/config.py` for paths and external endpoints (Neo4j, LanceDB).
- Install the package with `pip install -e .` for proper imports (no sys.path hacks).

## Core Capabilities

### 1) Build + Toolchain Management
- Use `src/mlirAgent/tools/build.py` for Ninja/CMake builds.
- Prefer Config-driven paths and flags.

### 2) Pass Analysis + Provenance
- Prefer the structural tracer in `src/mlirAgent/tools/provenance.py`.
- Use `src/mlirAgent/tools/trace_provenance.py` only when bindings are unavailable.
- For RLM-based analyses, see `src/mlirAgent/rlm/analysis.py`.

### 3) Evolutionary Optimization
- Evolve tasks live under `src/mlirAgent/evolve/tasks/`.
- Available tasks: `llvm_inlining/`, `llvm_bench.py`, `regalloc_priority/`.
- Configuration via `configs/` directory.

### 4) Recipe Mining
- Mining pipeline in `src/mlirAgent/mining/`.
- Synthesize recipes with `synthesize_recipes.py` (requires a model_fn callable).

### 5) Knowledge Graph (SCIP + Neo4j)
- SCIP ingestion scripts in `src/mlirAgent/scip/`.
- Graph status, inspection, and impact analysis tools included.

## Conventions
- Write outputs to `data/` or `data/artifacts/` to keep experiments reproducible.
- All file paths should come from `Config` or environment variables, not hardcoded.
