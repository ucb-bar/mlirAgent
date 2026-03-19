# mlirAgent: AI-Guided MLIR/LLVM Compiler Optimization

**mlirAgent** is a research framework for automating MLIR compiler development using LLM agents.
It bridges raw compiler source code and AI agents through high-quality datasets, semantic knowledge graphs, and evolutionary optimization.

Key result: **8.78% binary size reduction** via LLM-guided LLVM inlining heuristic evolution.

## Modules

| Module | Path | Purpose |
|--------|------|---------|
| **Mining** | `src/mlirAgent/mining/` | Extract "gold standard" compiler recipes from git history |
| **SCIP/Knowledge Graph** | `src/mlirAgent/scip/` | Hydrate Neo4j with SCIP-indexed codebase structure |
| **Tools** | `src/mlirAgent/tools/` | Compiler wrappers, verification, provenance tracing |
| **Evolve** | `src/mlirAgent/evolve/` | LLM-driven evolutionary optimization of compiler heuristics |
| **RLM** | `src/mlirAgent/rlm/` | Reasoning Language Model integration for log analysis |

## Setup

```bash
# Create a virtualenv and install in editable mode
python -m venv .venv
source .venv/bin/activate
pip install -e .

# With optional dependencies
pip install -e ".[evolve,dev]"
```

## Usage

```bash
# CLI
mlirAgent --help
mlirAgent build --fast
mlirAgent compile --input model.mlir

# MCP Server
mlirAgent-mcp
```

## Documentation

- `docs/neo4j_setup.md` — Neo4j knowledge graph setup
- `docs/agent_integration.md` — Agent integration guide
- `experiments/` — Research sandbox with artifact traces and experiment outputs
