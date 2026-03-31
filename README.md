# mlirAgent: AI-Guided MLIR/LLVM Compiler Optimization

**mlirAgent** is a research framework for automating MLIR compiler development using LLM agents. It bridges raw compiler source code and AI agents through structural IR fingerprinting, semantic knowledge graphs, and evolutionary optimization.

**Key results:**
- **8.78% binary size reduction** via LLM-guided LLVM inlining heuristic evolution (matching Magellan/ICML 2025 with 10x fewer iterations)
- **92.7% accuracy** at predicting compiler pass actions from structural fingerprints (vs 47.9% majority-class baseline)
- **LLMs cannot replace compiler passes** — Gemini 2.5 Pro scores 11.9% *below* identity baseline when attempting IR transformations

## Architecture

```
src/mlirAgent/
├── research/       ← Research pipeline (17 modules, see docs/research_results.md)
│   ├── schema.py               # TransformationRecord + SQLite storage
│   ├── llm_compiler.py         # LLM-as-compiler-pass experiment
│   ├── pattern_similarity.py   # 3-level cross-project matching
│   ├── prediction_model.py     # XGBoost action/structural prediction
│   └── ...
├── tools/          ← MCP tools (29 tools) for agent interaction
│   ├── pass_tracker.py         # Structural fingerprinting engine
│   ├── knowledge.py            # Knowledge graph queries
│   └── ...
├── evolve/         ← LLM-driven evolutionary optimization
│   └── tasks/                  # LLVM inlining, register allocation
├── mining/         ← Recipe extraction from LLVM git history
├── graph/          ← Code knowledge graph (18K symbols, 103K edges)
└── rlm/            ← Deterministic analysis layer
```

## Quick Start

```bash
# Environment (requires conda for MLIR bindings)
conda activate merlin-dev

# Run the MCP server (29 tools for LLM agents)
uv run mlirAgent-mcp

# Run research experiments
uv run python -m mlirAgent.research.run_experiments all \
    --artifacts-dir experiments/debug_info/artifacts_debug \
    --output-dir data/research/results

# Run the LLM compiler experiment (requires GOOGLE_API_KEY in .env)
uv run python scripts/run_llm_compiler.py --model gemini-2.5-pro --samples 30

# Run tests
uv run python -m pytest tests/ -v
```

## Documentation

| Document | Contents |
|----------|----------|
| [docs/research_results.md](docs/research_results.md) | Full experimental results, methodology, and reproduction steps |
| [docs/architecture.md](docs/architecture.md) | Three-tier MCP architecture |
| [docs/design_decisions.md](docs/design_decisions.md) | Key design rationale (9 decisions) |
| [docs/pass_tracking.md](docs/pass_tracking.md) | Structural fingerprinting and provenance |
| [docs/mcp_tools.md](docs/mcp_tools.md) | All 29 MCP tools reference |
| [docs/reproducing_results.md](docs/reproducing_results.md) | Step-by-step reproduction guide |
| [docs/testing.md](docs/testing.md) | Test suite documentation |

## Repository Layout

```
mlirAgent/
├── src/mlirAgent/          Source code (research, tools, evolve, mining, graph)
├── tests/                  Test suite (49+ tests)
├── scripts/                Experiment runner scripts
├── experiments/            Raw compilation artifacts and debug data
├── data/                   Generated databases and results (gitignored)
├── docs/                   Documentation
└── configs/                Agent, framework, and task configs for evolution
```
