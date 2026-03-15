# Repository Map (mlirEvolve)

## Key Paths
- `src/mlirAgent/tools/`: Tool wrappers and domain folders.
- `src/mlirAgent/scip/`: SCIP ingestion and graph analysis utilities.
- `src/mlirAgent/rlm/`: RLM-based analysis (prompts + wrappers).
- `docs/`: Setup notes and design plans (Neo4j, ingestion roadmap).
- `data/`: Artifacts, knowledge base, recipes, and experiment inputs.
- `experiments/`: One-off experiments and test artifacts.

## Tools Directory Layout
- `src/mlirAgent/tools/build.py`: Ninja/CMake build wrapper.
- `src/mlirAgent/tools/compiler.py`: iree-compile wrapper.
- `src/mlirAgent/tools/provenance.py`: Structural pass provenance tracer.
- `src/mlirAgent/tools/trace_provenance.py`: Text-based fallback tracer.
- `src/mlirAgent/tools/verifier.py`: FileCheck wrapper.
- Domain folders (currently empty):
  - `building/`, `dialects/`, `heuristics/`, `kernels/`, `passes/`, `perf/`
  - `HW-in-the-loop/`, `XPU/`, `instr/`

## Data + Graph Assets
- `data/knowledge_base/`: SCIP and mined recipe data.
- `data/cookbook/LLVM_recipes/`: YAML/JSON recipes used for heuristics.
