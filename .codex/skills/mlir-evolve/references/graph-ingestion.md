# Graph Ingestion Roadmap

## Goal
Build a unified Neo4j graph that connects build targets, files, code symbols, and MLIR dialect semantics.

## Planned Ingestion Sources
1) **SCIP (C++ Code Graph)**
- Implemented in `src/mlirAgent/scip/` using `scip-clang` outputs.

2) **Dialect Metadata (TableGen)**
- Use `iree-tblgen --gen-dialect-json` to dump ops/attrs/types.
- Target tool path: `src/mlirAgent/tools/dialects/`.

3) **CMake File API (Build Graph)**
- Use `.cmake/api/v1/query` + response JSON to map files -> targets -> link deps.
- Target tool path: `src/mlirAgent/tools/building/`.

## Integration Points
- Update `src/mlirAgent/tools/retriever.py` to query the unified graph.
- Keep ingestion outputs in `data/knowledge_base/` for reproducibility.
