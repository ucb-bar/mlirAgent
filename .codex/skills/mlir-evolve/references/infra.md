# Infra + Config

## Config Entry Point
- `src/mlirAgent/config.py` defines paths and external services.
- Override defaults via environment variables (e.g., `IREE_SRC_PATH`, `BUILD_DIR`).

## Neo4j + LanceDB
- See `docs/neo4j_setup.md` for local user-space setup.
- `Config.NEO4J_URI`, `Config.NEO4J_USER`, and `Config.NEO4J_PASSWORD` control access.
- `Config.LANCEDB_DIR` defines the local LanceDB path.

## Build + LLVM/IREE Binaries
- `Config.BUILD_DIR` and `Config.INSTALL_DIR` drive build outputs.
- `Config.IREE_COMPILE_PATH`, `Config.FILECHECK_PATH`, `Config.LLVM_LIT_PATH` must exist for tools to run.
