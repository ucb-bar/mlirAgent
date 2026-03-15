# Agent Integration (Portable + Stable)

This repo exposes tools in three layers so any agent (Codex, Claude Code, etc.) can use the same core logic.

## 1) Core API (Python)
All logic lives in `src/mlirAgent/tools/`. These functions return structured dicts for stability.

## 2) CLI Adapter
Run tools directly via the CLI to keep human and agent paths aligned.

Examples:
```bash
python -m mlirAgent.cli build --reconfigure
python -m mlirAgent.cli compile --input path/to/input.mlir --flags --iree-hal-target-backends=llvm-cpu
python -m mlirAgent.cli provenance --root data/artifacts --file input.mlir --line 42
python -m mlirAgent.cli verify --ir out.mlir --check checks.mlir
```

## 3) MCP Adapter (Cross-Agent)
An MCP server exposes the same tools for any MCP-capable agent. The server is implemented with the MCP Python SDK (FastMCP).

Run:
```bash
pip install -r requirements.txt
python -m mlirAgent.mcp_server
```

Tool list (current):
- `build`
- `compile_mlir`
- `compile_mlir_file`
- `verify_ir`
- `verify_ir_files`
- `provenance_trace`
- `provenance_trace_text`

## Environment
Configure paths and external services via environment variables in `src/mlirAgent/config.py`:
- `IREE_SRC_PATH`, `BUILD_DIR`, `IREE_COMPILE_PATH`, `FILECHECK_PATH`
- `NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD`
