# Tool Conventions

## Python Tool Interface
Prefer a single public entrypoint per tool module:
- `run_<tool_name>(..., **kwargs) -> dict`
- Return a dict with at least:
  - `success` (bool)
  - `returncode` (int, use -1 for internal failures)
  - `stdout` / `stderr` (str)
  - `command` (str)
  - Optional: `duration_seconds`, `artifact_path`, `error_summary`

## File Layout Rules
- Put new domain tools under `src/mlirAgent/tools/<domain>/`.
- Keep thin wrappers in `src/mlirAgent/tools/*.py` when they are core to most workflows.
- Avoid hard-coded paths; use `src/mlirAgent/config.py` and env vars.

## Logging + Artifacts
- Write outputs to `data/artifacts/` or a domain-specific subfolder.
- Keep logs deterministic and machine-readable (JSON where possible).
- For long-running tasks, include a timeout and return a helpful error summary.

## Safety
- Never embed credentials or IPs in code. Read from env vars.
- Default to read-only operations unless the tool is explicitly destructive.
