"""Unified entry point for LLVM heuristic evolution.

Supports two evolution frameworks dispatched via ``--framework``:
  - **gepa** (default): GEPA optimize_anything() with ASI as native side-info
  - **openevolve**: MAP-Elites with ManualLLM

Both frameworks share the same evaluator pipeline (``tasks/llvm_bench.py``),
which includes Optuna hyperparameter tuning when ``[hyperparam]`` annotations
are present in the evolved C++ code.

Usage::

    # GEPA with auto-respond (smoke test)
    python run.py --task llvm_inlining --max-evals 2 --auto

    # GEPA manual mode (Claude Code or human writes response files)
    python run.py --task llvm_inlining --max-evals 10

    # OpenEvolve
    python run.py --framework openevolve --task llvm_inlining --max-evals 10 --auto

    # Override Optuna trials
    python run.py --task llvm_inlining --max-evals 10 --optuna-trials 5
"""

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

_BASE_DIR = Path(__file__).resolve().parent

# Ensure local packages are importable
if str(_BASE_DIR) not in sys.path:
    sys.path.insert(0, str(_BASE_DIR))

_TASKS = ["llvm_inlining", "loop_unrolling", "regalloc_priority"]

_TASK_INITIAL = {
    "llvm_inlining": _BASE_DIR / "tasks" / "llvm_inlining" / "initial.cpp",
    "loop_unrolling": _BASE_DIR / "tasks" / "loop_unrolling" / "initial.cpp",
    "regalloc_priority": _BASE_DIR / "tasks" / "regalloc_priority" / "initial.cpp",
}


def main():
    parser = argparse.ArgumentParser(
        description="Evolve LLVM heuristics via LLM-guided search",
    )
    parser.add_argument(
        "--framework", "-f", default="gepa",
        choices=["gepa", "openevolve"],
        help="Evolution framework (default: gepa)",
    )
    parser.add_argument(
        "--task", "-t", required=True,
        choices=_TASKS,
        help="LLVM task to optimize",
    )
    parser.add_argument(
        "--initial", default=None,
        help="Override initial C++ source path",
    )
    parser.add_argument(
        "--max-evals", "-n", type=int, default=10,
        help="Max evaluator calls / iterations (default: 10)",
    )
    parser.add_argument(
        "--auto", action="store_true",
        help="Auto-respond to prompts (for smoke testing)",
    )
    parser.add_argument(
        "--prompts-dir", default=None,
        help="Prompt/response directory (default: auto)",
    )
    parser.add_argument(
        "--output", default=None,
        help="Save best code to this path",
    )
    parser.add_argument(
        "--resume", default=None,
        help="Resume from checkpoint (OpenEvolve only)",
    )
    parser.add_argument(
        "--poll-interval", type=float, default=2.0,
        help="ManualLM poll interval in seconds (default: 2.0)",
    )
    parser.add_argument(
        "--optuna-trials", type=int, default=None,
        help="Optuna inner-loop trials (overrides EVOLVE_OPTUNA_TRIALS env)",
    )
    args = parser.parse_args()

    # Set Optuna env var if specified
    if args.optuna_trials is not None:
        os.environ["EVOLVE_OPTUNA_TRIALS"] = str(args.optuna_trials)

    # Set up experiment directory
    mlirevolve_root = _BASE_DIR.parent.parent.parent
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = str(mlirevolve_root / "experiments" / f"run_{timestamp}")
    prompts_dir = args.prompts_dir or os.path.join(exp_dir, "prompts")
    os.makedirs(prompts_dir, exist_ok=True)
    os.makedirs(exp_dir, exist_ok=True)

    # Resolve initial program
    initial_file = args.initial or str(_TASK_INITIAL[args.task])
    if not os.path.exists(initial_file):
        print(f"Error: Initial source not found: {initial_file}")
        sys.exit(1)

    # Print header
    optuna_trials = args.optuna_trials or os.environ.get("EVOLVE_OPTUNA_TRIALS", "20")
    print(f"{'=' * 60}")
    print(f"Evolve LLVM Heuristics")
    print(f"  Framework:      {args.framework}")
    print(f"  Task:           {args.task}")
    print(f"  Initial:        {initial_file}")
    print(f"  Max evals:      {args.max_evals}")
    print(f"  Optuna trials:  {optuna_trials}")
    print(f"  Auto-respond:   {args.auto}")
    print(f"  Prompts:        {prompts_dir}")
    print(f"  Experiment:     {exp_dir}")
    print(f"{'=' * 60}")
    print()

    # Dispatch to framework adapter
    if args.framework == "gepa":
        from adapters import GEPAAdapter
        adapter = GEPAAdapter()
    else:
        from adapters import OpenEvolveAdapter
        adapter = OpenEvolveAdapter()

    result = adapter.run(
        task=args.task,
        initial_file=initial_file,
        prompts_dir=prompts_dir,
        max_evals=args.max_evals,
        auto_respond=args.auto,
        poll_interval=args.poll_interval,
        output=args.output or os.path.join(exp_dir, "best.cpp"),
        exp_dir=exp_dir,
        resume=args.resume,
    )

    print()
    print(f"{'=' * 60}")
    print(f"Done. Results in: {exp_dir}")
    print(f"{'=' * 60}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
