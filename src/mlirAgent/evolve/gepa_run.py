"""CLI runner for GEPA on LLVM evolution tasks.

Usage::

    python gepa_run.py --task llvm_inlining [--prompts-dir gepa_prompts]

Requires ``pip install gepa`` and environment variables:
  - LLVM_SRC_PATH: path to LLVM source tree
  - EVOLVE_BUILD_DIR: path to LLVM build directory
"""

import argparse
import json
import os
import sys
from pathlib import Path

# Ensure local packages are importable
sys.path.insert(0, str(Path(__file__).resolve().parent))

# Task → initial source file mapping
_TASK_INITIAL = {
    "llvm_inlining": "tasks/llvm_inlining/initial.cpp",
    "loop_unrolling": "tasks/loop_unrolling/initial.cpp",
    "regalloc_priority": "tasks/regalloc_priority/initial.cpp",
}


def main():
    parser = argparse.ArgumentParser(
        description="Run GEPA on LLVM heuristic evolution tasks"
    )
    parser.add_argument(
        "--task", required=True,
        choices=list(_TASK_INITIAL.keys()),
        help="Task to optimize",
    )
    parser.add_argument(
        "--initial", default=None,
        help="Path to initial C++ source (overrides default)",
    )
    parser.add_argument(
        "--prompts-dir", default="gepa_prompts",
        help="Directory for prompt/response files (default: gepa_prompts)",
    )
    parser.add_argument(
        "--poll-interval", type=float, default=2.0,
        help="Poll interval for response files in seconds (default: 2.0)",
    )
    parser.add_argument(
        "--max-iterations", type=int, default=10,
        help="Maximum GEPA iterations (default: 10)",
    )
    parser.add_argument(
        "--output", default=None,
        help="Path to save best code (default: tasks/<task>/gepa_best.cpp)",
    )
    args = parser.parse_args()

    # Import GEPA
    try:
        from gepa import optimize_anything
    except ImportError:
        print("Error: gepa not installed. Run: pip install gepa")
        sys.exit(1)

    from gepa_manual_lm import ManualLM
    from gepa_adapter import make_evaluator

    # Find initial program
    base_dir = Path(__file__).resolve().parent
    if args.initial:
        initial_file = Path(args.initial)
    else:
        initial_file = base_dir / _TASK_INITIAL[args.task]

    if not initial_file.exists():
        print(f"Error: initial source not found at {initial_file}")
        sys.exit(1)

    with open(initial_file) as f:
        initial_code = f.read()

    # Create LM and evaluator
    lm = ManualLM(
        prompts_dir=args.prompts_dir,
        poll_interval=args.poll_interval,
    )
    evaluator = make_evaluator(args.task)

    print(f"{'=' * 60}")
    print(f"GEPA Runner")
    print(f"  Task:           {args.task}")
    print(f"  Initial code:   {initial_file}")
    print(f"  Prompts dir:    {args.prompts_dir}")
    print(f"  Max iterations: {args.max_iterations}")
    print(f"{'=' * 60}")
    print()

    # Evaluate initial program first
    print("Evaluating initial program...")
    initial_score = evaluator(initial_code)
    print(f"  Initial score: {initial_score}")
    print()

    # Run GEPA
    result = optimize_anything(
        initial_code=initial_code,
        evaluate_fn=evaluator,
        lm=lm,
        max_iterations=args.max_iterations,
    )

    print()
    print(f"{'=' * 60}")
    print(f"GEPA Results:")
    print(f"  Best score:     {result.best_score}")
    print(f"  Initial score:  {initial_score}")
    print(f"  Improvement:    {result.best_score - initial_score:+.4f}")
    print(f"  Iterations:     {result.iterations}")
    print(f"{'=' * 60}")

    # Save best code
    output_path = args.output or str(
        base_dir / "tasks" / args.task / "gepa_best.cpp"
    )
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        f.write(result.best_code)
    print(f"Best code saved to: {output_path}")

    # Save summary
    summary = {
        "task": args.task,
        "initial_score": initial_score,
        "best_score": result.best_score,
        "iterations": result.iterations,
        "output_path": output_path,
    }
    summary_path = os.path.join(args.prompts_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
