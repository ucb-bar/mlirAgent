"""CLI runner for GEPA on LLVM evolution tasks.

Usage::

    python gepa_run.py --task llvm_inlining [--prompts-dir gepa_prompts]
    python gepa_run.py --task llvm_inlining --max-evals 2 --auto-respond

Requires ``pip install gepa`` and environment variables:
  - LLVM_SRC_PATH: path to LLVM source tree
  - EVOLVE_BUILD_DIR: path to LLVM build directory
"""

import argparse
import json
import os
import re
import sys
import threading
import time
from pathlib import Path

# Ensure local packages are importable
sys.path.insert(0, str(Path(__file__).resolve().parent))

# Task → initial source file mapping
_TASK_INITIAL = {
    "llvm_inlining": "tasks/llvm_inlining/initial.cpp",
    "loop_unrolling": "tasks/loop_unrolling/initial.cpp",
    "regalloc_priority": "tasks/regalloc_priority/initial.cpp",
}

# Task → objective string for GEPA
_TASK_OBJECTIVE = {
    "llvm_inlining": (
        "Maximize binary size reduction across CTMark benchmarks "
        "by modifying the inlining cost heuristic."
    ),
    "loop_unrolling": (
        "Maximize runtime speedup across CTMark benchmarks "
        "by modifying the loop unrolling heuristic."
    ),
    "regalloc_priority": (
        "Maximize runtime speedup across CTMark benchmarks "
        "by modifying the register allocation priority function."
    ),
}

_TASK_BACKGROUND = (
    "You are modifying a C++ heuristic function in LLVM's optimization pipeline. "
    "The function is compiled into the opt/llc tools and evaluated against CTMark "
    "benchmarks (real-world C/C++ programs). The evaluator returns a score based on "
    "binary size reduction and/or runtime speedup vs the default LLVM heuristic. "
    "Higher scores are better. The source code uses LLVM APIs (cl::opt for flags, "
    "InlineCost, LoopUnrollResult, etc.). Expose tunable constants as "
    "// [hyperparam]: flag-name, type, min, max comments for the autotuner."
)


def _auto_respond_thread(prompts_dir, initial_code, poll_interval=1.0):
    """Background thread that auto-creates response files for smoke testing.

    Watches for new prompt_NNN.md files and creates prompt_NNN.response.md
    with a trivially modified version of the initial code.
    """
    seen = set()
    prompt_re = re.compile(r"^prompt_(\d+)\.md$")

    while True:
        try:
            for fname in os.listdir(prompts_dir):
                m = prompt_re.match(fname)
                if not m:
                    continue
                num = m.group(1)
                response_name = f"prompt_{num}.response.md"
                if response_name in seen:
                    continue
                response_path = os.path.join(prompts_dir, response_name)
                if os.path.exists(response_path):
                    seen.add(response_name)
                    continue

                # Create a trivially modified version of the code
                modified = initial_code.replace(
                    "// EVOLVE-BLOCK-START",
                    f"// EVOLVE-BLOCK-START\n// Auto-response iteration {num}",
                )
                with open(response_path, "w") as f:
                    f.write(f"```cpp\n{modified}\n```\n")
                seen.add(response_name)
                print(f"  [auto-respond] Created {response_name}")
        except OSError:
            pass
        time.sleep(poll_interval)


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
        "--max-evals", type=int, default=10,
        help="Maximum evaluator calls (default: 10)",
    )
    parser.add_argument(
        "--output-dir", default=None,
        help="GEPA run directory for state/resume (default: <prompts-dir>/run)",
    )
    parser.add_argument(
        "--output", default=None,
        help="Path to save best code (default: tasks/<task>/gepa_best.cpp)",
    )
    parser.add_argument(
        "--auto-respond", action="store_true",
        help="Auto-create response files for smoke testing",
    )
    args = parser.parse_args()

    # Import GEPA
    try:
        from gepa.optimize_anything import (
            optimize_anything, GEPAConfig, EngineConfig, ReflectionConfig,
        )
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

    run_dir = args.output_dir or os.path.join(args.prompts_dir, "run")
    os.makedirs(run_dir, exist_ok=True)

    print(f"{'=' * 60}")
    print(f"GEPA Runner")
    print(f"  Task:           {args.task}")
    print(f"  Initial code:   {initial_file}")
    print(f"  Prompts dir:    {args.prompts_dir}")
    print(f"  Max evals:      {args.max_evals}")
    print(f"  Run dir:        {run_dir}")
    print(f"  Auto-respond:   {args.auto_respond}")
    print(f"{'=' * 60}")
    print()

    # Start auto-responder thread if requested
    if args.auto_respond:
        t = threading.Thread(
            target=_auto_respond_thread,
            args=(args.prompts_dir, initial_code, args.poll_interval),
            daemon=True,
        )
        t.start()
        print("  [auto-respond] Background responder started")

    # Run GEPA with real API
    objective = _TASK_OBJECTIVE[args.task]
    config = GEPAConfig(
        engine=EngineConfig(
            max_metric_calls=args.max_evals,
            parallel=False,
            run_dir=run_dir,
        ),
        reflection=ReflectionConfig(
            reflection_lm=lm,
        ),
    )

    result = optimize_anything(
        seed_candidate=initial_code,
        evaluator=evaluator,
        objective=objective,
        background=_TASK_BACKGROUND,
        config=config,
    )

    print()
    print(f"{'=' * 60}")
    print(f"GEPA Results:")
    print(f"  Best candidate: {len(result.best_candidate)} chars")
    print(f"  Num candidates: {result.num_candidates}")
    print(f"  Total evals:    {result.total_metric_calls}")
    print(f"{'=' * 60}")

    # Save best code
    output_path = args.output or str(
        base_dir / "tasks" / args.task / "gepa_best.cpp"
    )
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        f.write(result.best_candidate)
    print(f"Best code saved to: {output_path}")

    # Save summary
    summary = {
        "task": args.task,
        "num_candidates": result.num_candidates,
        "total_metric_calls": result.total_metric_calls,
        "output_path": output_path,
        "run_dir": run_dir,
    }
    summary_path = os.path.join(args.prompts_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
