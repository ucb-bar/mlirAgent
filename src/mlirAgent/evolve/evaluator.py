"""Evaluator bridges for LLVM heuristic evolution.

Creates framework-agnostic evaluator callables that:
1. Write candidate C++ to a temp file
2. Call the task-specific evaluate() (patch LLVM, build, benchmark)
3. Return (score, side_info) for GEPA or EvaluationResult for OpenEvolve

The actual compilation/benchmark logic lives in ``tasks/llvm_bench.py``
and per-task ``evaluate.py`` files.
"""

import os
import re
import sys
import tempfile
from pathlib import Path

# Ensure tasks package is importable when run standalone
sys.path.insert(0, str(Path(__file__).resolve().parent))

from tasks.llvm_bench import EvalConfig

_EVOLVE_BLOCK_RE = re.compile(
    r"(// EVOLVE-BLOCK-START\n)(.*?)(// EVOLVE-BLOCK-END)",
    re.DOTALL,
)


def extract_evolve_block(code):
    """Extract the EVOLVE-BLOCK content from C++ source code."""
    m = _EVOLVE_BLOCK_RE.search(code)
    if m:
        return m.group(2)
    return code


def inject_evolve_block(template, block):
    """Replace EVOLVE-BLOCK in *template* with new *block* content."""
    return _EVOLVE_BLOCK_RE.sub(
        lambda m: m.group(1) + block + m.group(3),
        template,
    )


# Task → (target_file, default baseline overrides)
_TASK_CONFIG = {
    "llvm_inlining": {
        "target_file": "llvm/lib/Analysis/EvolvedInlineCost.cpp",
    },
    "loop_unrolling": {
        "target_file": "llvm/lib/Transforms/Scalar/EvolvedLoopUnroll.cpp",
        "baseline_file": str(
            Path(__file__).resolve().parent
            / "tasks" / "loop_unrolling" / "baseline_unroll.json"
        ),
    },
    "regalloc_priority": {
        "target_file": "llvm/lib/CodeGen/EvolvedRegAllocPriority.cpp",
        "baseline_file": str(
            Path(__file__).resolve().parent
            / "tasks" / "regalloc_priority" / "baseline_regalloc.json"
        ),
    },
}


def _import_evaluate(task_name):
    """Import the task-specific evaluate function."""
    if task_name == "llvm_inlining":
        from tasks.llvm_inlining.evaluate import evaluate
    elif task_name == "loop_unrolling":
        from tasks.loop_unrolling.evaluate import evaluate
    elif task_name == "regalloc_priority":
        from tasks.regalloc_priority.evaluate import evaluate
    else:
        raise ValueError(f"Unknown task: {task_name}")
    return evaluate


def make_evaluator(task_name, config=None):
    """Create an evaluator function for a given task.

    Returns a callable ``code_str -> (score, side_info)`` matching GEPA's
    evaluator protocol.  *side_info* is a dict that may contain a
    ``"Feedback"`` key with ASI markdown text.

    The same evaluator works with OpenEvolve — the score is extracted from
    the returned tuple's first element.
    """
    evaluate = _import_evaluate(task_name)

    if config is None:
        tc = _TASK_CONFIG[task_name]
        config = EvalConfig.from_env(tc["target_file"], **{
            k: v for k, v in tc.items() if k != "target_file"
        })

    def evaluator(code_str):
        """Write code to temp file, evaluate, return (score, side_info)."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".cpp", delete=False, prefix="evolve_"
        ) as f:
            f.write(code_str)
            tmp_path = f.name
        try:
            result = evaluate(tmp_path, config=config)
            if isinstance(result, dict):
                score = result.get("combined_score", 0.0)
                side_info = {}
            else:
                # EvaluationResult from OpenEvolve
                score = result.metrics.get("combined_score", 0.0)
                if hasattr(result, "artifacts") and "asi" in result.artifacts:
                    side_info = {"Feedback": result.artifacts["asi"]}
                else:
                    side_info = {}
            return score, side_info
        finally:
            os.unlink(tmp_path)

    return evaluator
