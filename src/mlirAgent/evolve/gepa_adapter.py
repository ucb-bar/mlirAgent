"""GEPA adapter for LLVM heuristic evolution.

Bridges GEPA's ``optimize_anything`` API with our LLVM benchmark evaluator.
Handles EVOLVE-BLOCK extraction, code injection, and score retrieval.
"""

import os
import re
import sys
import tempfile
from pathlib import Path

# Ensure tasks package is importable
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


def make_evaluator(task_name, config=None):
    """Create an evaluator function for GEPA.

    Returns a callable ``code_str -> (score, side_info)`` matching GEPA's
    evaluator protocol.  *side_info* is a dict that may contain a
    ``"Feedback"`` key with ASI markdown text.
    """
    if task_name == "llvm_inlining":
        from tasks.llvm_inlining.evaluate import evaluate
        if config is None:
            config = EvalConfig.from_env(
                "llvm/lib/Analysis/EvolvedInlineCost.cpp"
            )
    elif task_name == "loop_unrolling":
        from tasks.loop_unrolling.evaluate import evaluate
        if config is None:
            config = EvalConfig.from_env(
                "llvm/lib/Transforms/Scalar/EvolvedLoopUnroll.cpp"
            )
    elif task_name == "regalloc_priority":
        from tasks.regalloc_priority.evaluate import evaluate
        if config is None:
            config = EvalConfig.from_env(
                "llvm/lib/CodeGen/EvolvedRegAllocPriority.cpp"
            )
    else:
        raise ValueError(f"Unknown task: {task_name}")

    def evaluator(code_str):
        """Write code to temp file, evaluate, return (score, side_info)."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".cpp", delete=False, prefix="gepa_"
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
