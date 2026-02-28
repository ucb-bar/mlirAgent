"""Framework adapters for LLVM heuristic evolution.

Each adapter wraps a specific evolution framework (GEPA, OpenEvolve) with a
common interface so ``run.py`` can dispatch to either one.
"""

import asyncio
import json
import os
import re
import sys
import threading
import time
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path

_BASE_DIR = Path(__file__).resolve().parent

# Ensure local packages are importable
if str(_BASE_DIR) not in sys.path:
    sys.path.insert(0, str(_BASE_DIR))

# Task → initial source file (relative to _BASE_DIR)
_TASK_INITIAL = {
    "llvm_inlining": "tasks/llvm_inlining/initial.cpp",
    "loop_unrolling": "tasks/loop_unrolling/initial.cpp",
    "regalloc_priority": "tasks/regalloc_priority/initial.cpp",
}

# Task → GEPA objective string
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

_GEPA_BACKGROUND = (
    "You are modifying a C++ heuristic function in LLVM's optimization pipeline. "
    "The function is compiled into the opt/llc tools and evaluated against CTMark "
    "benchmarks (real-world C/C++ programs). The evaluator returns a score based on "
    "binary size reduction and/or runtime speedup vs the default LLVM heuristic. "
    "Higher scores are better. The source code uses LLVM APIs (cl::opt for flags, "
    "InlineCost, LoopUnrollResult, etc.). Expose tunable constants as "
    "// [hyperparam]: flag-name, type, min, max comments for the autotuner."
)


class FrameworkAdapter(ABC):
    """Common interface for evolution framework adapters."""

    @abstractmethod
    def run(self, *, task, initial_file, prompts_dir, max_evals,
            auto_respond, poll_interval, output, exp_dir, **kwargs):
        """Run the evolution loop. Returns a result dict."""
        ...


# ---------------------------------------------------------------------------
# GEPA
# ---------------------------------------------------------------------------

def _auto_respond_thread(prompts_dir, initial_code, poll_interval=1.0):
    """Background thread that auto-creates response files for smoke testing."""
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


class GEPAAdapter(FrameworkAdapter):
    """Runs GEPA ``optimize_anything()`` with ManualLM and ASI feedback.

    GEPA's evaluator receives ``(score, {"Feedback": ASI_text})`` so it
    can embed rich diagnostic feedback into its reflection prompts.
    Optuna hyperparameter tuning runs inside each evaluation automatically
    when ``[hyperparam]`` annotations are present in the C++ code.
    """

    def run(self, *, task, initial_file, prompts_dir, max_evals,
            auto_respond, poll_interval=2.0, output=None, exp_dir=None,
            **kwargs):
        from gepa.optimize_anything import (
            optimize_anything, GEPAConfig, EngineConfig, ReflectionConfig,
        )
        from gepa_manual_lm import ManualLM
        from evaluator import make_evaluator

        with open(initial_file) as f:
            initial_code = f.read()

        lm = ManualLM(prompts_dir=prompts_dir, poll_interval=poll_interval)
        evaluator = make_evaluator(task)

        run_dir = os.path.join(exp_dir, "gepa_state") if exp_dir else None
        if run_dir:
            os.makedirs(run_dir, exist_ok=True)

        if auto_respond:
            t = threading.Thread(
                target=_auto_respond_thread,
                args=(prompts_dir, initial_code, poll_interval),
                daemon=True,
            )
            t.start()
            print("  [auto-respond] Background responder started")

        config = GEPAConfig(
            engine=EngineConfig(
                max_metric_calls=max_evals,
                parallel=False,
                run_dir=run_dir,
                use_cloudpickle=False,
                raise_on_exception=False,
            ),
            reflection=ReflectionConfig(
                reflection_lm=lm,
            ),
        )

        result = optimize_anything(
            seed_candidate=initial_code,
            evaluator=evaluator,
            objective=_TASK_OBJECTIVE[task],
            background=_GEPA_BACKGROUND,
            config=config,
        )

        # Save best code
        output_path = output or os.path.join(exp_dir or ".", "best.cpp")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as f:
            f.write(result.best_candidate)
        print(f"Best code saved to: {output_path}")

        summary = {
            "framework": "gepa",
            "task": task,
            "num_candidates": result.num_candidates,
            "total_metric_calls": result.total_metric_calls,
            "output_path": output_path,
        }
        summary_path = os.path.join(exp_dir or prompts_dir, "summary.json")
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"Summary saved to: {summary_path}")

        return summary


# ---------------------------------------------------------------------------
# OpenEvolve
# ---------------------------------------------------------------------------

class OpenEvolveAdapter(FrameworkAdapter):
    """Runs OpenEvolve with ManualLLM (file-based prompt/response).

    OpenEvolve uses MAP-Elites population management. ASI feedback is
    passed through ``EvaluationResult.artifacts["asi"]``.  Optuna runs
    inside each evaluation automatically.
    """

    def run(self, *, task, initial_file, prompts_dir, max_evals,
            auto_respond, poll_interval=2.0, output=None, exp_dir=None,
            resume=None, **kwargs):
        return asyncio.run(self._run_async(
            task=task, initial_file=initial_file, prompts_dir=prompts_dir,
            max_evals=max_evals, auto_respond=auto_respond,
            output=output, exp_dir=exp_dir, resume=resume,
        ))

    async def _run_async(self, *, task, initial_file, prompts_dir,
                         max_evals, auto_respond, output, exp_dir, resume):
        # Ensure openevolve is importable
        mlirevolve_root = _BASE_DIR.parent.parent.parent
        oe_path = str(mlirevolve_root / "third_party" / "openevolve")
        if oe_path not in sys.path:
            sys.path.insert(0, oe_path)

        from openevolve.config import Config as OEConfig, LLMModelConfig
        from openevolve.controller import OpenEvolve
        from openevolve.llm.manual import create_manual_llm

        os.environ["MANUAL_LLM_PROMPTS_DIR"] = prompts_dir

        # Build config
        configs_dir = str(mlirevolve_root / "configs")
        fw_yaml = os.path.join(configs_dir, "frameworks", "manual.yaml")
        cfg = OEConfig.from_yaml(fw_yaml) if os.path.exists(fw_yaml) else OEConfig()

        cfg.max_iterations = max_evals
        cfg.file_suffix = ".cpp"
        cfg.language = "cpp"

        manual_model = LLMModelConfig(
            name="manual",
            init_client=create_manual_llm,
            weight=1.0,
        )
        cfg.llm.models = [manual_model]
        cfg.llm.evaluator_models = [manual_model]

        cfg.database.population_size = 10
        cfg.database.archive_size = 10
        cfg.database.num_islands = 1
        cfg.database.migration_interval = 999
        cfg.checkpoint_interval = 1
        cfg.diff_based_evolution = False

        # Evaluator path (the task's evaluate.py)
        evaluator_path = str(_BASE_DIR / "tasks" / task / "evaluate.py")

        oe_output_dir = os.path.join(exp_dir, "openevolve_output")
        scores_path = os.path.join(exp_dir, "scores.jsonl")
        os.makedirs(oe_output_dir, exist_ok=True)

        openevolve = OpenEvolve(
            initial_program_path=initial_file,
            evaluation_file=evaluator_path,
            config=cfg,
            output_dir=oe_output_dir,
        )

        if resume:
            if os.path.exists(resume):
                print(f"Resuming from checkpoint: {resume}")
                openevolve.database.load(resume)
            else:
                print(f"Warning: Checkpoint not found: {resume}")

        # Auto-respond thread
        stop_event = asyncio.Event()
        responder_task = None
        if auto_respond:
            loop = asyncio.get_event_loop()
            responder_task = loop.run_in_executor(
                None, _oe_auto_respond, prompts_dir, stop_event,
            )

        # Score logging hook
        _original_add = openevolve.database.add

        def _logging_add(program, *a, **kw):
            result = _original_add(program, *a, **kw)
            entry = {
                "timestamp": time.time(),
                "iteration": program.iteration_found,
                "program_id": program.id,
                "metrics": program.metrics,
            }
            best = openevolve.database.get_best_program()
            if best:
                entry["best_score"] = best.metrics.get("combined_score", 0)
            with open(scores_path, "a") as f:
                f.write(json.dumps(entry, default=str) + "\n")
            return result

        openevolve.database.add = _logging_add

        try:
            print(f"Starting OpenEvolve ({cfg.max_iterations} iterations)...")
            best = await openevolve.run(
                iterations=cfg.max_iterations,
                checkpoint_path=resume,
            )
            if best:
                print(f"\nBest metrics:")
                for k, v in best.metrics.items():
                    if isinstance(v, float):
                        print(f"  {k}: {v:.4f}")
                    else:
                        print(f"  {k}: {v}")

                # Save best code
                if output and hasattr(best, "code"):
                    os.makedirs(os.path.dirname(output), exist_ok=True)
                    with open(output, "w") as f:
                        f.write(best.code)
                    print(f"Best code saved to: {output}")
        finally:
            stop_event.set()
            if responder_task:
                await asyncio.sleep(2)

        summary = {
            "framework": "openevolve",
            "task": task,
            "output_dir": oe_output_dir,
        }
        summary_path = os.path.join(exp_dir, "summary.json")
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

        return summary


def _oe_auto_respond(prompts_dir, stop_event):
    """Auto-responder for OpenEvolve (trivial code modification)."""
    import glob

    responded = set()
    pattern = re.compile(r"prompt_\d+\.md$")

    while not stop_event.is_set():
        prompt_files = sorted(
            f for f in glob.glob(os.path.join(prompts_dir, "prompt_*.md"))
            if pattern.search(f)
        )
        for pf in prompt_files:
            if pf in responded:
                continue
            resp_path = pf.replace(".md", ".response.md")
            if os.path.exists(resp_path):
                responded.add(pf)
                continue

            with open(pf) as f:
                prompt_text = f.read()

            # Trivial modification: add a comment to the code
            num = Path(pf).stem.split("_")[-1]
            response = f"// Auto-response iteration {num}\n{prompt_text[:500]}"
            with open(resp_path, "w") as f:
                f.write(response)
            responded.add(pf)
            print(f"  [auto] Responded to {os.path.basename(pf)}")

        time.sleep(1)
