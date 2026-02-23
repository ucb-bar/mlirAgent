# Evolve: Evolutionary LLVM Heuristic Optimization

Automated framework for evolving LLVM compiler heuristics using
[OpenEvolve](../../third_party/openevolve/) with LLM-guided search.

## End-to-End Flow

### One-Time Setup

**1. Build LLVM with evolved hooks**

```bash
# Shallow clone
git clone --depth 1 https://github.com/llvm/llvm-project.git /scratch/ashvin/llvm-project

# Add evolved heuristic files to the LLVM tree:
#   llvm/include/llvm/Analysis/EvolvedInlineCost.h
#   llvm/lib/Analysis/EvolvedInlineCost.cpp          (inlining hook)
#   llvm/include/llvm/CodeGen/EvolvedRegAllocPriority.h
#   llvm/lib/CodeGen/EvolvedRegAllocPriority.cpp      (regalloc hook)
# Register them in the corresponding CMakeLists.txt files.
# Hook into InlineCost.cpp and RegAllocGreedy.cpp with cl::opt flags.

# Configure: Release, X86-only, GCC 13 + gold linker
cmake -G Ninja -B /scratch/ashvin/llvm-build \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLVM_TARGETS_TO_BUILD=X86 \
  -DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=g++ \
  /scratch/ashvin/llvm-project/llvm

# Build (produces bin/opt and bin/llc, ~657MB)
ninja -C /scratch/ashvin/llvm-build bin/opt bin/llc
```

**2. Prepare CTMark benchmarks as .bc files**

The benchmarks come from [llvm-test-suite](https://github.com/llvm/llvm-test-suite)
CTMark. They are compiled to LLVM bitcode (.bc) with frontend optimizations
only, so our evolved passes have full control over LLVM-level optimization:

```bash
# compile_testsuite.sh does this for each benchmark:
clang-18 -O1 -Xclang -disable-llvm-optzns -emit-llvm -c source.c -o source.bc
llvm-link *.bc -o benchmark.bc    # multi-file benchmarks
```

`-O1 -Xclang -disable-llvm-optzns` enables Clang frontend opts (type lowering,
etc.) but skips all LLVM IR passes. The resulting .bc files contain unoptimized
IR ready for our `opt -O2` pipeline.

The 8 benchmarks used (2 excluded: clamav=segfault, 7zip=link error):

| Benchmark | Language | Source | Description |
|-----------|----------|--------|-------------|
| bullet | C++ | MultiSource/Benchmarks/Bullet | Physics engine simulation |
| consumer-typeset | C | MultiSource/Applications/lout | Document typesetting (Lout) |
| kimwitu | C++ | MultiSource/Applications/kimwitu++ | Tree pattern matching |
| lencod | C | MultiSource/Applications/JM/lencod | H.264 video encoder |
| mafft | C | MultiSource/Applications/mafft | Multiple sequence alignment |
| spass | C | MultiSource/Applications/SPASS | First-order theorem prover |
| sqlite3 | C | MultiSource/Applications/sqlite3 | SQL database engine |
| tramp3d-v4 | C++ | MultiSource/Benchmarks/tramp3d-v4 | Template metaprogramming |

The .bc files and runtime data live in `tasks/llvm_inlining/benchmarks/testsuite/`:
```
testsuite/
  bullet.bc, consumer-typeset.bc, kimwitu.bc, ...
  data/
    bullet/           # landscape.mdl, Taru.mdl
    consumer-typeset/  # large.lout, data/, font/, maps/, hyph/, include/
    kimwitu/          # inputs/f1.k, f2.k, f3.k
    lencod/           # encoder_small.cfg, foreman_part_qcif_444.yuv, ...
    mafft/            # pyruvate_decarboxylase.fasta
    spass/            # problem.dfg
    sqlite3/          # commands, sqlite3rc, test1.sql-test15.sql
```

### Running an Experiment

```bash
# Set environment
export LLVM_SRC_PATH=/scratch/ashvin/llvm-project
export EVOLVE_BUILD_DIR=/scratch/ashvin/llvm-build
export EVOLVE_OPTUNA_TRIALS=5   # 0 to disable Optuna

# Launch (--wait mode: you respond to prompts manually or via Claude Code)
python -m mlirAgent.evolve.manual_run --example llvm_inlining -n 10 --wait

# Or auto mode (built-in heuristic strategies respond automatically)
python -m mlirAgent.evolve.manual_run --example regalloc_priority -n 10 --auto
```

This creates an experiment directory:
```
experiments/run_20260219_132604/
  scores.jsonl                    # One JSON line per iteration with all metrics
  prompts/
    prompt_001.md                 # OpenEvolve prompt (parent code + history)
    prompt_001.response.md        # LLM/agent response (new code)
    prompt_002.md
    ...
  openevolve_output/
    checkpoints/checkpoint_N/     # Population state for --resume
    best/best_program.cpp         # Best evolved program
    logs/openevolve_*.log         # Detailed log
```

### What Happens Each Iteration

```
                        ┌─────────────────────────────────┐
                        │        OpenEvolve Controller     │
                        │  (population, MAP-Elites, etc.)  │
                        └────────────┬────────────────────┘
                                     │ 1. Sample parent program
                                     │    from population
                                     ▼
                        ┌─────────────────────────────────┐
                        │          ManualLLM Bridge        │
                        │  Write prompt_NNN.md to disk     │
                        │  Poll for prompt_NNN.response.md │
                        └────────────┬────────────────────┘
                                     │ 2. External responder
                                     │    writes response file
                                     ▼
                        ┌─────────────────────────────────┐
                        │       Task Evaluator (evaluate.py)│
                        └────────────┬────────────────────┘
                                     │
              ┌──────────────────────┼──────────────────────┐
              ▼                      ▼                      ▼
    3. patch_source()       4. build_llvm()         5. load_baseline()
    Copy evolved .cpp       ninja -C BUILD_DIR      Compile & run all
    into LLVM tree          bin/opt bin/llc          benchmarks with
    (backup original)       (~3.5s incremental)     default LLVM (once,
                                                    cached to .json)
              │                      │                      │
              └──────────────────────┼──────────────────────┘
                                     │
                                     ▼
                        ┌─────────────────────────────────┐
                        │  6. [Optuna inner-loop]          │
                        │  If [hyperparam] annotations:    │
                        │  Run N trials on 3-bench subset  │
                        │  (sqlite3, spass, tramp3d-v4)    │
                        │  Each trial = compile+run subset │
                        │  Find best flag values           │
                        └────────────┬────────────────────┘
                                     │
                                     ▼
                        ┌─────────────────────────────────┐
                        │  7. eval_benchmarks()            │
                        │  For EACH of 8 .bc benchmarks:   │
                        │  ┌─────────────────────────────┐ │
                        │  │ a. opt -O2                  │ │
                        │  │    [-use-evolved-inline-cost]│ │
                        │  │    bench.bc → bench_opt.bc  │ │
                        │  ├─────────────────────────────┤ │
                        │  │ b. llc -O2 -filetype=obj    │ │
                        │  │    -relocation-model=pic    │ │
                        │  │    [-use-evolved-regalloc-*]│ │
                        │  │    [-ae-flag=value ...]     │ │
                        │  │    bench_opt.bc → bench.o   │ │
                        │  ├─────────────────────────────┤ │
                        │  │ c. gcc bench.o -o bench     │ │
                        │  │    -lm -lpthread -ldl       │ │
                        │  │    [-lstdc++ for C++ bench]  │ │
                        │  ├─────────────────────────────┤ │
                        │  │ d. size bench.o → .text size│ │
                        │  │    stat bench   → binary sz │ │
                        │  ├─────────────────────────────┤ │
                        │  │ e. Run 5x, take median:     │ │
                        │  │    ./bench [args] [<stdin]   │ │
                        │  └─────────────────────────────┘ │
                        └────────────┬────────────────────┘
                                     │
                                     ▼
                        ┌─────────────────────────────────┐
                        │  8. score_fn()                   │
                        │  Compare vs baseline:            │
                        │  Inlining: bin_red% + speedup*10 │
                        │  RegAlloc: 5*speedup% + bin_red% │
                        └────────────┬────────────────────┘
                                     │
                                     ▼
                        ┌─────────────────────────────────┐
                        │  9. restore_source()             │
                        │  Put original .cpp back          │
                        │  Return score to OpenEvolve      │
                        └─────────────────────────────────┘
```

### Per-Benchmark Execution Commands

Each benchmark is run with CTMark reference inputs:

| Benchmark | Command | Runtime |
|-----------|---------|---------|
| bullet | `./bullet` (reads landscape.mdl, Taru.mdl) | ~3.4s |
| consumer-typeset | `./consumer-typeset -x -I data/include -D data/data -F data/font -C data/maps -H data/hyph large.lout` | ~0.1s |
| kimwitu | `./kimwitu -f test -o -v -s kcc inputs/f3.k inputs/f2.k inputs/f1.k` | ~0.06s |
| lencod | `./lencod -d data/encoder_small.cfg -p InputFile=data/foreman_part_qcif_444.yuv ...` | no runtime (hangs) |
| mafft | `./mafft -b 62 -g 0.100 -f 2.00 -h 0.100 -L < pyruvate_decarboxylase.fasta` | ~15s |
| spass | `./spass problem.dfg` | ~8s |
| sqlite3 | `./sqlite3 -init sqlite3rc :memory: < commands` (runs test1-15.sql) | ~0.002s |
| tramp3d-v4 | `./tramp3d-v4 --cartvis 1.0 0.0 --rhomin 1e-8 -n 4 --domain 32 32 32` | ~0.11s |

Each benchmark is run **5 times** and the **median** wall-clock time is used
(via `time.time()` around `subprocess.run()`). This reduces noise from OS
scheduling and process startup, though very short benchmarks (sqlite3 at 2ms)
remain unreliable.

## ASI — Actionable Side Information

ASI is a structured diagnostic feedback mechanism inspired by GEPA's "text
gradients". Instead of returning only a scalar score to the LLM, the evaluator
generates a multi-tier markdown narrative explaining *why* the code scored as it
did and *what to change*.

### Tiers

| Tier | Content | Overhead | Config |
|------|---------|----------|--------|
| **1** | Score decomposition + per-benchmark signal classification | Zero | Always on |
| **2** | Compiler statistics delta (`-stats` output vs baseline) | Zero | `EVOLVE_ENABLE_STATS=1` (default) |
| **3** | Runtime variance (CoV from 5 runs, STABLE/MODERATE/NOISY) | Zero | Always on |
| **4** | Hardware perf counters (instructions, cycles, cache/branch misses) | ~1s | `EVOLVE_ENABLE_PERF=1` |
| **5** | Optimization decision changes (`-pass-remarks-output` YAML diff) | ~20% | `EVOLVE_ENABLE_REMARKS=1` |

### Tier 1: Score Decomposition

Breaks the score into its components (speedup vs binary reduction) and
classifies each benchmark's signal reliability:

- **UNRELIABLE (<10ms)** — baseline runtime too short (e.g., sqlite3 at 2ms)
- **HIGH_VARIANCE (<100ms)** — borderline runtime stability
- **REAL (code changed)** — text section changed AND meaningful speedup
- **NOISE (same code)** — speedup without code change (measurement artifact)
- **MARGINAL** — small or no change

### Tier 2: Compiler Statistics Delta

Compares LLVM `-stats` output between evolved and baseline compilations. Shows
which optimization passes changed behavior (e.g., "inline - Number of functions
inlined: 1234 → 1567, +27%").

### Tier 5: Optimization Decision Changes

Compares per-decision optimization remarks (YAML) between evolved and baseline.
Identifies "flipped" decisions — functions that changed from inlined→rejected
or rejected→inlined — with their cost/threshold values. This gives the LLM
precise targets: "function X was rejected because cost=500 exceeds threshold=225;
lower the cost or raise the threshold."

The remarks parser uses a line-by-line state machine (not PyYAML) for
performance on 62MB files. Only `inline` and `loop-unroll` pass remarks are
extracted.

### Example ASI Output

```markdown
## Performance Analysis (Score: 8.78)

### Score Decomposition
Formula: binary_reduction% + (avg_speedup - 1) x 10
- Avg speedup: 1.0023x (+0.23%) x 0.1 = 0.02
- Binary reduction: 9.24% x 1.0 = 9.24

### Per-Benchmark Results
| Benchmark | Speedup | Text D | Binary D | Baseline RT | Signal |
|-----------|---------|--------|----------|-------------|--------|
| spass     | +0.3%   | +12.31%| +10.42%  | 8.1s        | REAL   |
| tramp3d-v4| -1.2%   | -3.45% | -2.11%   | 0.11s       | HIGH_VARIANCE |

### Optimization Decisions
**spass** (412 decisions changed vs baseline):
- 287 newly passed (were rejected)
- 125 newly rejected (were passed)

| Function | Callee | Direction | BL Cost/Thresh | Ev Cost/Thresh |
|----------|--------|-----------|----------------|----------------|
| memory_Free | allocBlock | now passed | 500/225 | -15025/225 |
```

## GEPA Integration

[GEPA](https://github.com/google-deepmind/gepa) (Generalist Evolutionary
Prompt Architect) is an optimization framework that uses LLM reflections to
evolve arbitrary text parameters. We integrate GEPA as an alternative to
OpenEvolve for driving LLVM heuristic evolution.

### Architecture

```
GEPA optimize_anything()
  │
  ├─ evaluator(code_str) → (score, {"Feedback": ASI_markdown})
  │    └─ Our make_evaluator(): patch LLVM, build, benchmark, generate ASI
  │
  └─ reflection_lm(prompt) → str
       └─ ManualLM: write prompt to disk, poll for response file
```

Key insight: GEPA's evaluator protocol accepts `(score, side_info_dict)` tuples.
We pass our ASI as `{"Feedback": asi_text}`, which GEPA includes in its
reflection prompt alongside the candidate code. This gives the LLM rich
diagnostic context for proposing improvements.

### Usage

```bash
# Manual mode: prompts appear as prompt_NNN.md, you write prompt_NNN.response.md
python gepa_run.py --task llvm_inlining --max-evals 10

# Auto mode for smoke testing (auto-responds with trivially modified code)
python gepa_run.py --task llvm_inlining --max-evals 2 --auto-respond
```

### Configuration

| Flag | Default | Description |
|------|---------|-------------|
| `--task` | (required) | `llvm_inlining`, `loop_unrolling`, or `regalloc_priority` |
| `--max-evals` | 10 | Maximum evaluator calls (seed + proposals) |
| `--prompts-dir` | `gepa_prompts` | Directory for prompt/response files |
| `--output-dir` | `<prompts-dir>/run` | GEPA state directory (for resume) |
| `--auto-respond` | off | Spawn background thread that auto-creates responses |

### GEPA vs OpenEvolve

| Feature | OpenEvolve | GEPA |
|---------|-----------|------|
| Population | MAP-Elites (50 candidates) | Pareto frontier |
| Feedback | Scalar score only → ASI via artifacts | Native side-info channel |
| LLM interface | ManualLLM (file-based) | ManualLM (file-based) |
| Hyperparameter tuning | Optuna inner-loop | Not integrated (future) |
| Resume | Checkpoint directory | `run_dir` state |

Both frameworks use our same evaluation pipeline (`llvm_bench.py`), so scores
are directly comparable.

## LLVM Hooks

### Inlining (`-use-evolved-inline-cost`)
- **Header:** `llvm/include/llvm/Analysis/EvolvedInlineCost.h`
- **Source:** `llvm/lib/Analysis/EvolvedInlineCost.cpp`
- **Hooked in:** `InlineCost.cpp` — `getInlineCost()` checks the `cl::opt` flag
- **Flag on:** `opt` (inlining is a middle-end IR pass)
- **Function:** `getEvolvedInlineCost(const InlineCostFeatures &F)` returns
  negative=inline, positive=don't inline. Mapped to `InlineCost::get(cost, 0)`.
- **Features:** callsite_cost, unsimplified_common_instructions, simplified_instructions,
  dead_blocks, constant_args, num_loops, nested_inlines, etc.

### RegAlloc Priority (`-use-evolved-regalloc-priority`)
- **Header:** `llvm/include/llvm/CodeGen/EvolvedRegAllocPriority.h`
- **Source:** `llvm/lib/CodeGen/EvolvedRegAllocPriority.cpp`
- **Hooked in:** `RegAllocGreedy.cpp` — `DefaultPriorityAdvisor::getPriority()`
  checks the `cl::opt` flag, extracts features, calls evolved function
- **Flag on:** `llc` (register allocation is a CodeGen pass)
- **Function:** `computeEvolvedRegAllocPriority(const RegAllocPriorityFeatures &F)`
  returns unsigned priority (higher = allocated first)
- **Features:** Size, Stage, IsLocal, ForceGlobal, AllocationPriority,
  HasPreference, NumAllocatable, BeginDist, EndDist, NumInstrs, IsCSR

### Bit-packed priority encoding (regalloc)

The default LLVM priority uses a bit-packed unsigned:
```
Bit 31:    1 = RS_Assign (above RS_Split)
Bit 30:    1 = HasPreference (register hint)
Bit 29:    GlobalBit (global ranges above local)
Bits 24-28: AllocationPriority (register class, 5 bits)
Bits 0-23:  Size or BeginDist (clamped to 24 bits)
```
This creates hard priority boundaries. Structural changes to this encoding
consistently hurt performance in experiments.

## Hyperparameter Convention

Evolved C++ code can declare tunable numeric knobs via comments:

```cpp
// [hyperparam]: ae-inline-base-threshold, int, 50, 300
static cl::opt<int> BaseThreshold("ae-inline-base-threshold", cl::init(100), ...);
```

Format: `// [hyperparam]: flag-name, type, min, max`

When `EVOLVE_OPTUNA_TRIALS > 0`, the evaluator:
1. Parses `[hyperparam]` annotations from the evolved C++ code
2. Creates an Optuna study with one parameter per annotation
3. Runs N trials on a 3-benchmark subset (sqlite3, spass, tramp3d-v4)
4. Each trial: compile subset with trial params as LLVM flags, score
5. Best params are passed as flags in the final full-suite evaluation

Example: Optuna suggests `-ae-inline-base-threshold=173`, which is passed
to `opt` (or `llc` for regalloc flags) during compilation.

## Configuration

`EvalConfig` dataclass supports both programmatic and env-var configuration:

```python
from mlirAgent.evolve.tasks.llvm_bench import EvalConfig

# From environment variables
config = EvalConfig.from_env("llvm/lib/Analysis/EvolvedInlineCost.cpp")

# Programmatic with overrides
config = EvalConfig.from_env(
    "llvm/lib/Analysis/EvolvedInlineCost.cpp",
    optuna_trials=5,
    opt_timeout=60,
)
```

| Env Var | Default | Description |
|---------|---------|-------------|
| `LLVM_SRC_PATH` | (required) | LLVM source tree root |
| `EVOLVE_BUILD_DIR` | (required) | LLVM ninja build directory |
| `EVOLVE_OPT_TIMEOUT` | 120 | Per-benchmark opt/llc timeout (seconds) |
| `EVOLVE_OPTUNA_TRIALS` | 20 | Optuna trials (0 = disable) |
| `EVOLVE_ENABLE_STATS` | 1 | Tier 2: collect `-stats` output |
| `EVOLVE_ENABLE_PERF` | 0 | Tier 4: collect perf counters |
| `EVOLVE_ENABLE_REMARKS` | 0 | Tier 5: collect optimization remarks (~20% overhead) |

## Task Structure

```
src/mlirAgent/evolve/
  manual_run.py                  # OpenEvolve orchestrator: --auto/--wait/--resume
  gepa_run.py                   # GEPA orchestrator: --auto-respond
  gepa_adapter.py               # GEPA evaluator bridge (score, side_info)
  gepa_manual_lm.py             # File-based LLM for GEPA
  tasks/
    llvm_bench.py                # Shared: EvalConfig, compile, baseline, Optuna, ASI
    llvm_inlining/
      evaluate.py                # _score(): bin_red% + speedup*10
      initial.cpp                # Seed: sums heuristic features - threshold
      task.py                    # OpenEvolve Task class
      benchmarks/
        compile_testsuite.sh     # Script to build .bc from llvm-test-suite
        testsuite/               # .bc files (gitignored, built locally)
          data/                  # Runtime input data per benchmark
    loop_unrolling/
      evaluate.py                # _score(): 5*speedup% + bin_red%
      initial.cpp                # Seed: LLVM default unroll heuristic
    regalloc_priority/
      evaluate.py                # _score(): 5*speedup% + bin_red%
      initial.cpp                # Seed: LLVM default bit-packed priority
      baseline_regalloc.json     # Separate baseline (uses -use-evolved-* on llc)
  README.md                      # This file
configs/
  frameworks/manual.yaml         # OpenEvolve config (pop=10, 1 island, seed=42)
experiments/                     # Output (gitignored)
  run_YYYYMMDD_HHMMSS/
```

### Adding a New Task

1. Create `tasks/my_task/` with `initial.cpp` and `evaluate.py`
2. In `evaluate.py`, define `_score(total_binary, baseline_binary, speedups)`
3. Call shared functions from `llvm_bench.py` with the right evolved flags
4. Add entry to `EXAMPLES` dict in `manual_run.py`
5. If the evolved code affects `llc` (not `opt`), use `flag_target="llc"` in
   `optuna_tune()` and pass flags via `evolved_llc_flags`

## Scoring Formulas

**Inlining:** `binary_reduction_pct + (avg_speedup - 1.0) * 10`
- Primary signal: linked binary size reduction vs baseline
- Secondary: small bonus for runtime improvement
- Comparable to Magellan (ICML 2025) binary reduction metric

**RegAlloc:** `5.0 * speedup_pct + 1.0 * binary_reduction_pct`
- Primary signal: runtime improvement (regalloc most affects execution speed)
- Secondary: binary size reduction
- Warning: dominated by measurement noise for short-running benchmarks

## Experiment Results (CTMark, Feb 2026)

### LLVM Inlining
| Experiment | Responder | Optuna | Iters | Best Score | Binary Reduction |
|-----------|-----------|--------|-------|------------|-----------------|
| Exp A | Claude | No | 10 | 8.65 | 8.78% |
| Exp C | Auto | 5 trials | 10 | 8.66 | 8.41% |
| Exp D | Claude | 5 trials | 11 | **8.78** | **9.24%** |

All match Magellan's reported range (4.27-8.79%) with only 10 iterations.
Claude + Optuna combined is slightly better than either alone.

### RegAlloc Priority
| Experiment | Measurement | Iters | Best Score | Notes |
|-----------|-------------|-------|------------|-------|
| Exp E | Single run | 8 | 63.39 | **INVALIDATED** (sqlite3 2ms noise) |
| Exp F | Median-of-5 | 11 | 8.82 | Pressure-proportional priority |

Exp E results were entirely from sqlite3 measurement noise (1.89x "speedup"
was an artifact of 2ms runtime variance). After fixing `run_benchmark()` to
use median-of-5 runs, the only positive innovation was pressure-proportional
priority: boosting global ranges in constrained register classes.

### Key Insights
- Os-level inlining hurts tramp3d-v4 (C++ templates need inlining)
- Code structure > hyperparameters for peak inlining score
- Optuna adds robustness (100% positive scores vs 80%)
- RegAlloc priority bit-packed encoding is fragile — structural changes hurt
- Benchmarks under 10ms are unreliable even with median-of-5 runs
