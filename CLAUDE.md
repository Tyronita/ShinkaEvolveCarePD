# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repo is an orchestration layer for running **ShinkaEvolve** evolutionary experiments on the **CARE-PD** Parkinson's Disease gait dataset. The agent manages genome evaluation, RunPod GPU pod lifecycle, metrics logging, and leaderboard benchmarking.

Key external repos (cloned separately, not in this directory):
- `ShinkaEvolve/` — evolutionary optimization framework
- `CARE-PD/` — dataset and model code (MotionAGFormer, SMPL gait meshes)

## Environment Setup

```bash
# Install uv (build tool)
pip install -r requirements.txt

# Install ShinkaEvolve
git clone https://github.com/SakanaAI/ShinkaEvolve
cd ShinkaEvolve
uv venv --python 3.11
.venv/Scripts/activate          # Windows
# source .venv/bin/activate     # Linux/Mac
uv pip install -e .
cd ..

# Additional dependencies
uv pip install datasets python-dotenv

# Install CARE-PD
git clone https://github.com/TaatiTeam/CARE-PD.git
cd CARE-PD
uv pip install -r requirements.txt
uv pip install torch==2.6.0+cu118 torchvision==0.21.0+cu118 --index-url https://download.pytorch.org/whl/cu118

# Download dataset (requires HF_TOKEN)
mkdir -p assets/datasets
huggingface-cli download vida-adl/CARE-PD --repo-type dataset --local-dir ./assets/datasets
python data/smpl_reader.py --dataset PD-GaM BMCLab 3DGait T-SDU-PD DNE E-LC KUL-DT-T T-LTC T-SDU
cd ..
```

## Secrets

Copy and populate the example env files:
- `.env.local.example` → `.env` (local dev):
  - `RUNPOD_API_KEY` — pod lifecycle management
  - `ANTHROPIC_API_KEY` — Claude mutation/meta models
  - `GEMINI_API_KEY` — Gemini mutation/meta models
- `.env.remote.example` → RunPod Secrets console at `console.runpod.io/user/secrets/`: create `HF_TOKEN`, `ANTHROPIC_API_KEY`, `GEMINI_API_KEY`

RunPod secrets are **not** auto-injected. In the pod/endpoint environment variables section, you must manually map each secret to its env var name:

| Environment Variable | Value (click the secret icon) |
|---|---|
| `ANTHROPIC_API_KEY` | `{{ RUNPOD_SECRET_ANTHROPIC_API_KEY }}` |
| `GEMINI_API_KEY` | `{{ RUNPOD_SECRET_GEMINI_API_KEY }}` |
| `HF_TOKEN` | `{{ RUNPOD_SECRET_HF_TOKEN }}` |

Shinka loads `.env` from both `ShinkaEvolve/.env` and the task's working directory (task dir wins). Place keys in `ShinkaEvolveCarePD/.env` so they apply to all task runs.

**Model pool for this project** (Gemini + Anthropic only — no OpenAI):
```yaml
llm_models:
  - "claude-opus-4-6"
  - "claude-sonnet-4-6"
  - "gemini-2.5-pro"
  - "gemini-2.5-flash"
embedding_model: "gemini-embedding-exp-03-07"
```

Model names must match `ShinkaEvolve/shinka/llm/providers/pricing.csv` exactly — **no `-preview` suffix**.

## Dataset

```python
from datasets import load_dataset
dataset = load_dataset("vida-adl/CARE-PD")
```

Or use `read_dataset.py` which loads `.env` and authenticates via `HF_TOKEN` before downloading.

## Architecture

This project is structured as an **autonomous orchestration agent**:

1. **Genome Evaluation** — ShinkaEvolve generates genomes; each is evaluated on two fitness metrics:
   - *Clinical Prediction*: Macro-F1 or RMSE for UPDRS gait severity from CARE-PD features/SMPL meshes
   - *Motion Reconstruction*: MPJPE (mm) for 3D pose reconstruction (baselines: ~60.8 mm pretrained, ~7.5 mm fine-tuned MotionAGFormer)

2. **Pod Management** — RunPod GPU pods are spawned/stopped dynamically via `RUNPOD_API_KEY`. Pods run parallel genome evaluations with no master/worker hierarchy. Checkpoints must be written to persistent storage to survive pod restarts.

3. **Execution Loop**:
   - Generate genomes → assign to pods → evaluate both fitness metrics → log to leaderboard format → save checkpoints → rebalance pod allocation

4. **Metrics / Leaderboard** — Per genome: generation number, genome ID, Macro-F1 (%), MPJPE (mm), optional cross-cohort breakdowns. Versioned checkpoints for reproducibility.

## ShinkaEvolve Research Context

Source: `Research/MLST - Robert Lange - Sakana AI - Transcript.md` — interview with Robert Lange (Sakana AI founding researcher) on the ShinkaEvolve paper.

### How ShinkaEvolve works (relevant to implementation decisions)

- **Archive-based evolution**: Maintains a database of programs. Each generation samples parent program(s), asks an LLM to mutate/diff/rewrite/crossover them, evaluates the result, and adds it back to the archive. This is the core loop to replicate in `evaluate.py`.
- **Three mutation operators**: (1) diff/patch — targeted edits; (2) full rewrite — more diversity, escapes local optima; (3) crossover — two parent programs combined. Diversity of operators matters.
- **Evolvable code markers**: Parts of the program marked as mutable vs. immutable (e.g., imports, evaluation harness stay fixed). Mutations that touch immutable regions are rejected and resampled.
- **Meta scratch pad**: Program summaries are extracted and distilled into global insights that become part of the system prompt — Shinka's mechanism for propagating semantic knowledge across the tree.
- **UCB bandit for model selection**: Each LLM provider is one arm of a multi-armed bandit. The system tracks which model yielded improvements for similar parent nodes and adaptively weights model selection. No single model dominates; stochasticity is preserved across arms.
- **Islands / knowledge diffusion trade-off**: Too much diffusion homogenises the population; too little wastes stepping stones. Current approach tries to auto-tune this, but it remains problem-dependent.

### Starting solution matters

Starting from an already-optimised solution risks local optima with low novelty. Starting from an impoverished (even empty) solution gives more room for diversity but requires more generations. For CARE-PD, prefer a minimal but correct initial genome over a hand-tuned one.

### The "problem-problem"

The hardest open research question: current systems optimise solutions to a *fixed* problem. Real breakthroughs often require *reformulating* the problem first (surrogate problems, recursive reductions). For CARE-PD this means: don't over-constrain the fitness function early — allow proxy formulations that are easier to optimise, then tighten.

### Multi-objective fitness in the context of this paper

Shinka can illuminate a *convex hull* of trade-offs rather than a single optimum (demonstrated on MoE load-balancing loss design). For CARE-PD, the dual fitness (Macro-F1 + MPJPE) should be treated as a Pareto front, not a single weighted sum — log both separately and let the archive retain Pareto-optimal genomes.

### Scaling philosophy

The authors intentionally kept evaluation count low (~150–200 LLM calls for circle packing state-of-the-art). For CARE-PD with expensive GPU evaluations on RunPod, this sample efficiency is the main justification for using Shinka over random search. Budget pods accordingly.

## Task Directory

The live ShinkaEvolve task lives in `care_pd_task/`:

```
care_pd_task/
├── initial.py          # Baseline genome (mean+std features + RandomForest)
├── evaluate.py         # Standalone harness (no shinka import — works under any Python)
├── shinka_task.yaml    # Full task config (models, islands, budget, system message)
└── leaderboard.csv     # Persistent per-genome results (auto-created on first run)
```

### Key fixes baked in permanently

- **`ShinkaEvolve/shinka/launch/scheduler.py`** — `sys.executable` replaces hardcoded `"python"` so subprocesses always use the same venv Python that launched Shinka (Windows-safe).
- **`evaluate.py`** — walks up ancestor directories to find `CARE-PD/` and `care_pd_task/` regardless of where Shinka copies the file (it copies to `results_dir/evaluate.py`).
- **`evaluate.py`** — zero shinka imports: runs standalone under the venv Python.

### Running locally

```bash
# From ShinkaEvolveCarePD/ — always run from this directory
PYTHONUTF8=1 PYTHONIOENCODING=utf-8 \
ShinkaEvolve/.venv/Scripts/shinka_run \
  --task-dir care_pd_task \
  --results_dir results/care_pd_local \
  --num_generations 30 \
  --config-fname shinka_task.yaml
```

Smoke test (3 generations, verify pipeline):
```bash
PYTHONUTF8=1 PYTHONIOENCODING=utf-8 \
ShinkaEvolve/.venv/Scripts/shinka_run \
  --task-dir care_pd_task \
  --results_dir care_pd_task/results_smoke \
  --num_generations 3 \
  --config-fname shinka_task.yaml \
  --max-evaluation-jobs 1
```

Baseline standalone test (no Shinka, instant feedback):
```bash
ShinkaEvolve/.venv/Scripts/python care_pd_task/evaluate.py \
  --program_path care_pd_task/initial.py \
  --results_dir care_pd_task/results_test
```

### Leaderboard columns

`care_pd_task/leaderboard.csv` tracks every evaluated genome:

| Column | Description |
|---|---|
| `genome_id` | `gen_N` from program path |
| `macro_f1` | Primary fitness (higher = better). Baseline: ~0.565 |
| `f1_class0/1/2` | Per-class F1 for UPDRS 0/1/2. Class 2 (severe PD) is clinically most important |
| `mpjpe_mm` | `"N/A"` in Phase 1; float (mm) in Phase 2 when MotionAGFormer is added |
| `fold_f1_json` | Per-fold macro-F1 — reveals generalisation across subjects |
| `confusion_matrix` | In `metrics.json` public fields |
| `eval_time_s` | Wall-clock seconds — flag genomes >60s as too slow |

## Running on RunPod

### Startup script

`runpod_startup.sh` is the authoritative RunPod startup script. Set it as the pod startup command:

```
bash /workspace/ShinkaEvolveCarePD/runpod_startup.sh
```

It handles in order: repo clone/pull → venv + deps → sys.executable patch → dataset download → pipeline validation → ShinkaEvolve docs (port 8888) → Web UI (port 8080) → git persistence daemon → evolution run → final git push.

**Key env vars** (set directly as pod environment variables — not secrets):
| Var | Default | Description |
|---|---|---|
| `NUM_GENERATIONS` | 100 | Shinka generations to run |
| `MAX_EVAL_JOBS` | 4 | Parallel eval workers |
| `MAX_API_COSTS` | 50.0 | API cost cap ($) |
| `SMOKE_ONLY` | false | If true, exit after validation |
| `WEBUI_PORT` | 8080 | shinka_visualize port |
| `DOCS_PORT` | 8888 | ShinkaEvolve docs port |

### RunPod secrets → environment variables

In the pod/endpoint template, map each secret under "Environment Variables":

| Variable | Value |
|---|---|
| `ANTHROPIC_API_KEY` | `{{ RUNPOD_SECRET_ANTHROPIC_API_KEY }}` |
| `GEMINI_API_KEY` | `{{ RUNPOD_SECRET_GEMINI_API_KEY }}` |
| `HF_TOKEN` | `{{ RUNPOD_SECRET_HF_TOKEN }}` |
| `GITHUB_TOKEN` | `{{ RUNPOD_SECRET_GITHUB_TOKEN }}` |
| `GITHUB_EMAIL` | `{{ RUNPOD_SECRET_GITHUB_EMAIL }}` |

`RUNPOD_API_KEY` stays in the local `.env` only — used by the orchestrator, not the pod.

### Persistence strategy

RunPod ephemeral storage disappears on pod termination. Two layers of persistence:

1. **Network volume** (`/workspace`) — attach a RunPod network volume at `/workspace`. All repos, checkpoints, and results live here across pod restarts ($0.07/GB/month).
2. **Git push** — `runpod_startup.sh` runs a background daemon that commits and pushes `leaderboard.csv` + evolved genomes every 30 minutes via `GITHUB_TOKEN`. Final push runs after the evolution completes.

Create a GitHub PAT (Personal Access Token) with `repo` scope, add it as `RUNPOD_SECRET_GITHUB_TOKEN`, and set `RUNPOD_SECRET_GITHUB_REPO` to `YourUser/ShinkaEvolveCarePD`.

### ShinkaEvolve Web UI (remote access)

The startup script launches `shinka_visualize` on port 8080. SSH tunnel to access locally:
```bash
ssh -L 8080:localhost:8080 root@<pod-ip> -p <ssh-port> -i ~/.ssh/your_key
# then open http://localhost:8080
```

### ShinkaEvolve docs hosting

The startup script serves `ShinkaEvolve/docs/` via `python -m http.server` on port 8888.
SSH tunnel: `ssh -L 8888:localhost:8888 root@<pod-ip> -p <ssh-port> -i ~/.ssh/your_key`

### CARE-PD inference with benchmark models (Phase 2)

The CARE-PD benchmark models (MotionAGFormer, MixSTE, etc.) require h36m preprocessed NPZ files at `CARE-PD/assets/datasets/h36m/` — these are **not** in the HF dataset and must be downloaded separately from Harvard Dataverse. Without them, only the ShinkaEvolve pipeline (which uses raw SMPL PKL files) can run.

Phase 1 ShinkaEvolve evolution works fully without h36m data. h36m is needed for Phase 2 (MPJPE metric via MotionAGFormer fine-tuning).

## ShinkaEvolve Skills

The Claude Code ShinkaEvolve skills are installed via:
```bash
npx skills add SakanaAI/ShinkaEvolve --skill '*' -a claude-code -a codex -y
```

Relevant skills: `shinka-setup`, `shinka-run`, `shinka-inspect`, `shinka-convert`.
