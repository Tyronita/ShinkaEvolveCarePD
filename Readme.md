# ShinkaEvolveCarePD

**Automated algorithm discovery for Parkinson's Disease gait severity classification**  
Using [ShinkaEvolve](https://github.com/SakanaAI/ShinkaEvolve) (Sakana AI) × [CARE-PD](https://github.com/TaatiTeam/CARE-PD) (Taati Lab, University of Toronto)

---

## Motivation

**Parkinson's Disease (PD)** is the second most common neurodegenerative disorder, affecting over 10 million people worldwide. One of its most debilitating symptoms is gait impairment — changes in how patients walk that worsen as the disease progresses.

Clinicians score gait severity using the **UPDRS Gait Scale** (Unified Parkinson's Disease Rating Scale):
- **0** — Normal gait
- **1** — Mild impairment (slight shuffle, reduced arm swing)
- **2** — Severe impairment (significant festination, freezing, or falls)

This assessment is currently **subjective, time-consuming, and requires a specialist in the room**. Patients in rural areas or developing countries often go unmonitored between clinic visits. An automated system that predicts UPDRS gait score from wearable or markerless motion capture could enable remote, continuous, objective monitoring.

The **CARE-PD dataset** provides 3D body pose sequences (SMPL format) from 781 walking bouts across 9 international cohorts, paired with expert UPDRS gait labels. This is one of the largest and most diverse PD gait datasets available.

---

## What This Repo Does

We apply **ShinkaEvolve** — Sakana AI's LLM-driven evolutionary optimization framework — to automatically discover high-performing algorithms for UPDRS gait score prediction, without manual architecture search.

### How ShinkaEvolve Works

ShinkaEvolve maintains an **archive of programs** (Python scripts). Each generation:
1. Samples one or two parent programs from the archive
2. Asks a frontier LLM (Claude, Gemini) to **mutate** the program via diff/patch, full rewrite, or crossover
3. Evaluates the new program on the fitness metric (Macro-F1)
4. Adds it to the archive if it runs correctly

The system uses a **UCB multi-armed bandit** to adaptively select which LLM and mutation operator to use, based on what's been working. A **meta scratchpad** distills insights from the best programs into the system prompt, propagating semantic knowledge across the evolutionary tree.

**Key advantage**: LLMs understand code semantics — mutations aren't random bit-flips, they're targeted, meaningful changes like "add an attention pooling layer" or "switch to 9D rotation representation".

### The Classification Pipeline

```
Walking bout (SMPL pose sequence: T frames × 72 parameters)
    ↓
6D rotation representation  [Zhou et al. 2019 — continuous rotation]
    ↓
Crop/pad to fixed 60 frames
    ↓
MotionCLIP Transformer encoder  →  512-dim motion embedding
    ↓
MLP classifier  →  3-class logits (UPDRS 0 / 1 / 2)
    ↓
FocalLoss with inverse-frequency class weighting
```

The entire pipeline above is wrapped in an **EVOLVE-BLOCK** — ShinkaEvolve can mutate any part of it. The evaluation harness (data loading, 6-fold cross-validation, metric computation) is immutable.

---

## Task 1 Results

> Full analysis: [`care_pd_task1_results.md`](care_pd_task1_results.md)

### Evolution Run Summary (`care_pd_full`, ~90 generations on RunPod GPU)

| Genome | Macro-F1 | F1-Normal | F1-Mild | F1-Severe | Notes |
|---|---:|---:|---:|---:|---|
| Baseline (RandomForest) | 0.565 | 0.702 | 0.495 | 0.497 | Hand-engineered features |
| Initial genome (MotionCLIP) | 0.565 | 0.702 | 0.495 | 0.497 | Task 1 seed |
| gen_13 | 0.409 | 0.502 | 0.500 | 0.226 | Early improvement on severe |
| gen_24 | 0.446 | 0.517 | 0.461 | 0.360 | Better severe detection |
| gen_21 | 0.623 | 0.774 | 0.584 | **0.512** | Best severe F1 — Pareto-optimal |
| gen_28 | 0.634 | 0.789 | 0.634 | 0.481 | Early peak |
| gen_31 | 0.636 | 0.791 | 0.643 | 0.473 | Incremental gain |
| gen_55 | 0.614 | 0.739 | 0.582 | **0.521** | Strong severe detection |
| gen_64 | 0.617 | 0.742 | 0.647 | 0.461 | — |
| **gen_65** | **0.655** | **0.772** | **0.673** | **0.521** | **Best overall — new peak** |
| gen_69 | 0.619 | 0.779 | 0.600 | 0.478 | — |
| gen_88 | 0.616 | 0.769 | 0.594 | 0.484 | Late-run strong result |
| CARE-PD LOSO SOTA | 0.68+ | — | — | — | Published target |
| MIDA SOTA | 0.74+ | — | — | — | Upper bound |

**+16.0% improvement** over the starting baseline (0.565 → 0.655) with zero manual architecture design after the seed program.

### Evolution Lineage

The winning genomes emerged from a non-monotonic search — many intermediate programs scored **below** the baseline before the archive converged on strong solutions:

```
Generation 0  →  0.346  (initial random mutations, degraded)
Generation 8  →  0.315  (worse still)
Generation 13 →  0.409  (first real progress on severe class)
Generation 21 →  0.623  ← breakthrough
Generation 28 →  0.634
Generation 31 →  0.636
Generation 36 →  0.612
Generation 55 →  0.614  (strong severe PD detection)
Generation 65 →  0.655  ← best overall
Generation 69 →  0.619
Generation 88 →  0.616
```

After gen_28, the archive plateaued around 0.59–0.62 for ~30 generations before gen_65 broke through to a new best.

### Pareto Trade-off: Overall vs. Severe PD

Two genomes represent distinct clinical trade-offs:

- **gen_65** (0.655 macro-F1, **0.521 F1-severe**): Best on both dimensions — strongest overall
- **gen_21** (0.623 macro-F1, **0.512 F1-severe**): Pareto-optimal at lower compute; similar severe-class performance

In Parkinson's care, **missing a severe patient is more costly than a false alarm**. A clinical system should prefer gen_65's operating point.

### Generalization Across Subject Cohorts (gen_65 — best overall, per fold)

| Fold | Macro-F1 |
|---:|---:|
| 1 | 0.457 |
| 2 | 0.482 |
| 3 | 0.560 |
| 4 | **0.878** |
| 5 | 0.419 |
| 6 | 0.488 |

High variance across folds (0.42–0.88) indicates cohort-specific variation. Fold 4 is an outlier; improving folds 1, 2, and 5 is the key target for Task 2.

### Task v2 Smoke Test (3 generations, local)

| Genome | Macro-F1 | F1-Severe |
|---|---:|---:|
| initial (best of 3 runs) | 0.596 | 0.510 |
| gen_2 | **0.608** | 0.503 |

The v2 seed is a stronger starting point than v1. Gen_2 already approaches Task 1's early peak after just 2 generations.

---

## Repository Structure

```
ShinkaEvolveCarePD/
├── care_pd_task/           # Task 1 — MotionCLIP + MLP seed
│   ├── initial.py          # Baseline genome (seed program)
│   ├── evaluate.py         # Standalone eval harness (no shinka imports)
│   ├── shinka_task.yaml    # Evolution config (models, budget, islands)
│   └── leaderboard.csv     # Per-genome results log (59 evaluated genomes)
├── care_pd_task_v2/        # Task 2 — improved seed architecture
│   ├── initial.py
│   ├── evaluate.py
│   └── shinka_task.yaml
├── results/
│   └── care_pd_full/       # Full RunPod evolution run (~90 genomes)
├── care_pd_task1_results.md  # Detailed Task 1 results analysis
├── runpod_startup.sh       # Authoritative RunPod pod startup script
└── CLAUDE.md               # Claude Code project instructions
```

---

## Setup

### Prerequisites

```bash
# Install build tool and ShinkaEvolve skills for Claude Code
pip install -r requirements.txt
npx skills add SakanaAI/ShinkaEvolve --skill '*' -a claude-code -y
```

### Install ShinkaEvolve

```bash
git clone https://github.com/SakanaAI/ShinkaEvolve
cd ShinkaEvolve
uv venv --python 3.11
.venv/Scripts/activate          # Windows
# source .venv/bin/activate     # Linux/Mac
uv pip install -e .
cd ..
uv pip install datasets python-dotenv
```

### Install CARE-PD

```bash
git clone https://github.com/TaatiTeam/CARE-PD.git
cd CARE-PD
uv pip install -r requirements.txt
uv pip install torch==2.6.0+cu118 torchvision==0.21.0+cu118 \
  --index-url https://download.pytorch.org/whl/cu118

# Download dataset (requires HF_TOKEN)
mkdir -p assets/datasets
huggingface-cli download vida-adl/CARE-PD --repo-type dataset \
  --local-dir ./assets/datasets
python data/smpl_reader.py \
  --dataset PD-GaM BMCLab 3DGait T-SDU-PD DNE E-LC KUL-DT-T T-LTC T-SDU
cd ..
```

### Secrets

Copy and populate:
- `.env.local.example` → `.env` (local dev): `RUNPOD_API_KEY`, `ANTHROPIC_API_KEY`, `GEMINI_API_KEY`
- `.env.remote.example` → [RunPod Secrets console](https://console.runpod.io/user/secrets/): `HF_TOKEN`, `ANTHROPIC_API_KEY`, `GEMINI_API_KEY`

In the pod/endpoint environment variables, map secrets:

| Variable | Value |
|---|---|
| `ANTHROPIC_API_KEY` | `{{ RUNPOD_SECRET_ANTHROPIC_API_KEY }}` |
| `GEMINI_API_KEY` | `{{ RUNPOD_SECRET_GEMINI_API_KEY }}` |
| `HF_TOKEN` | `{{ RUNPOD_SECRET_HF_TOKEN }}` |
| `GITHUB_TOKEN` | `{{ RUNPOD_SECRET_GITHUB_TOKEN }}` |

---

## Running Experiments

### Local baseline test (instant feedback, no Shinka)

```bash
ShinkaEvolve/.venv/Scripts/python care_pd_task/evaluate.py \
  --program_path care_pd_task/initial.py \
  --results_dir care_pd_task/results_test
```

### Smoke test (3 generations, verify full pipeline)

```bash
PYTHONUTF8=1 PYTHONIOENCODING=utf-8 \
ShinkaEvolve/.venv/Scripts/shinka_run \
  --task-dir care_pd_task \
  --results_dir care_pd_task/results_smoke \
  --num_generations 3 \
  --config-fname shinka_task.yaml \
  --max-evaluation-jobs 1
```

### Full evolution run (local)

```bash
PYTHONUTF8=1 PYTHONIOENCODING=utf-8 \
ShinkaEvolve/.venv/Scripts/shinka_run \
  --task-dir care_pd_task \
  --results_dir results/care_pd_local \
  --num_generations 30 \
  --config-fname shinka_task.yaml
```

### RunPod (GPU pod)

Set startup command in RunPod template:
```
bash /workspace/ShinkaEvolveCarePD/runpod_startup.sh
```

Key environment variables:

| Variable | Default | Description |
|---|---|---|
| `NUM_GENERATIONS` | 100 | Shinka generations |
| `MAX_EVAL_JOBS` | 4 | Parallel workers |
| `MAX_API_COSTS` | 50.0 | API cost cap ($) |
| `SMOKE_ONLY` | false | Exit after validation |

---

## Model Pool

LLM models used for mutation (Gemini + Anthropic only):

```yaml
llm_models:
  - "claude-opus-4-6"
  - "claude-sonnet-4-6"
  - "gemini-2.5-pro"
  - "gemini-2.5-flash"
embedding_model: "gemini-embedding-exp-03-07"
```

---

## Research Context

This project is grounded in the ShinkaEvolve paper (Sakana AI). Key design decisions informed by the paper:

- **Starting simple**: The seed genome is intentionally minimal — a frozen MotionCLIP encoder + thin MLP — to give the evolutionary search maximum room for novelty.
- **Dual fitness as Pareto front**: Macro-F1 (clinical prediction) and MPJPE (motion reconstruction) are logged separately, not combined into a single weighted score, to reveal the convex hull of trade-offs.
- **Sample efficiency**: ~59 successful GPU evaluations achieved a 16% improvement. This compares favorably to grid search or NAS approaches requiring thousands of evaluations.
- **The "problem-problem"**: The fitness function itself (Macro-F1 on BMCLab fold 4) may be overly narrow. Future work should consider multi-cohort evaluation and proxy metrics that are easier to optimize early.

For a deep dive on how ShinkaEvolve works, see `Research/MLST - Robert Lange - Sakana AI - Transcript.md`.

---

## Citation

If you use CARE-PD:
```
@inproceedings{care-pd,
  title={CARE-PD: ...},
  author={...},
  booktitle={...},
  year={2024}
}
```

If you use ShinkaEvolve:
```
@article{shinka-evolve,
  title={...},
  author={Sakana AI},
  year={2025}
}
```
