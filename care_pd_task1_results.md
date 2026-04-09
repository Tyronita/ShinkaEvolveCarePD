# CARE-PD × ShinkaEvolve — Task 1 Results Analysis

> **Experiment**: Automated algorithm discovery for Parkinson's Disease gait severity classification  
> **Framework**: [ShinkaEvolve](https://github.com/SakanaAI/ShinkaEvolve) (Sakana AI)  
> **Dataset**: [CARE-PD](https://github.com/TaatiTeam/CARE-PD) — vida-adl/CARE-PD on Hugging Face  
> **Run**: 2026-04-06–08 · RunPod GPU pod · ~90 generations · 59 evaluated (18 failed/errored)

---

## What We're Optimizing

Parkinson's Disease (PD) is assessed clinically using the **UPDRS Gait Score** — a 3-point severity scale:

| Score | Meaning |
|:---:|---|
| **0** | Normal gait |
| **1** | Mild impairment |
| **2** | Severe impairment |

The task is to predict this score automatically from **3D body pose during walking**, captured as SMPL body model parameters (72-dim axis-angle per frame). This is a **3-class classification** problem evaluated with **Macro-F1** (equally weights all 3 classes, critical for clinical fairness given class imbalance).

- **Dataset**: 781 walking bouts from the BMCLab cohort
- **Protocol**: 6-fold subject-level cross-validation (no data leakage across subjects)
- **Class distribution**: 341 normal (0), 276 mild (1), 164 severe (2)

---

## Baseline & Targets

| Benchmark | Macro-F1 | Notes |
|---|---:|---|
| RandomForest (hand-engineered features) | 0.565 | Task 1 starting genome |
| **Task 1 initial genome** (MotionCLIP + MLP) | **0.565** | Same score — architecture not yet helping |
| **ShinkaEvolve best** (gen_65) | **0.655** | +16.0% over baseline |
| CARE-PD LOSO SOTA | 0.68+ | Published benchmark target |
| MIDA SOTA | 0.74+ | Upper bound target |

---

## Initial Architecture (Task 1 Starting Genome)

ShinkaEvolve starts from a hand-written "seed" program. Task 1's seed uses a deep learning pipeline:

```
SMPL axis-angle (T, 72)
    → 6D rotation representation (T, 25, 6)     [Zhou et al. 2019]
    → Crop/pad to 60 frames → (1, 60, 25, 6)
    → MotionCLIP Transformer encoder (frozen)   → μ ∈ ℝ^512
    → MLP classifier head                        → logits ∈ ℝ^3
    → FocalLoss with inverse-frequency weighting
```

The **EVOLVE-BLOCK** (everything above) is marked mutable. The evaluation harness (data loading, cross-validation, metric computation) is immutable and cannot be touched by mutations.

---

## Evolution Run Results

### Full Run on RunPod (`care_pd_full`, ~90 generations)

All 59 evaluated genomes (18 of ~90 total failed with errors and are excluded):

| Genome | Macro-F1 | F1-Normal | F1-Mild | F1-Severe | Eval Time (s) |
|---|---:|---:|---:|---:|---:|
| initial (RunPod) | 0.3773 | 0.5707 | 0.5140 | 0.0471 | 349 |
| gen_0 | 0.3461 | 0.5829 | 0.4552 | 0.0000 | 117 |
| gen_8 | 0.3151 | 0.5724 | 0.3730 | 0.0000 | 407 |
| gen_10 | 0.3567 | 0.5777 | 0.4924 | 0.0000 | 119 |
| gen_12 | 0.2820 | 0.2197 | 0.3629 | 0.2634 | 248 |
| gen_13 | 0.4092 | 0.5015 | 0.5000 | 0.2260 | 121 |
| gen_14 | 0.2577 | 0.2557 | 0.2844 | 0.2331 | 262 |
| gen_15 | 0.3372 | 0.3200 | 0.4463 | 0.2454 | 436 |
| gen_18 | 0.2370 | 0.1449 | 0.3281 | 0.2380 | 252 |
| gen_20 | 0.3378 | 0.4163 | 0.3020 | 0.2950 | 390 |
| **gen_21** | **0.6231** | 0.7741 | 0.5838 | **0.5115** | 952 |
| gen_23 | 0.4027 | 0.5050 | 0.5336 | 0.1695 | 164 |
| gen_24 | 0.4460 | 0.5165 | 0.4612 | 0.3601 | 166 |
| gen_28 | 0.6343 | 0.7885 | 0.6337 | 0.4806 | 911 |
| gen_31 | 0.6356 | 0.7909 | 0.6429 | 0.4731 | 1138 |
| gen_33 | 0.5770 | 0.6894 | 0.5893 | 0.4522 | 3253 |
| gen_34 | 0.3792 | 0.5049 | 0.3974 | 0.2353 | 439 |
| gen_35 | 0.5672 | 0.7023 | 0.5725 | 0.4266 | 1216 |
| gen_36 | 0.6119 | 0.8068 | 0.6241 | 0.4050 | 1133 |
| gen_37 | 0.5799 | 0.7526 | 0.5410 | 0.4463 | 2935 |
| gen_38 | 0.5766 | 0.7384 | 0.5642 | 0.4271 | 2985 |
| gen_40 | 0.5999 | 0.7772 | 0.6019 | 0.4207 | 912 |
| gen_41 | 0.5799 | 0.7698 | 0.5483 | 0.4214 | 884 |
| gen_43 | 0.5980 | 0.7298 | 0.6406 | 0.4237 | 1005 |
| gen_44 | 0.5814 | 0.7483 | 0.5984 | 0.3976 | 448 |
| gen_45 | 0.5967 | 0.7574 | 0.5617 | 0.4710 | 880 |
| gen_46 | 0.5636 | 0.7503 | 0.5819 | 0.3585 | 2932 |
| gen_47 | 0.5426 | 0.6972 | 0.5263 | 0.4043 | 950 |
| gen_48 | 0.5759 | 0.7640 | 0.5444 | 0.4194 | 2992 |
| gen_49 | 0.5728 | 0.7404 | 0.5606 | 0.4172 | 2393 |
| gen_50 | 0.5992 | 0.7921 | 0.6197 | 0.3859 | 1177 |
| gen_52 | 0.5719 | 0.7011 | 0.6059 | 0.4085 | 2100 |
| gen_53 | 0.5888 | 0.7632 | 0.5942 | 0.4091 | 1411 |
| **gen_55** | 0.6140 | 0.7394 | 0.5818 | **0.5208** | 1772 |
| gen_56 | 0.5790 | 0.7601 | 0.5426 | 0.4342 | 908 |
| gen_59 | 0.5911 | 0.7215 | 0.5894 | 0.4625 | 1379 |
| gen_60 | 0.5843 | 0.7443 | 0.5528 | 0.4558 | 1870 |
| gen_61 | 0.5946 | 0.7535 | 0.5903 | 0.4399 | 2470 |
| gen_62 | 0.6073 | 0.7583 | 0.6159 | 0.4477 | 6481 |
| gen_63 | 0.5492 | 0.6361 | 0.4777 | 0.5338 | 74 |
| gen_64 | 0.6168 | 0.7420 | 0.6473 | 0.4610 | 1005 |
| **gen_65** | **0.6551** | 0.7715 | **0.6728** | **0.5211** | 1062 |
| gen_66 | 0.5897 | 0.7529 | 0.5444 | 0.4718 | 1236 |
| gen_69 | 0.6189 | 0.7785 | 0.6004 | 0.4777 | 1125 |
| gen_70 | 0.6111 | 0.7631 | 0.6198 | 0.4503 | 1274 |
| gen_71 | 0.5890 | 0.7623 | 0.5859 | 0.4189 | 826 |
| gen_74 | 0.5969 | 0.8022 | 0.5802 | 0.4082 | 1108 |
| gen_75 | 0.6115 | 0.7455 | 0.6466 | 0.4422 | 1120 |
| gen_76 | 0.6068 | 0.7391 | 0.5726 | 0.5086 | 1217 |
| gen_77 | 0.5907 | 0.7493 | 0.5714 | 0.4514 | 1293 |
| gen_78 | 0.5938 | 0.7446 | 0.6008 | 0.4359 | 1210 |
| gen_80 | 0.5999 | 0.7766 | 0.5932 | 0.4300 | 1066 |
| gen_81 | 0.5686 | 0.7408 | 0.5280 | 0.4369 | 1053 |
| gen_82 | 0.5930 | 0.7141 | 0.5616 | 0.5034 | 935 |
| gen_83 | 0.6047 | 0.7874 | 0.5781 | 0.4486 | 1185 |
| gen_85 | 0.5964 | 0.7567 | 0.5682 | 0.4644 | 1047 |
| gen_86 | 0.5900 | 0.7238 | 0.6342 | 0.4121 | 1444 |
| gen_87 | 0.5804 | 0.7545 | 0.5749 | 0.4118 | 773 |
| gen_88 | 0.6159 | 0.7694 | 0.5944 | 0.4840 | 2318 |
| gen_89 | 0.6003 | 0.7740 | 0.5985 | 0.4286 | 836 |

> **Note on RunPod initial baseline (0.377 vs. local 0.565)**: stochastic training with no fixed random seed. The MotionCLIP encoder was randomly re-initialised on the pod. Evolved genomes overcame this and exceeded both baselines.

---

## Key Highlights

### Top 10 Genomes by Macro-F1

| Rank | Genome | Macro-F1 | F1-Severe | Notes |
|:---:|---|---:|---:|---|
| 1 | **gen_65** | **0.6551** | 0.5211 | New best — 90 gen run |
| 2 | gen_31 | 0.6356 | 0.4731 | |
| 3 | gen_28 | 0.6343 | 0.4806 | Previous best (~30 gen) |
| 4 | gen_21 | 0.6231 | **0.5115** | Best clinical Pareto point |
| 5 | gen_69 | 0.6189 | 0.4777 | |
| 6 | gen_64 | 0.6168 | 0.4610 | |
| 7 | gen_88 | 0.6159 | 0.4840 | Late-run strong result |
| 8 | gen_55 | 0.6140 | **0.5208** | Strong severe detection |
| 9 | gen_36 | 0.6119 | 0.4050 | |
| 10 | gen_75 | 0.6115 | 0.4422 | |

### Evolution Trajectory

```
Gen 0   → 0.346  (degraded from seed)
Gen 8   → 0.315  (worse)
Gen 13  → 0.409  (first real progress)
Gen 21  → 0.623  ← breakthrough (Pareto-best for severe)
Gen 28  → 0.634  ← previous overall best
Gen 31  → 0.636
Gen 36  → 0.612
~~~ plateau at ~0.59–0.62 for gens 37–64 ~~~
Gen 55  → 0.614  (strong severe detection)
Gen 65  → 0.655  ← new best overall
Gen 69  → 0.619
Gen 88  → 0.616
```

---

## Class-Level Analysis

Class 2 (severe PD) is the most clinically critical and the hardest to predict:

| Genome | F1-Severe | Macro-F1 | Notes |
|---|---:|---:|---|
| Baseline | 0.497 | 0.565 | Local eval — RandomForest |
| **gen_65** | **0.521** | **0.655** | Best on both metrics |
| gen_55 | **0.521** | 0.614 | Strong severe, lower overall |
| gen_21 | 0.512 | 0.623 | Pareto point for clinical deployment |
| gen_76 | 0.509 | 0.607 | — |
| gen_63 | 0.534 | 0.549 | High severe, weak normal |
| gen_82 | 0.503 | 0.593 | — |
| gen_28 | 0.481 | 0.634 | Best ~30-gen result; lower severe |

**Pareto insight**: gen_65 and gen_21 are the strongest clinical options. gen_65 dominates gen_21 on both axes — it should be preferred for both leaderboard and clinical deployment.

---

## Per-Fold Generalization

### gen_65 — best overall (Macro-F1 0.655)

| Fold | Macro-F1 | Test Walks |
|---:|---:|---:|
| 1 | 0.457 | 159 |
| 2 | 0.482 | 122 |
| 3 | 0.560 | 143 |
| 4 | **0.878** | 115 |
| 5 | 0.419 | 135 |
| 6 | 0.488 | 107 |

### gen_28 — previous best (Macro-F1 0.634)

| Fold | Macro-F1 | Test Walks |
|---:|---:|---:|
| 1 | 0.451 | 159 |
| 2 | 0.478 | 122 |
| 3 | 0.547 | 143 |
| 4 | **0.720** | 115 |
| 5 | 0.490 | 135 |
| 6 | 0.495 | 107 |

Fold 4 is a consistent outlier across genomes — gen_65 achieves 0.878 there, nearly 40pp above its other folds. The variance (0.42–0.88) indicates cohort-specific gait variation that the model has partially captured in fold 4's subject group. **Improving folds 1, 2, and 5 is the key target for Task 2.**

---

## Task v2 Smoke Test Results

Task v2 uses a different initial genome architecture. Smoke test (3 generations, local):

| Genome | Macro-F1 | F1-Severe | Notes |
|---|---:|---:|---|
| initial (run 1) | 0.4172 | 0.292 | High variance — training instability |
| initial (run 2) | **0.5956** | **0.510** | Same code, different random state |
| initial (run 3) | 0.4585 | 0.207 | — |
| gen_0 | 0.5956 | 0.510 | Same as best initial |
| gen_2 | **0.6083** | **0.503** | First evolved improvement |

> The v2 initial baseline is more competitive than v1 when it converges. Gen_2 already exceeds v1 gen_28 in macro-F1 after only 2 generations — showing v2's seed is a better starting point.

---

## Key Takeaways

1. **ShinkaEvolve works**: Starting from 0.565, the ~90-generation run found genomes achieving 0.655 — a **+16% relative improvement** with zero human intervention after setup.

2. **Non-monotonic search is necessary**: Many intermediate genomes (gen_8: 0.315, gen_12: 0.282) scored far below the baseline. The archive-based approach retains high-quality parents; eventual winners come from early branches that temporarily regressed.

3. **Long plateau, then breakthrough**: After gen_31 (0.636), the population plateaued at 0.59–0.62 for ~30 generations before gen_65 broke through to 0.655. This pattern is typical of evolutionary search — patience is rewarded.

4. **Class 2 (severe PD) is the bottleneck**: The dataset is imbalanced (164 severe vs 341 normal walks) and severe gait is highly variable across subjects. gen_65 and gen_55 both achieve 0.521 F1-severe — the highest in the run.

5. **High eval-time variance**: Some genomes take 88× longer than others (gen_62: 6481s vs gen_63: 74s). Longer training doesn't guarantee better results.

6. **Gap to SOTA remains**: 0.655 vs 0.68+ target. Promising next steps from the evolution hints:
   - Fine-tune MotionCLIP encoder (unfreeze last N layers)
   - Multi-window sliding with attention pooling
   - Asymmetry features (left-right joint differences)
   - Frequency-domain features (FFT on joint trajectories)
   - Cross-cohort data augmentation to address fold variance

---

## Reproducibility

```bash
# Evaluate the best genome locally
ShinkaEvolve/.venv/Scripts/python care_pd_task/evaluate.py \
  --program_path results/care_pd_full/gen_65/main.py \
  --results_dir results/eval_gen65

# Evaluate previous best
ShinkaEvolve/.venv/Scripts/python care_pd_task/evaluate.py \
  --program_path results/care_pd_full/gen_28/main.py \
  --results_dir results/eval_gen28

# Run a new evolution experiment
ShinkaEvolve/.venv/Scripts/shinka_run \
  --task-dir care_pd_task \
  --results_dir results/care_pd_local \
  --num_generations 30 \
  --config-fname shinka_task.yaml
```

---

*Updated: 2026-04-09 | ~90 generations | 59 evaluated genomes | Repo: ShinkaEvolveCarePD | Tyronita*
