# CARE-PD – Benchmark Models and Reported Scores

## Overview

CARE-PD is a multi-site clinical gait dataset for Parkinson’s disease, released with a benchmark suite and baselines as part of the NeurIPS 2025 Datasets & Benchmarks Track submission and an accompanying GitHub repository. The benchmark focuses on two tasks: (1) supervised clinical severity estimation (UPDRS-gait scores) under several generalization protocols, and (2) motion pretext tasks (2D-to-3D lifting and 3D reconstruction) measuring reconstruction error and downstream clinical prediction quality.[1][2][3]

The public materials (paper, code, project page) currently report only the authors’ baseline results; there is no separate live leaderboard or external-challenge ranking yet.[4][1]

## Benchmarked Models

### Representation-learning backbones

For clinical score estimation, the paper evaluates seven frozen motion encoders, each paired with lightweight classifiers (linear and k-NN probes):[2]

- POTR (pose transformer for 3D pose lifting from 2D keypoints).[2]
- MixSTE (spatio-temporal transformer for 3D pose from 2D joints).[2]
- PoseFormerV2 (improved transformer-based 3D pose lifting).[2]
- MotionBERT (BERT-style encoder for 3D pose from 2D inputs).[2]
- MotionAGFormer (graph-transformer for 2D-to-3D motion lifting).[2]
- MotionCLIP (SMPL-based motion-text model using 6D rotations).[2]
- MoMask (VQ-VAE-based motion model trained on HumanML3D, used for reconstruction/generation and as an encoder).[2]

These encoders are kept frozen; only the probe on top is trained for UPDRS-gait classification, to test how much clinically relevant information is already present in generic motion representations.[2]

### Handcrafted-feature baseline

As a traditional baseline, the benchmark includes a Random Forest classifier trained on engineered gait features (spatiotemporal metrics, stability measures, and posture/arm-swing indicators) derived from reconstructed joints. This baseline is competitive in-domain but generally less robust across sites than learned motion encoders.[2]

### Pretext-task models

For motion pretext tasks, the benchmark uses two high-capacity models:[2]

- MotionAGFormer, evaluated on 2D-to-3D lifting (SMPL joints) and then probed for UPDRS-gait prediction.
- MoMask, evaluated on 3D reconstruction/generation and then probed for UPDRS-gait prediction.

## Clinical Severity Estimation Results

### Protocols and metrics

Clinical prediction is framed as estimating per-walk UPDRS-gait scores in \{0,1,2,3\} (0 = normal, 3 = severe impairment) from 3D mesh sequences. Performance is measured using macro-averaged F1 in two variants: \(F1_{0-3}\) (all classes) and \(F1_{0-2}\) (excluding class 3 from the metric while keeping its samples in training), to account for the rarity of class 3 in some datasets.[2]

Four protocols reflect different deployment regimes:[2]

- Within-dataset LOSO (Leave-One-Subject-Out) – train and test on the same cohort, holding out subjects.
- Cross-dataset – train on one cohort, test on the others (full domain shift).
- Leave-One-Dataset-Out (LODO) – train on the union of D−1 cohorts, test on the remaining cohort.
- Multi-dataset In-domain Adaptation (MIDA) – start from the LODO setting, then fine-tune the probe on a small in-domain training split for the target cohort.

### Summary of encoder performance

Across the four UPDRS-labeled datasets (PD-GaM, BMClab, T-SDU-PD, 3DGait), encoders achieve strong within-dataset performance but degrade significantly under domain shift.[2]

- Within-dataset LOSO: encoders reach up to macro-F1 ≈ 0.73 on PD-GaM and ≈ 0.68 on BMClab, with lower scores on T-SDU-PD and especially 3DGait (mean ≈ 0.27 with larger variance).[2]
- Cross-dataset: transferring to unseen cohorts typically reduces F1 by 0.2–0.4; when MoMask is trained on PD-GaM, its average cross-site F1 remains above 0.40, outperforming several weaker encoders’ within-site scores.[2]
- LODO: for the more diverse targets (BMClab, PD-GaM), MixSTE and MotionAGFormer reach macro-F1 around 0.50 on both label configurations, while performance on the small 3DGait set is ≤ 0.18 for all deep backbones.[2]
- MIDA: fine-tuning the probe with limited in-domain data yields the best results—on BMClab all backbones exceed 0.69, with MixSTE, MotionAGFormer, MotionBERT, and PoseFormerV2 reaching about 0.74–0.78 macro-F1. PD-GaM shows similar gains, with these models in the 0.63–0.70 range.[2]

Table-level averages in the appendix quantify these trends across encoders:[2]

- Mean LOSO macro-F1 (over 7 encoders) is about 55.9 ± 13.6% on BMClab, 62.0 ± 5.6% on PD-GaM, 41.7 ± 5.2% on T-SDU-PD, and 27.1 ± 8.2% on 3DGait.[2]
- Mean cross-dataset macro-F1 collapses to around 27–29% across all four cohorts.[2]
- Mean MIDA macro-F1 recovers to roughly 61.5 ± 12.2% on BMClab, 65.2 ± 4.4% on PD-GaM, 43.6 ± 4.3% on T-SDU-PD, and 37.2 ± 8.3% on 3DGait.[2]

Overall, encoders consistently outperform the Random Forest baseline in cross-site and multi-site setups, though the handcrafted model can still be competitive in some in-domain settings such as T-SDU-PD.[2]

### Viewpoint/2D-encoder results

For 2D models (MotionBERT, MixSTE, PoseFormerV2, MotionAGFormer), the paper analyzes performance when using posterior view only, lateral view only, or combining views.[2]

- Within/cross-dataset settings, combined views yield slightly higher average performance (≈ 34% macro-F1) compared to single-view inputs (≈ 31–34%).[2]
- Under LODO, the combined setup rises to about 36% vs. ≈ 31% for single-view variants.[2]
- Under MIDA, combined views achieve the largest gains, with average macro-F1 around 57% compared to ≈ 49% (posterior) and 45% (lateral) alone.[2]

These results indicate that multi-view information materially improves robustness to site and view changes, especially when combined with multi-site training and in-domain adaptation.[2]

## Motion Pretext Task Results

### Experimental setup

For each pretext model, four training regimes are compared while testing exclusively on CARE-PD:[2]

1. Zero-shot: model pre-trained on a generic motion dataset (H3.6M for MotionAGFormer, HumanML3D for MoMask) and evaluated directly on CARE-PD.
2. Fine-tune on CARE-PD.
3. Fine-tune on a large healthy-gait mixture from external datasets (without PD pathology) to control for “more walking” vs. “pathological gait”.
4. Train from scratch on CARE-PD.

Metrics include MPJPE, PA-MPJPE, and acceleration error for 3D pose reconstruction, as well as downstream macro-F1 for UPDRS-gait when the CARE-PD-finetuned encoder is paired with the same probe used in the main clinical benchmark.[2]

### MotionAGFormer results (2D-to-3D lifting)

Table 2 of the paper reports the following for MotionAGFormer on the pooled CARE-PD test sets:[2]

| Train Data | Finetune Data | MPJPE (mm) ↓ | PA-MPJPE (mm) ↓ | Acc (mm/s²) ↓ | UPDRS F1 ↑ |
|------------|---------------|--------------|------------------|---------------|------------|
| H3.6M | – | 60.7 | 21.4 | 99.8 | 48.1 |
| H3.6M | Healthy gait | 29.8 | 7.3 | 35.4 | 50.1 |
| H3.6M | CARE-PD | 7.5 | 2.6 | 11.6 | 65.1 |
| CARE-PD | – | 9.0 | 3.2 | 13.8 | 62.3 |

Finetuning on CARE-PD reduces MPJPE from roughly 60.7 mm (zero-shot) to 7.5 mm and improves macro-F1 on UPDRS-gait from 48.1 to 65.1, a gain of about 17 percentage points. Training from scratch on CARE-PD achieves comparable F1 (≈ 62.3) at slightly higher errors than the H3.6M-pretrained-and-finetuned variant.[2]

### MoMask results (3D reconstruction)

For MoMask, the analogous table is:[2]

| Train Data | Finetune Data | MPJPE (mm) ↓ | PA-MPJPE (mm) ↓ | Acc (mm/s²) ↓ | UPDRS F1 ↑ |
|------------|---------------|--------------|------------------|---------------|------------|
| HumanML3D | – | 22.5 | 17.8 | 4.3 | 41.4 |
| HumanML3D | Healthy gait | 22.3 | 13.7 | 4.4 | 40.6 |
| HumanML3D | CARE-PD | 8.7 | 6.3 | 2.2 | 62.7 |
| CARE-PD | – | 9.6 | 7.3 | 3.3 | 59.8 |

Here, CARE-PD finetuning lowers MPJPE from 22.5 to 8.7 mm and increases UPDRS F1 from 41.4 to 62.7, an improvement of about 21 percentage points. Again, models trained from scratch on CARE-PD attain similar downstream F1 with modestly higher reconstruction error.[2]

### Interpretation

Across both pretext models, finetuning on healthy gait alone yields only modest improvements over generic pretraining, whereas finetuning on CARE-PD produces large gains in both reconstruction error and downstream clinical prediction. This supports the conclusion that exposure to pathological gait kinematics, not just additional walking data, is necessary to learn clinically useful motion representations for PD.[2]

## Repository and Project Page Status

The GitHub repository currently focuses on dataset preparation, training scripts, and evaluation pipelines (within-dataset, cross-dataset, LODO, and MIDA), but does not host a separate machine-readable leaderboard or results table beyond what is described in the paper. The official project site notes an upcoming benchmark and challenge and, at the time of writing, does not yet list external challenge results or a dynamic ranking; it instead refers back to the NeurIPS 2025 paper and released code/data.[3][1][4]

Consequently, the "current" scores for CARE-PD are the baseline results reported in the NeurIPS 2025 paper, summarized above, with no additional public submissions or leaderboard entries documented yet.[4][2]