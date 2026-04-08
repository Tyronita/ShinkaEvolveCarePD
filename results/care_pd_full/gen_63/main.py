"""
CARE-PD: UPDRS Gait Score Prediction from SMPL Pose Sequences
Task: 3-class classification (UPDRS_GAIT in {0, 1, 2}) on BMCLab dataset.
Data: raw SMPL pose (frames, 72) per walk. 6-fold subject-level CV.
Fitness: macro-F1 (higher is better). Baseline: 0.565 (RandomForest).
Target: 0.68+ (CARE-PD LOSO SOTA), 0.74+ (MIDA SOTA).

Architecture overview:
  1. SMPL axis-angle (T, 72) → 6D rotation (T, 25, 6)   [inline, mutable]
  2. Crop/pad to 60 frames → (1, 60, 25, 6)              [inline, mutable]
  3. MotionCLIP Transformer encoder → mu (1, 512)         [inline, mutable]
  4. MLP classifier head → logits (1, 3)                  [inline, mutable]
  5. FocalLoss with class weighting                        [inline, mutable]

Everything in the EVOLVE-BLOCK is mutable. Promising mutations:
  - Fine-tune MotionCLIP encoder (unfreeze last N layers, lower LR=1e-5)
  - Change latent_dim / num_layers / num_heads / ff_size
  - Multi-window: slide 60-frame windows, aggregate with attention pooling
  - Add temporal self-attention on top of per-window embeddings
  - Try 9D rotation (full matrix), velocity features, or frequency-domain features
  - Modify FocalLoss alpha/gamma for class imbalance
  - Add asymmetry features (L-R joint rotation differences) as extra channels
  - Use the Skeleton FK utility to derive 3D joint positions from rotations
"""

import collections
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ===========================================================================
# EVOLVE-BLOCK-START
# ===========================================================================

# ── Rotation utilities ──────────────────────────────────────────────────────
# Inlined from CARE-PD data/preprocessing/preprocessing_utils.py
# (Zhou et al. 2019 — "On the Continuity of Rotation Representations")
# Mutable: try 9D (full matrix), quaternion, or euler representations.

def _aa_to_quat(aa: torch.Tensor) -> torch.Tensor:
    """Axis-angle (..., 3) → quaternion (..., 4) [w, x, y, z]."""
    angles = torch.norm(aa, p=2, dim=-1, keepdim=True).clamp(min=1e-8)
    half   = 0.5 * angles
    small  = angles.abs() < 1e-6
    s      = torch.where(small, 0.5 - angles * angles / 48, torch.sin(half) / angles)
    return torch.cat([torch.cos(half), aa * s], dim=-1)


def _quat_to_mat(q: torch.Tensor) -> torch.Tensor:
    """Quaternion (..., 4) [w, x, y, z] → rotation matrix (..., 3, 3)."""
    r, i, j, k = torch.unbind(q, -1)
    s = 2.0 / (q * q).sum(-1)
    o = torch.stack([
        1 - s*(j*j + k*k), s*(i*j - k*r),     s*(i*k + j*r),
        s*(i*j + k*r),     1 - s*(i*i + k*k), s*(j*k - i*r),
        s*(i*k - j*r),     s*(j*k + i*r),     1 - s*(i*i + j*j),
    ], dim=-1)
    return o.reshape(q.shape[:-1] + (3, 3))


def _mat_to_6d(mat: torch.Tensor) -> torch.Tensor:
    """Rotation matrix (..., 3, 3) → 6D representation (..., 6) [first 2 rows]."""
    return mat[..., :2, :].clone().reshape(*mat.shape[:-2], 6)


def smpl_to_6d(pose: np.ndarray) -> np.ndarray:
    """
    Convert SMPL axis-angle pose to 6D rotation + zero translation.

    Args:
        pose: (T, 72) float32  — 24 joints × 3 axis-angle params per frame
    Returns:
        (T, 25, 6) float32  — 24 joints × 6D rotation + 1 zero translation slot
    """
    T  = len(pose)
    aa = torch.from_numpy(pose.reshape(T, 24, 3)).float()
    r6 = _mat_to_6d(_quat_to_mat(_aa_to_quat(aa))).numpy()        # (T, 24, 6)
    return np.concatenate([r6, np.zeros((T, 1, 6), np.float32)], 1)  # (T, 25, 6)


def crop_pad_frames(seq: np.ndarray, n: int = 60) -> np.ndarray:
    """Center-crop or repeat-pad a (T, ...) sequence to exactly n frames."""
    T = len(seq)
    if T >= n:
        s = (T - n) // 2
        return seq[s:s + n]
    reps = (n + T - 1) // T
    return np.tile(seq, (reps,) + (1,) * (seq.ndim - 1))[:n]


# ── Physics-Informed Gait Feature Extraction ──────────────────────────────
# Targeted features for PD severity classification based on clinical markers

def extract_gait_features(pose: np.ndarray) -> np.ndarray:
    """
    Extract physics-informed gait features from SMPL pose sequence.
    Designed specifically for PD severity classification with focus on:
    - Trunk sway (postural instability)
    - Cadence instability (gait rhythm)
    - Ankle height oscillation (freezing of gait)
    
    Returns:
        1D feature vector capturing PD-relevant gait dynamics.
    """
    T = len(pose)
    if T == 0:
        return np.zeros(15)  # Return zero vector for empty sequences
    
    rot6d = smpl_to_6d(pose)  # (T, 25, 6)
    
    features = []
    
    # 1. Lateral trunk oscillation (spine joints 3,6,9) - x-axis rotation
    spine_joints = [3, 6, 9]
    trunk_oscillation = []
    for joint_idx in spine_joints:
        # Use x-axis rotation component (0) as proxy for lateral sway
        x_rot = rot6d[:, joint_idx, 0]  # (T,)
        trunk_oscillation.append(x_rot.std())
    features.append(np.array(trunk_oscillation))  # (3,)
    
    # 2. Cadence instability - split sequence into 4 segments, compute FFT peak frequency
    cadence_instability = []
    n_segments = 4
    segment_length = max(1, T // n_segments)
    
    for j_idx in [0, 7, 8]:  # pelvis, L_ankle, R_ankle
        for ch in [0, 1, 2]:  # first 3 components
            if T <= 1:
                cadence_instability.extend([0.0, 0.0])
                continue
                
            freqs = np.fft.rfftfreq(segment_length, d=1.0/30)
            gait_mask = (freqs >= 0.5) & (freqs <= 3.0)
            
            peak_freqs = []
            for seg in range(n_segments):
                start_idx = seg * segment_length
                end_idx = min(start_idx + segment_length, T)
                if end_idx - start_idx < 2:
                    continue
                    
                sig = rot6d[start_idx:end_idx, j_idx, ch] - rot6d[start_idx:end_idx, j_idx, ch].mean()
                fft_mag = np.abs(np.fft.rfft(sig))
                
                if gait_mask.any() and len(fft_mag) > 0:
                    try:
                        peak_freq = freqs[gait_mask][np.argmax(fft_mag[gait_mask])]
                        peak_freqs.append(peak_freq)
                    except:
                        peak_freqs.append(0.0)
                else:
                    peak_freqs.append(0.0)
            
            if len(peak_freqs) > 1:
                cadence_instability.append(np.std(peak_freqs))
                cadence_instability.append(np.mean(peak_freqs))
            else:
                cadence_instability.extend([0.0, 0.0])
    
    features.append(np.array(cadence_instability))  # (6,)
    
    # 3. Ankle height oscillation coefficient of variation
    ankle_cv = []
    for j_idx in [7, 8]:  # L_ankle, R_ankle
        if T <= 1:
            ankle_cv.append(0.0)
            continue
            
        # Use y-axis rotation (1) as proxy for height
        y_rot = rot6d[:, j_idx, 1] - rot6d[:, j_idx, 1].mean()
        std = np.std(y_rot)
        mean_abs = np.mean(np.abs(y_rot))
        if mean_abs > 1e-8:
            cv = std / mean_abs
        else:
            cv = 0.0
        ankle_cv.append(cv)
    
    features.append(np.array(ankle_cv))  # (2,)
    
    # 4. Sequence duration and length
    features.append(np.array([T / 30.0, np.log1p(T)]))  # (2,)
    
    # Combine all features
    result = np.concatenate([f.ravel() for f in features])
    return np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)

# ── Weighted Ensemble Classifier ──────────────────────────────────────────
# Physics-informed ensemble with proper class weighting

def train_ensemble_classifier(X_train: np.ndarray, y_train: np.ndarray,
                             X_test: np.ndarray) -> np.ndarray:
    """
    Train a voting classifier of Random Forest and Gradient Boosting.
    Both use StandardScaler and handle class imbalance via class_weight.
    This approach is more robust than neural networks for this dataset.
    """
    # Place sklearn imports inside function to prevent module not found errors
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    
    # Ensure all features are finite
    X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
    X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Compute class weights using training labels only
    n_cls = 3
    counts = np.bincount(y_train, minlength=n_cls).astype(float)
    # Handle zero counts by setting minimum count to 1.0
    counts = np.where(counts == 0, 1.0, counts)
    class_weight = {i: counts.sum() / (n_cls * counts[i]) for i in range(n_cls)}
    
    # Random Forest pipeline with proper class weighting
    rf = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=2,
            class_weight=class_weight,
            random_state=42, 
            n_jobs=-1
        ))
    ])
    
    # Gradient Boosting pipeline
    gb = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', GradientBoostingClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            random_state=42
        ))
    ])
    
    # Soft voting ensemble
    ensemble = VotingClassifier([('rf', rf), ('gb', gb)], voting='soft')
    ensemble.fit(X_train, y_train)
    return ensemble.predict(X_test)


def train_and_predict(train_poses: list, y_train: np.ndarray,
                       test_poses: list, care_pd_dir: str = None) -> np.ndarray:
    """
    Main prediction pipeline.
    
    Primary: physics-informed gait features → sklearn ensemble (RF + GB)
    This leverages domain-specific knowledge for PD gait analysis.
    """
    print('[genome] Extracting physics-informed gait features...', flush=True)
    
    # Extract gait features
    X_train_gait = np.stack([extract_gait_features(p) for p in train_poses])
    X_test_gait = np.stack([extract_gait_features(p) for p in test_poses])
    
    # Use only handcrafted features (no MotionCLIP)
    X_train = X_train_gait
    X_test = X_test_gait
    
    # Ensure features are finite
    X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
    X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)
    
    print(f'[genome] Training sklearn voting ensemble on {X_train.shape[1]}-dim features', flush=True)
    return train_ensemble_classifier(X_train, y_train, X_test)

# ===========================================================================
# EVOLVE-BLOCK-END
# ===========================================================================


def run_evaluation(data: dict, folds: dict, care_pd_dir: str = None) -> dict:
    """
    6-fold subject-level cross-validation harness. Fixed — not evolved.

    Prints per-class and per-fold F1 after each fold so the meta-LLM can
    read stdout and reason about which strategies improve class-2 (severe PD).
    """
    all_preds, all_labels = [], []
    per_fold_results = {}

    for fold_id, split in folds.items():
        def get_data(subjects):
            poses, labels = [], []
            for sub in subjects:
                if sub not in data:
                    continue
                for wd in data[sub].values():
                    poses.append(wd['pose'].astype(np.float32))
                    labels.append(int(wd['UPDRS_GAIT']))
            return poses, np.array(labels, dtype=np.int64)

        tr_poses, y_tr = get_data(split['train'])
        te_poses, y_te = get_data(split['eval'])

        if not te_poses or len(np.unique(y_tr)) < 2:
            continue

        preds = train_and_predict(tr_poses, y_tr, te_poses, care_pd_dir=care_pd_dir)
        all_preds.extend(preds.tolist())
        all_labels.extend(y_te.tolist())

        fold_f1 = float(f1_score(y_te, preds, average='macro', zero_division=0))
        pc_f1   = f1_score(y_te, preds, average=None, labels=[0,1,2], zero_division=0)
        per_fold_results[str(fold_id)] = {
            'macro_f1':      round(fold_f1, 4),
            'n_test_walks':  int(len(y_te)),
            'n_train_walks': int(len(y_tr)),
        }
        print(f'  fold {fold_id}: macro_f1={fold_f1:.4f}  '
              f'f1=[normal:{pc_f1[0]:.3f}, mild:{pc_f1[1]:.3f}, severe:{pc_f1[2]:.3f}]',
              flush=True)

    macro_f1     = float(f1_score(all_labels, all_preds, average='macro',  zero_division=0))
    per_class_f1 = f1_score(all_labels, all_preds, average=None, labels=[0,1,2], zero_division=0)

    print(f'\n[result] macro_f1={macro_f1:.4f}  '
          f'f1_class=[normal:{per_class_f1[0]:.4f}, mild:{per_class_f1[1]:.4f}, '
          f'severe:{per_class_f1[2]:.4f}]  device={DEVICE}', flush=True)

    return {
        'combined_score':   macro_f1,
        'all_preds':        all_preds,
        'all_labels':       all_labels,
        'per_fold_results': per_fold_results,
        'per_class_f1':     per_class_f1.tolist(),
    }