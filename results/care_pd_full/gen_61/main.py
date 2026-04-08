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

def _aa_to_quat(aa: torch.Tensor) -> torch.Tensor:
    angles = torch.norm(aa, p=2, dim=-1, keepdim=True).clamp(min=1e-8)
    half = 0.5 * angles
    small = angles.abs() < 1e-6
    s = torch.where(small, 0.5 - angles * angles / 48, torch.sin(half) / angles)
    return torch.cat([torch.cos(half), aa * s], dim=-1)


def _quat_to_mat(q: torch.Tensor) -> torch.Tensor:
    r, i, j, k = torch.unbind(q, -1)
    s = 2.0 / (q * q).sum(-1)
    o = torch.stack([
        1 - s*(j*j + k*k), s*(i*j - k*r),     s*(i*k + j*r),
        s*(i*j + k*r),     1 - s*(i*i + k*k), s*(j*k - i*r),
        s*(i*k - j*r),     s*(j*k + i*r),     1 - s*(i*i + j*j),
    ], dim=-1)
    return o.reshape(q.shape[:-1] + (3, 3))


def _mat_to_6d(mat: torch.Tensor) -> torch.Tensor:
    return mat[..., :2, :].clone().reshape(*mat.shape[:-2], 6)


def smpl_to_6d(pose: np.ndarray) -> np.ndarray:
    T = len(pose)
    aa = torch.from_numpy(pose.reshape(T, 24, 3)).float()
    r6 = _mat_to_6d(_quat_to_mat(_aa_to_quat(aa))).numpy()
    return np.concatenate([r6, np.zeros((T, 1, 6), np.float32)], 1)


# ── Comprehensive gait feature extraction with temporal segmentation ───────

def _segment_stats(arr, n_segments=4):
    """Split array into n_segments along time axis, compute per-segment mean/std,
    then return cross-segment variability (std of segment means, std of segment stds)."""
    T = len(arr)
    if T < n_segments:
        # Not enough frames for segmentation
        D = arr.shape[1] if arr.ndim > 1 else 1
        return np.zeros(D * 2)
    
    seg_len = T // n_segments
    seg_means = []
    seg_stds = []
    for i in range(n_segments):
        start = i * seg_len
        end = start + seg_len if i < n_segments - 1 else T
        seg = arr[start:end]
        seg_means.append(seg.mean(0))
        seg_stds.append(seg.std(0))
    
    seg_means = np.array(seg_means)  # (n_segments, D)
    seg_stds = np.array(seg_stds)
    
    # Cross-segment variability
    var_of_means = np.std(seg_means, axis=0)  # (D,)
    var_of_stds = np.std(seg_stds, axis=0)    # (D,)
    
    return np.concatenate([var_of_means.ravel(), var_of_stds.ravel()])


def extract_gait_features(pose: np.ndarray) -> np.ndarray:
    """
    Extract comprehensive handcrafted gait features from SMPL pose sequence.
    ~2800+ dimensions covering 15+ feature categories.
    """
    T = len(pose)
    rot6d = smpl_to_6d(pose)  # (T, 25, 6)
    flat = rot6d[:, :24, :].reshape(T, -1)  # (T, 144)

    features = []

    # ── Category A: Global rotation statistics (432 dims) ──
    features.append(flat.mean(0))           # (144,)
    features.append(flat.std(0))            # (144,)
    features.append(flat.max(0) - flat.min(0))  # range (144,)

    # ── Category B: Velocity statistics (432 dims) ──
    if T > 1:
        vel = np.diff(flat, axis=0)  # (T-1, 144)
        features.append(vel.mean(0))
        features.append(vel.std(0))
        features.append(np.abs(vel).mean(0))
    else:
        vel = np.zeros((1, 144))
        features.extend([np.zeros(144)] * 3)

    # ── Category C: Acceleration statistics (432 dims) ──
    if T > 2:
        acc = np.diff(flat, axis=0, n=2)  # (T-2, 144)
        features.append(acc.mean(0))
        features.append(acc.std(0))
        features.append(np.abs(acc).mean(0))
    else:
        acc = np.zeros((1, 144))
        features.extend([np.zeros(144)] * 3)

    # ── Category I: Jerk statistics (432 dims) ──
    if T > 3:
        jerk = np.diff(flat, axis=0, n=3)  # (T-3, 144)
        features.append(jerk.mean(0))
        features.append(jerk.std(0))
        features.append(np.abs(jerk).mean(0))
    else:
        features.extend([np.zeros(144)] * 3)

    # ── Category D: Left-right asymmetry (60 dims) ──
    asym_pairs = [(1,2), (4,5), (7,8), (10,11), (16,17)]  # hip, knee, ankle, foot, shoulder
    for l_idx, r_idx in asym_pairs:
        diff = rot6d[:, l_idx, :] - rot6d[:, r_idx, :]  # (T, 6)
        features.append(diff.mean(0))   # (6,)
        features.append(diff.std(0))    # (6,)

    # ── Category D2: Velocity asymmetry (60 dims) ──
    if T > 1:
        joint_vel = np.diff(rot6d[:, :24, :], axis=0)  # (T-1, 24, 6)
        for l_idx, r_idx in asym_pairs:
            vdiff = joint_vel[:, l_idx, :] - joint_vel[:, r_idx, :]
            features.append(vdiff.mean(0))
            features.append(vdiff.std(0))
    else:
        features.extend([np.zeros(6)] * 10)

    # ── Category E: Key joint velocity stats (324 dims) ──
    key_joints = [0, 1, 2, 4, 5, 7, 8, 10, 11]
    if T > 1:
        joint_vel = np.diff(rot6d[:, :24, :], axis=0)
        for j in key_joints:
            jv = joint_vel[:, j, :]  # (T-1, 6)
            features.append(jv.mean(0))
            features.append(jv.std(0))
            # Percentile features
            for q in [10, 25, 75, 90]:
                features.append(np.percentile(jv, q, axis=0))
    else:
        features.extend([np.zeros(6)] * (9 * 6))

    # ── Category J: Coefficient of variation for key joints (48 dims) ──
    if T > 1:
        joint_vel = np.diff(rot6d[:, :24, :], axis=0)
        cv_joints = [1, 2, 4, 5, 7, 8, 10, 11]
        for j in cv_joints:
            jv = joint_vel[:, j, :]
            mean_abs = np.abs(jv).mean(0)
            std_val = jv.std(0)
            cv = std_val / (mean_abs + 1e-8)
            features.append(cv)
    else:
        features.extend([np.zeros(6)] * 8)

    # ── Category K: Trunk stability (36 dims) ──
    spine_joints = [3, 6, 9]
    if T > 1:
        spine_vel = np.diff(rot6d[:, spine_joints, :], axis=0).reshape(-1, 18)
        features.append(spine_vel.mean(0))
        features.append(spine_vel.std(0))
    else:
        features.extend([np.zeros(18)] * 2)

    # Trunk acceleration stability
    if T > 2:
        spine_acc = np.diff(rot6d[:, spine_joints, :], axis=0, n=2).reshape(-1, 18)
        features.append(spine_acc.std(0))
    else:
        features.append(np.zeros(18))

    # ── Category L: Arm swing magnitude and asymmetry (48 dims) ──
    arm_pairs = [(16, 17), (18, 19)]  # shoulders, elbows
    if T > 1:
        joint_vel = np.diff(rot6d[:, :24, :], axis=0)
        for l_j, r_j in arm_pairs:
            l_vel = joint_vel[:, l_j, :]  # (T-1, 6)
            r_vel = joint_vel[:, r_j, :]
            features.append(np.abs(l_vel).mean(0))  # L magnitude
            features.append(np.abs(r_vel).mean(0))  # R magnitude
            features.append(l_vel.std(0))
            features.append(r_vel.std(0))
            # Asymmetry
            arm_diff = l_vel - r_vel
            features.append(arm_diff.mean(0))
            features.append(arm_diff.std(0))
    else:
        features.extend([np.zeros(6)] * 12)

    # ── Category F: FFT frequency features (27 dims) ──
    freqs = np.fft.rfftfreq(T, d=1.0/30) if T > 1 else np.array([0.0])
    for j_idx in [0, 7, 8]:
        for ch in range(3):
            if T <= 1:
                features.append(np.zeros(3))
                continue
            sig = rot6d[:, j_idx, ch] - rot6d[:, j_idx, ch].mean()
            fft_mag = np.abs(np.fft.rfft(sig))
            gait_mask = (freqs >= 0.5) & (freqs <= 3.0)
            hf_mask = freqs > 3.0

            dominant_freq = freqs[gait_mask][np.argmax(fft_mag[gait_mask])] if gait_mask.any() else 0.0
            total_power = fft_mag.sum() + 1e-8
            gait_power = fft_mag[gait_mask].sum() if gait_mask.any() else 0.0
            hf_power = fft_mag[hf_mask].sum() if hf_mask.any() else 0.0

            features.append(np.array([dominant_freq, gait_power/total_power, hf_power/total_power]))

    # ── Extended FFT: spectral entropy and bandwidth (18 dims) ──
    for j_idx in [0, 7, 8]:
        for ch in range(3):
            if T <= 4:
                features.append(np.zeros(2))
                continue
            sig = rot6d[:, j_idx, ch] - rot6d[:, j_idx, ch].mean()
            fft_mag = np.abs(np.fft.rfft(sig))
            psd = fft_mag ** 2
            psd_norm = psd / (psd.sum() + 1e-12)
            # Spectral entropy
            spec_entropy = -np.sum(psd_norm * np.log(psd_norm + 1e-12))
            # Spectral centroid
            if len(freqs) == len(psd_norm):
                spec_centroid = np.sum(freqs * psd_norm)
            else:
                spec_centroid = 0.0
            features.append(np.array([spec_entropy, spec_centroid]))

    # ── Category G: Autocorrelation regularity (6 dims) ──
    if T > 30:
        reg_feats = []
        for ch in range(3):
            pelvis_sig = rot6d[:, 0, ch] - rot6d[:, 0, ch].mean()
            ac = np.correlate(pelvis_sig, pelvis_sig, mode='full')
            ac = ac[T-1:] / (ac[T-1] + 1e-8)
            ac_mean = ac[1:min(31, len(ac))].mean() if len(ac) > 1 else 0.0
            peak_idx = np.argmax(ac[1:min(60, len(ac))]) + 1 if len(ac) > 1 else 1
            reg_feats.extend([ac_mean, peak_idx / 30.0])
        features.append(np.array(reg_feats))
    else:
        features.append(np.zeros(6))

    # ── Temporal segmentation: rotation (288 dims) ──
    features.append(_segment_stats(flat, n_segments=4))

    # ── Temporal segmentation: velocity (288 dims) ──
    if T > 1:
        features.append(_segment_stats(vel, n_segments=4))
    else:
        features.append(np.zeros(288))

    # ── Temporal segmentation: acceleration (288 dims) ──
    if T > 2:
        features.append(_segment_stats(acc, n_segments=4))
    else:
        features.append(np.zeros(288))

    # ── Cross-joint correlation features (15 dims) ──
    # Correlation between key joint pairs' velocity magnitudes
    if T > 10:
        joint_vel = np.diff(rot6d[:, :24, :], axis=0)
        vel_mags = np.linalg.norm(joint_vel, axis=-1)  # (T-1, 24)
        corr_pairs = [(1,2), (4,5), (7,8), (10,11), (1,4), (2,5),
                       (4,7), (5,8), (7,10), (8,11), (16,17), (18,19),
                       (0,3), (3,6), (6,9)]
        corr_feats = []
        for j1, j2 in corr_pairs:
            if vel_mags[:, j1].std() > 1e-8 and vel_mags[:, j2].std() > 1e-8:
                c = np.corrcoef(vel_mags[:, j1], vel_mags[:, j2])[0, 1]
                corr_feats.append(c if np.isfinite(c) else 0.0)
            else:
                corr_feats.append(0.0)
        features.append(np.array(corr_feats))
    else:
        features.append(np.zeros(15))

    # ── Range of motion ratios (24 dims) ──
    # ROM per joint normalized by pelvis ROM
    rom_per_joint = []
    for j in range(24):
        rom = rot6d[:, j, :].max(0) - rot6d[:, j, :].min(0)
        rom_per_joint.append(np.linalg.norm(rom))
    rom_arr = np.array(rom_per_joint)
    pelvis_rom = rom_arr[0] + 1e-8
    features.append(rom_arr / pelvis_rom)

    # ── Velocity magnitude statistics per joint (72 dims) ──
    if T > 1:
        joint_vel = np.diff(rot6d[:, :24, :], axis=0)
        vel_mags = np.linalg.norm(joint_vel, axis=-1)  # (T-1, 24)
        features.append(vel_mags.mean(0))   # (24,)
        features.append(vel_mags.std(0))    # (24,)
        features.append(vel_mags.max(0))    # (24,)
    else:
        features.extend([np.zeros(24)] * 3)

    # ── Category H: Sequence-level features (4 dims) ──
    duration = T / 30.0
    features.append(np.array([duration, np.log1p(T), T, 1.0 / (duration + 0.1)]))

    return np.concatenate([f.ravel() for f in features])


# ── Ensemble sklearn classifier with XGBoost ──────────────────────────────

def train_ensemble_classifier(X_train: np.ndarray, y_train: np.ndarray,
                             X_test: np.ndarray) -> np.ndarray:
    """
    Train a voting classifier of RF, GB, and optionally XGBoost.
    Uses probability threshold tuning to boost severe class recall.
    """
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.ensemble import VotingClassifier, ExtraTreesClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline

    # Ensure all features are finite
    X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
    X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)

    # Compute class weights
    n_cls = 3
    counts = np.bincount(y_train, minlength=n_cls).astype(float)
    total = counts.sum()
    
    if counts.min() > 0:
        base_weights = {i: total / (n_cls * counts[i]) for i in range(n_cls)}
    else:
        base_weights = {i: 1.0 for i in range(n_cls)}

    # Moderate boost for severe class
    class_weights = {
        0: base_weights[0],
        1: base_weights[1] * 1.2,
        2: base_weights[2] * 1.8
    }

    # Random Forest
    rf = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', RandomForestClassifier(
            n_estimators=500,
            max_depth=None,
            min_samples_leaf=2,
            class_weight=class_weights,
            max_features='sqrt',
            random_state=42,
            n_jobs=-1
        ))
    ])

    # Gradient Boosting
    gb = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', GradientBoostingClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.03,
            subsample=0.8,
            min_samples_leaf=5,
            random_state=42
        ))
    ])

    # Extra Trees (different bias than RF)
    et = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', ExtraTreesClassifier(
            n_estimators=400,
            max_depth=None,
            min_samples_leaf=2,
            class_weight=class_weights,
            max_features='sqrt',
            random_state=42,
            n_jobs=-1
        ))
    ])

    estimators = [('rf', rf), ('gb', gb), ('et', et)]

    # Try to add XGBoost
    try:
        from xgboost import XGBClassifier
        
        # Compute sample weights for XGBoost
        sw = np.array([class_weights[y] for y in y_train])
        
        xgb = Pipeline([
            ('scaler', StandardScaler()),
            ('clf', XGBClassifier(
                n_estimators=300,
                max_depth=5,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.7,
                reg_alpha=0.1,
                reg_lambda=1.0,
                random_state=42,
                use_label_encoder=False,
                eval_metric='mlogloss',
                n_jobs=-1
            ))
        ])
        estimators.append(('xgb', xgb))
        print('[genome] XGBoost available, using 4-model ensemble', flush=True)
    except ImportError:
        print('[genome] XGBoost not available, using 3-model ensemble', flush=True)

    # Soft voting ensemble
    ensemble = VotingClassifier(
        estimators=estimators,
        voting='soft'
    )

    ensemble.fit(X_train, y_train)
    
    # Get probabilities for threshold tuning
    probs = ensemble.predict_proba(X_test)  # (N, 3)
    
    # Threshold tuning: lower threshold for class 2 to boost recall
    # Adjust by scaling class 2 probabilities up
    adjusted_probs = probs.copy()
    adjusted_probs[:, 2] *= 1.3  # Boost severe class probability
    adjusted_probs[:, 0] *= 0.95  # Slightly reduce normal class
    
    preds = np.argmax(adjusted_probs, axis=1)
    
    return preds


def train_and_predict(train_poses: list, y_train: np.ndarray,
                       test_poses: list, care_pd_dir: str = None) -> np.ndarray:
    """
    Main prediction pipeline: handcrafted gait features → sklearn ensemble.
    Skips MotionCLIP for speed (marginal benefit vs 1000s cost).
    """
    print('[genome] Extracting comprehensive gait features...', flush=True)
    
    # Extract gait features
    X_train_gait = np.stack([extract_gait_features(p) for p in train_poses])
    X_test_gait = np.stack([extract_gait_features(p) for p in test_poses])
    
    # Clean features
    X_train = np.nan_to_num(X_train_gait, nan=0.0, posinf=0.0, neginf=0.0)
    X_test = np.nan_to_num(X_test_gait, nan=0.0, posinf=0.0, neginf=0.0)
    
    print(f'[genome] Feature dimensions: {X_train.shape[1]}', flush=True)
    print(f'[genome] Training ensemble on {X_train.shape[0]} samples', flush=True)
    
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