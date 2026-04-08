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


# ── Handcrafted gait feature extraction ─────────────────────────────────────

def extract_gait_features(pose: np.ndarray) -> np.ndarray:
    """
    Extract comprehensive handcrafted gait features from SMPL pose sequence.
    Returns a 1D feature vector capturing PD-relevant gait characteristics.

    Key features:
    - Global rotation statistics (mean, std, range)
    - Velocity & acceleration (motion dynamics)
    - Key joint velocities (pelvis, hips, knees, ankles)
    - Left-right asymmetry (hallmark of PD)
    - Stride frequency via FFT
    - Gait regularity via autocorrelation
    - Freeze-of-gait proxy (high-freq ankle power)
    - Walk duration
    """
    T = len(pose)
    rot6d = smpl_to_6d(pose)  # (T, 25, 6)
    flat = rot6d[:, :24, :].reshape(T, -1)  # (T, 144) — use only 24 joints

    features = []

    # 1. Global rotation statistics per joint
    features.append(flat.mean(0))           # (144,)
    features.append(flat.std(0))            # (144,)
    features.append(flat.max(0) - flat.min(0))  # range (144,)

    # 2. Velocity (1st derivative) statistics
    vel = np.diff(flat, axis=0)  # (T-1, 144)
    features.append(vel.mean(0))
    features.append(vel.std(0))
    features.append(np.abs(vel).mean(0))    # mean absolute velocity

    # 3. Acceleration (2nd derivative) — jerkiness proxy for PD tremor
    if T > 2:
        acc = np.diff(vel, axis=0)  # (T-2, 144)
        features.append(acc.std(0))
    else:
        features.append(np.zeros(144))

    # 4. Key joint velocities (pelvis, hips, knees, ankles, feet)
    key_joints = [0, 1, 2, 4, 5, 7, 8, 10, 11]
    joint_vel = np.diff(rot6d[:, :24, :], axis=0)  # (T-1, 24, 6)
    kj_vel = joint_vel[:, key_joints, :]  # (T-1, 9, 6)
    features.append(np.abs(kj_vel).mean(axis=0).flatten())   # (54,)
    features.append(kj_vel.std(axis=0).flatten())             # (54,)

    # 5. Left-right asymmetry (KEY for PD — affected side shows reduced motion)
    lr_pairs = [(1, 2), (4, 5), (7, 8), (10, 11), (16, 17), (18, 19), (20, 21)]
    for l_idx, r_idx in lr_pairs:
        diff = rot6d[:, l_idx, :] - rot6d[:, r_idx, :]  # (T, 6)
        features.append(diff.mean(0))          # (6,) mean asymmetry
        features.append(diff.std(0))           # (6,) variability of asymmetry
        features.append(np.abs(diff).mean(0))  # (6,) magnitude

    # 6. Joint range of motion (ROM) — reduced in PD
    for j in [0, 1, 2, 4, 5, 7, 8, 10, 11, 12, 16, 17]:
        joint_data = rot6d[:, j, :]
        rom = joint_data.max(0) - joint_data.min(0)  # (6,)
        features.append(rom)

    # 7. Stride frequency via FFT on key lower body joints
    for j in [0, 1, 2, 7, 8]:  # pelvis, hips, ankles
        for comp in range(3):  # first 3 components
            signal = rot6d[:, j, comp]
            signal = signal - signal.mean()
            if T > 10:
                fft_mag = np.abs(np.fft.rfft(signal))
                freqs = np.fft.rfftfreq(T, d=1.0/30)
                gait_mask = (freqs >= 0.5) & (freqs <= 3.0)
                hf_mask = freqs > 3.0
                total_power = fft_mag[1:].sum() + 1e-8
                if gait_mask.any():
                    gait_fft = fft_mag[gait_mask]
                    gait_freqs = freqs[gait_mask]
                    dom_freq = gait_freqs[np.argmax(gait_fft)]
                    gait_power_ratio = gait_fft.sum() / total_power
                    features.append(np.array([dom_freq, gait_power_ratio]))
                else:
                    features.append(np.zeros(2))
                hf_ratio = fft_mag[hf_mask].sum() / total_power if hf_mask.any() else 0.0
                features.append(np.array([hf_ratio]))
            else:
                features.append(np.zeros(3))

    # 8. Gait regularity via autocorrelation of pelvis
    pelvis_signal = rot6d[:, 0, 0]
    pelvis_signal = pelvis_signal - pelvis_signal.mean()
    if T > 30:
        ac = np.correlate(pelvis_signal, pelvis_signal, mode='full')
        ac = ac[T-1:] / (ac[T-1] + 1e-8)
        n_lags = min(60, len(ac))
        ac_feat = ac[:n_lags]
        if n_lags < 60:
            ac_feat = np.pad(ac_feat, (0, 60 - n_lags))
        features.append(ac_feat)
        # Summary: mean regularity over first 10 lags
        features.append(np.array([ac[1:11].mean() if len(ac) > 10 else 0.0]))
    else:
        features.append(np.zeros(61))

    # 9. Freeze-of-gait proxy: high-freq power ratio for ankles
    for j in [7, 8]:
        signal = rot6d[:, j, 0] - rot6d[:, j, 0].mean()
        if T > 10:
            fft_mag = np.abs(np.fft.rfft(signal))
            freqs = np.fft.rfftfreq(T, d=1.0/30)
            hf = fft_mag[freqs > 3.0].sum()
            lf = fft_mag[(freqs >= 0.5) & (freqs <= 3.0)].sum() + 1e-8
            features.append(np.array([hf / lf]))
        else:
            features.append(np.zeros(1))

    # 10. Walk duration and summary motion stats
    duration = T / 30.0
    mean_vel = np.abs(vel).mean()
    mean_vel_std = vel.std(0).mean()
    features.append(np.array([duration, np.log1p(T), mean_vel, mean_vel_std]))

    result = np.concatenate([f.ravel() for f in features]).astype(np.float32)
    return np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)


# ── MotionCLIP encoder ──────────────────────────────────────────────────────

class _PosEnc(nn.Module):
    def __init__(self, d: int, drop: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.drop = nn.Dropout(drop)
        pe = torch.zeros(max_len, d)
        pos = torch.arange(0, max_len).float().unsqueeze(1)
        div = torch.exp(torch.arange(0, d, 2).float() * (-np.log(10000.0) / d))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0).transpose(0, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(x + self.pe[:x.shape[0]])


class MotionCLIPEncoder(nn.Module):
    """
    MotionCLIP Transformer encoder (inlined from CARE-PD).
    Input:  (B, T, 25, 6) — 6D SMPL rotations, variable T frames
    Output: (B, latent_dim) — motion embedding (mu token)
    """
    def __init__(self, njoints: int = 25, nfeats: int = 6,
                 latent_dim: int = 512, ff_size: int = 1024,
                 num_layers: int = 8, num_heads: int = 4,
                 dropout: float = 0.1):
        super().__init__()
        self.input_feats  = njoints * nfeats
        self.latent_dim   = latent_dim
        self.muQuery      = nn.Parameter(torch.randn(1, latent_dim))
        self.sigmaQuery   = nn.Parameter(torch.randn(1, latent_dim))
        self.embed        = nn.Linear(self.input_feats, latent_dim)
        self.pos_enc      = _PosEnc(latent_dim, dropout)
        layer = nn.TransformerEncoderLayer(
            d_model=latent_dim, nhead=num_heads,
            dim_feedforward=ff_size, dropout=dropout, activation='gelu')
        self.encoder      = nn.TransformerEncoder(layer, num_layers=num_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, 25, 6) → mu: (B, latent_dim)"""
        B, T, J, F = x.shape
        x = x.permute(1, 0, 2, 3).reshape(T, B, J * F)
        x = self.embed(x)
        y    = torch.zeros(B, dtype=torch.long, device=x.device)
        xseq = torch.cat([self.muQuery[y].unsqueeze(0),
                          self.sigmaQuery[y].unsqueeze(0), x], dim=0)
        xseq = self.pos_enc(xseq)
        mask = torch.ones(B, T + 2, dtype=torch.bool, device=x.device)
        out  = self.encoder(xseq, src_key_padding_mask=~mask)
        return out[0]   # mu token — (B, D)


def _load_checkpoint(model: nn.Module, path: str) -> nn.Module:
    """Load pretrained weights, stripping DataParallel 'module.' prefix."""
    ckpt  = torch.load(path, map_location='cpu', weights_only=False)
    sd    = ckpt.get('state_dict', ckpt)
    msd   = model.state_dict()
    first = next(iter(msd))
    new   = collections.OrderedDict()
    for k, v in sd.items():
        if k.startswith('module.') and 'module.' not in first:
            k = k[7:]
        if k in msd:
            new[k] = v
    msd.update(new)
    model.load_state_dict(msd, strict=True)
    print(f'[genome] checkpoint: {len(new)}/{len(msd)} layers loaded from {os.path.basename(path)}')
    return model


def load_motionclip(care_pd_dir: str) -> nn.Module:
    """Load pretrained MotionCLIP encoder, frozen, on DEVICE. Returns None if unavailable."""
    if care_pd_dir is None:
        return None
    ckpt = os.path.join(care_pd_dir, 'assets', 'Pretrained_checkpoints',
                        'motionclip', 'motionclip_encoder_checkpoint_0100.pth.tar')
    if not os.path.isfile(ckpt):
        print(f'[genome] MotionCLIP checkpoint not found: {ckpt}')
        return None
    try:
        enc = MotionCLIPEncoder(
            njoints=25, nfeats=6, latent_dim=512,
            ff_size=1024, num_layers=8, num_heads=4).to(DEVICE)
        _load_checkpoint(enc, ckpt)
        enc.eval()
        for p in enc.parameters():
            p.requires_grad_(False)
        return enc
    except Exception as e:
        print(f'[genome] MotionCLIP load failed: {e}')
        return None


@torch.no_grad()
def extract_motionclip_features(encoder: nn.Module, poses: list) -> np.ndarray:
    """
    Extract MotionCLIP embeddings using multi-window mean pooling.

    Slide 60-frame windows with 30-frame stride → encode in batches → mean pool.
    Simple mean pooling is effective and avoids untrainable attention pooling.
    """
    window_size = 60
    hop = 30
    embs = []

    for pose in poses:
        rot6d = smpl_to_6d(pose)  # (T, 25, 6)
        T = rot6d.shape[0]

        # Pad if too short
        if T < window_size:
            reps = (window_size + T - 1) // T
            rot6d = np.tile(rot6d, (reps,) + (1,) * (rot6d.ndim - 1))[:window_size]
            T = window_size

        # Generate windows
        starts = list(range(0, T - window_size + 1, hop))
        if len(starts) == 0:
            starts = [0]

        windows = np.stack([rot6d[s:s + window_size] for s in starts])  # (N_w, 60, 25, 6)
        x = torch.from_numpy(windows).float().to(DEVICE)
        window_embs = encoder(x)  # (N_w, 512)

        # Simple mean pooling — more robust than untrainable attention
        pooled = window_embs.mean(0, keepdim=True).cpu().numpy()  # (1, 512)
        embs.append(pooled)

    return np.nan_to_num(np.vstack(embs), nan=0.0)  # (N, 512)


def train_and_predict(train_poses: list, y_train: np.ndarray,
                      test_poses: list, care_pd_dir: str = None) -> np.ndarray:
    """
    Main prediction pipeline.

    Primary: Handcrafted gait features → Calibrated RF + GBC ensemble
    Optional augmentation: MotionCLIP embeddings concatenated if available

    Uses probability calibration (isotonic) on RF to improve severe class recall.
    """
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.calibration import CalibratedClassifierCV

    # Compute class weights (inverse frequency)
    counts = np.bincount(y_train, minlength=3).astype(float)
    counts = np.where(counts == 0, 1.0, counts)
    class_weight = {i: float(counts.sum() / (3 * counts[i])) for i in range(3)}
    sample_weight = np.array([class_weight[y] for y in y_train])

    print(f'[genome] Class weights: {class_weight}', flush=True)
    print(f'[genome] Train class dist: {np.bincount(y_train, minlength=3)}', flush=True)

    # Extract handcrafted gait features (primary signal)
    print('[genome] Extracting handcrafted gait features...', flush=True)
    X_tr = np.stack([extract_gait_features(p) for p in train_poses])
    X_te = np.stack([extract_gait_features(p) for p in test_poses])
    print(f'[genome] Gait feature dim: {X_tr.shape[1]}', flush=True)

    # Optionally augment with MotionCLIP embeddings
    enc = load_motionclip(care_pd_dir)
    if enc is not None:
        print(f'[genome] Augmenting with MotionCLIP embeddings (device={DEVICE})', flush=True)
        try:
            mc_tr = extract_motionclip_features(enc, train_poses)  # (N, 512)
            mc_te = extract_motionclip_features(enc, test_poses)    # (N, 512)
            X_tr = np.concatenate([X_tr, mc_tr], axis=1)
            X_te = np.concatenate([X_te, mc_te], axis=1)
            print(f'[genome] Combined feature dim: {X_tr.shape[1]}', flush=True)
        except Exception as e:
            print(f'[genome] MotionCLIP feature extraction failed: {e}', flush=True)
    else:
        print('[genome] Using handcrafted features only (MotionCLIP unavailable)', flush=True)

    # Build weighted ensemble: Calibrated RF (class_weight) + GBC (sample_weight via fit_params)
    # Calibrate RF probabilities to improve severe class prediction (isotonic corrects minority class bias)
    rf_base = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', RandomForestClassifier(
            n_estimators=500,
            max_depth=None,
            min_samples_leaf=1,
            class_weight=class_weight,
            random_state=42,
            n_jobs=-1
        ))
    ])
    rf = CalibratedClassifierCV(rf_base, method='isotonic', cv=3)

    gb = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.8,
            random_state=42
        ))
    ])

    # Fit GBC with sample_weight to handle class imbalance
    print('[genome] Fitting GradientBoosting...', flush=True)
    # GBC in pipeline: pass sample_weight via fit_params
    gb.fit(X_tr, y_train, clf__sample_weight=sample_weight)

    # Fit Calibrated RF
    print('[genome] Fitting Calibrated RandomForest...', flush=True)
    rf.fit(X_tr, y_train)

    # Soft voting ensemble with calibrated probabilities
    print('[genome] Running soft voting ensemble with calibrated RF...', flush=True)
    rf_proba = rf.predict_proba(X_te)  # (N, 3) - now properly calibrated
    gb_proba = gb.predict_proba(X_te)  # (N, 3)

    # Weighted average: Calibrated RF and GBC equally weighted
    avg_proba = 0.5 * rf_proba + 0.5 * gb_proba
    preds = np.argmax(avg_proba, axis=1)

    return preds

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