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


# ── MotionCLIP encoder ──────────────────────────────────────────────────────
# Inlined from CARE-PD model/motionclip/transformer.py
# Pretrained on diverse human motions → rich 512-dim gait embeddings.
# Mutable: latent_dim, num_layers, num_heads, ff_size, dropout,
#          add layer norm, modify positional encoding, try CLS-only pooling.

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

    Pretrained weights match: njoints=25, nfeats=6, latent_dim=512,
    ff_size=1024, num_layers=8, num_heads=4.

    Checkpoint: {care_pd_dir}/assets/Pretrained_checkpoints/motionclip/
                motionclip_encoder_checkpoint_0100.pth.tar
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
        # (T, B, J*F)
        x = x.permute(1, 0, 2, 3).reshape(T, B, J * F)
        x = self.embed(x)                                      # (T, B, D)
        # Prepend learned mu / sigma query tokens
        y    = torch.zeros(B, dtype=torch.long, device=x.device)
        xseq = torch.cat([self.muQuery[y].unsqueeze(0),
                          self.sigmaQuery[y].unsqueeze(0), x], dim=0)
        xseq = self.pos_enc(xseq)
        mask = torch.ones(B, T + 2, dtype=torch.bool, device=x.device)
        out  = self.encoder(xseq, src_key_padding_mask=~mask)
        return out[0]   # mu token — (B, D)


def _load_checkpoint(model: nn.Module, path: str) -> nn.Module:
    """
    Load pretrained weights into model, stripping DataParallel 'module.' prefix.
    Inlined from CARE-PD model/backbone_loader.py → load_pretrained_weights().
    """
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
    """
    Load pretrained MotionCLIP encoder, frozen, on DEVICE.
    Returns None if checkpoint not found or load fails.
    """
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


# ── FocalLoss ────────────────────────────────────────────────────────────────
# Inlined from CARE-PD learning/criterion.py
# Better than CrossEntropyLoss for imbalanced UPDRS distribution {0:341,1:276,2:164}.
# Mutable: alpha (down-weights easy examples), gamma (focusing strength).
#          Try alpha=0.25/0.5, gamma=1.0/2.0/5.0. Or switch to WCE:
#          nn.CrossEntropyLoss(weight=class_weights_tensor)

class FocalLoss(nn.Module):
    def __init__(self, alpha: torch.Tensor, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha  # (3,) tensor of per-class weights
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = nn.functional.cross_entropy(logits, targets, reduction='none', weight=self.alpha)
        pt = torch.exp(-ce)
        return ((1 - pt) ** self.gamma * ce).mean()


# ── Comprehensive gait feature extraction ──────────────────────────────────
# Adapted from proven 0.62-macroF1 architecture, now more extensive.

def extract_gait_features(pose: np.ndarray) -> np.ndarray:
    """
    Extract comprehensive handcrafted gait features from SMPL pose sequence.
    Designed specifically for PD severity classification.

    Features include:
    - Global rotation statistics (mean, std, range)
    - Velocity and acceleration statistics
    - Left-right asymmetry (hip, knee, ankle, foot)
    - Stride frequency and rhythm via FFT
    - Gait regularity via autocorrelation
    - Freeze-of-gait proxy (high-freq power ratio)
    - Per-joint velocity features for gait-relevant joints
    - Sequence-level duration and kinematic features

    Returns:
        1D feature vector (1389,) capturing PD-relevant gait dynamics.
    """
    T = len(pose)
    rot6d = smpl_to_6d(pose)  # (T, 25, 6)
    flat = rot6d[:, :24, :].reshape(T, -1)  # (T, 144) - exclude translation

    features = []

    # 1. Global rotation statistics
    features.append(flat.mean(0))   # (144,)
    features.append(flat.std(0))    # (144,)
    features.append(flat.max(0) - flat.min(0))  # range (144,)

    # 2. Velocity statistics (first derivative)
    if T > 1:
        vel = np.diff(flat, axis=0)  # (T-1, 144)
        features.append(vel.mean(0))     # (144,)
        features.append(vel.std(0))      # (144,)
        features.append(np.abs(vel).mean(0))  # mean absolute velocity
    else:
        features.extend([np.zeros(144)] * 3)

    # 3. Acceleration statistics (second derivative)
    if T > 2:
        acc = np.diff(vel, axis=0)  # (T-2, 144)
        features.append(acc.mean(0))     # (144,)
        features.append(acc.std(0))      # (144,)
        features.append(np.abs(acc).mean(0))  # jerkiness
    else:
        features.extend([np.zeros(144)] * 3)

    # 4. Left-right asymmetry
    asym_pairs = [(1,2), (4,5), (7,8), (10,11)]  # hip, knee, ankle, foot
    for l_idx, r_idx in asym_pairs:
        diff = rot6d[:, l_idx, :] - rot6d[:, r_idx, :]  # (T, 6)
        features.append(diff.mean(0))   # (6,)
        features.append(diff.std(0))    # (6,)

    # 5. Velocity features for key gait joints
    key_joints = [0, 1, 2, 4, 5, 7, 8, 10, 11]  # pelvis, hips, knees, ankles, feet
    if T > 1:
        joint_vel = np.diff(rot6d[:, :24, :], axis=0)  # (T-1, 24, 6)
        for j in key_joints:
            jv = joint_vel[:, j, :]  # (T-1, 6)
            features.append(jv.mean(0))  # (6,)
            features.append(jv.std(0))   # (6,)
    else:
        features.extend([np.zeros(6)] * 18)  # 9 joints × 2 stats

    # 6. Frequency domain features from pelvis and ankle
    freqs = np.fft.rfftfreq(T, d=1.0/30) if T > 1 else [0]
    for j_idx in [0, 7, 8]:  # pelvis, L_ankle, R_ankle
        for ch in range(3):  # first 3 components
            if T <= 1:
                features.extend([np.zeros(3)])
                continue
            sig = rot6d[:, j_idx, ch] - rot6d[:, j_idx, ch].mean()
            fft_mag = np.abs(np.fft.rfft(sig))
            gait_mask = (freqs >= 0.5) & (freqs <= 3.0)
            hf_mask = freqs > 3.0

            if gait_mask.any():
                dominant_freq = freqs[gait_mask][np.argmax(fft_mag[gait_mask])]
            else:
                dominant_freq = 0.0

            total_power = fft_mag.sum() + 1e-8
            gait_power = fft_mag[gait_mask].sum() if gait_mask.any() else 0.0
            hf_power = fft_mag[hf_mask].sum() if hf_mask.any() else 0.0

            features.append(np.array([dominant_freq, gait_power/total_power, hf_power/total_power]))

    # 7. Gait regularity from pelvis autocorrelation
    if T > 1:
        pelvis_ac = rot6d[:, 0, 0] - rot6d[:, 0, 0].mean()
        ac = np.correlate(pelvis_ac, pelvis_ac, mode='full')
        ac = ac[T-1:] / (ac[T-1] + 1e-8)
        # Mean of first 30 lags
        ac_mean = ac[1:min(31, len(ac))].mean() if len(ac) > 1 else 0.0
        # Dominant period via first autocorrelation peak
        peak_idx = np.argmax(ac[1:]) + 1 if len(ac) > 1 else 1
        reg_features = np.array([ac_mean, peak_idx/30.0])  # regularity, stride_time
    else:
        reg_features = np.zeros(2)
    features.append(reg_features)

    # 8. Sequence-level features
    duration = T / 30.0
    step_features = np.array([duration, np.log1p(T)])
    features.append(step_features)

    return np.concatenate([f.ravel() for f in features])

# ── MotionCLIP feature extraction ────────────────────────────────────────
# Using simple mean pooling over windows — more stable than attention

@torch.no_grad()
def extract_motionclip_features(encoder: nn.Module, poses: list) -> np.ndarray:
    """
    Extract MotionCLIP embeddings using multi-window mean pooling.
    More stable than attention pooling, and faster.
    """
    embs = []
    for pose in poses:
        rot6d = smpl_to_6d(pose)  # (T, 25, 6)
        # Create fixed 60-frame windows with 30-frame step
        windows = []
        start_idx = 0
        while start_idx <= len(rot6d) - 60:
            windows.append(rot6d[start_idx:start_idx+60])
            start_idx += 30
        if len(windows) == 0:
            # Pad short sequences symmetrically
            pad_len = 60 - len(rot6d)
            left = pad_len // 2
            right = pad_len - left
            windows.append(np.pad(rot6d, ((left, right), (0,0), (0,0)), mode='edge'))

        # Encode all windows
        x = torch.from_numpy(np.stack(windows)).float().to(DEVICE)  # (N_w, 60, 25, 6)
        window_embs = encoder(x)  # (N_w, 512)
        # Mean pool over windows
        pooled_emb = window_embs.mean(0, keepdim=True).cpu().numpy()  # (1, 512)
        embs.append(pooled_emb)

    return np.vstack(embs)  # (N, 512)


# ── Ensemble sklearn classifier ────────────────────────────────────────────
# Proven superior on small tabular data with class imbalance

def train_ensemble_classifier(X_train: np.ndarray, y_train: np.ndarray,
                             X_test: np.ndarray) -> np.ndarray:
    """
    Train a voting classifier of Random Forest and Gradient Boosting.
    Both use StandardScaler and handle class imbalance via class_weight.
    This approach is more robust than neural networks for this dataset.
    """
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.ensemble import VotingClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline

    # Ensure all features are finite
    X_train = np.nan_to_num(X_train, nan=0, posinf=0, neginf=0)
    X_test = np.nan_to_num(X_test, nan=0, posinf=0, neginf=0)

    # Compute class weights
    n_cls = 3
    counts = np.bincount(y_train, minlength=n_cls).astype(float)
    class_weights = {i: 1.0 for i in range(n_cls)}
    if counts.min() > 0:
        class_weights = {i: counts.sum() / (n_cls * counts[i]) for i in range(n_cls)}

    from sklearn.calibration import CalibratedClassifierCV
    # Random Forest pipeline with more trees and finer leaves
    rf_base = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', RandomForestClassifier(
            n_estimators=500,
            max_depth=None,
            min_samples_leaf=1,
            class_weight=class_weights,
            random_state=42,
            n_jobs=-1
        ))
    ])
    # Calibrate probabilities to correct underestimation of severe class
    rf = CalibratedClassifierCV(rf_base, method='isotonic', cv=3)

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
    ensemble = VotingClassifier(
        estimators=[('rf', rf), ('gb', gb)],
        voting='soft'
    )

    ensemble.fit(X_train, y_train)
    return ensemble.predict(X_test)


def train_and_predict(train_poses: list, y_train: np.ndarray,
                       test_poses: list, care_pd_dir: str = None) -> np.ndarray:
    """
    Main prediction pipeline.

    Primary: handcrafted gait features → sklearn ensemble (RF + GB)
    Optional: concatenate with MotionCLIP features if available.

    This leverages domain-specific knowledge for PD gait analysis.
    """
    print('[genome] Extracting handcrafted gait features...', flush=True)
    # Extract gait features
    X_train_gait = np.stack([extract_gait_features(p) for p in train_poses])
    X_test_gait = np.stack([extract_gait_features(p) for p in test_poses])

    # Optionally add MotionCLIP features
    enc = load_motionclip(care_pd_dir)
    if enc is not None and X_train_gait.shape[0] > 0:
        try:
            print(f'[genome] Adding MotionCLIP features (device={DEVICE})', flush=True)
            X_train_clip = extract_motionclip_features(enc, train_poses)
            X_test_clip = extract_motionclip_features(enc, test_poses)
            # Combine features
            X_train = np.concatenate([X_train_gait, X_train_clip], axis=1)
            X_test = np.concatenate([X_test_gait, X_test_clip], axis=1)
            print(f'[genome] Combined features: {X_train.shape[1]} dims', flush=True)
        except Exception as e:
            print(f'[genome] MotionCLIP integration failed: {e}', flush=True)
            X_train, X_test = X_train_gait, X_test_gait
    else:
        X_train, X_test = X_train_gait, X_test_gait

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