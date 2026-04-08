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


def get_sliding_windows(seq: np.ndarray, window_size: int = 60, step: int = 30) -> np.ndarray:
    """
    Generate sliding windows from sequence with specified step size.

    Args:
        seq: (T, ...) input sequence
        window_size: size of each window
        step: step size between windows (50% overlap if step=30)

    Returns:
        (N, window_size, ...) array of windows
    """
    T = len(seq)
    if T <= window_size:
        # If sequence is too short, repeat-pad it first
        reps = (window_size + T - 1) // T
        seq = np.tile(seq, (reps,) + (1,) * (seq.ndim - 1))
        T = len(seq)

    # Generate starting indices for windows
    starts = np.arange(0, T - window_size + 1, step)
    if len(starts) == 0:
        starts = np.array([0])  # Ensure at least one window

    windows = np.array([seq[s:s + window_size] for s in starts])
    return windows


def extract_gait_features(pose: np.ndarray) -> np.ndarray:
    """
    Extract comprehensive handcrafted gait features from SMPL pose sequence.
    Returns a 1D feature vector capturing PD-relevant gait characteristics.
    """
    T = len(pose)
    rot6d = smpl_to_6d(pose)  # (T, 25, 6)
    flat = rot6d.reshape(T, -1)  # (T, 150)

    features = []

    # 1. Global rotation statistics per joint (mean, std, percentiles)
    features.append(flat.mean(0))       # (150,)
    features.append(flat.std(0))        # (150,)
    features.append(np.percentile(flat, 25, axis=0))  # (150,)
    features.append(np.percentile(flat, 75, axis=0))  # (150,)

    # 2. Velocity (1st derivative) statistics
    vel = np.diff(flat, axis=0)  # (T-1, 150)
    features.append(vel.mean(0))
    features.append(vel.std(0))
    features.append(np.abs(vel).mean(0))  # mean absolute velocity

    # 3. Acceleration (2nd derivative) statistics
    if T > 2:
        acc = np.diff(vel, axis=0)  # (T-2, 150)
        features.append(acc.mean(0))
        features.append(acc.std(0))
        features.append(np.abs(acc).mean(0))  # jerkiness proxy
    else:
        features.extend([np.zeros(150)] * 3)

    # 4. Left-right asymmetry (KEY for PD — affected side shows reduced motion)
    lr_pairs = [(1, 2), (4, 5), (7, 8), (10, 11), (16, 17), (18, 19), (20, 21)]
    for l_idx, r_idx in lr_pairs:
        diff = rot6d[:, l_idx, :] - rot6d[:, r_idx, :]  # (T, 6)
        features.append(diff.mean(0))   # (6,) mean asymmetry
        features.append(diff.std(0))    # (6,) variability of asymmetry
        features.append(np.abs(diff).mean(0))  # (6,) magnitude

    # 5. Joint range of motion (ROM) — reduced in PD
    for j in [0, 1, 2, 4, 5, 7, 8, 12, 16, 17, 18, 19]:
        joint_data = rot6d[:, j, :]
        rom = joint_data.max(0) - joint_data.min(0)  # (6,)
        features.append(rom)

    # 6. Stride frequency via FFT on lower body joints
    for j in [0, 1, 2, 7, 8]:  # pelvis, hips, ankles
        for comp in range(min(3, 6)):  # first 3 components
            signal = rot6d[:, j, comp]
            signal = signal - signal.mean()
            if T > 10:
                fft_mag = np.abs(np.fft.rfft(signal))
                freqs = np.fft.rfftfreq(T, d=1.0/30)
                gait_mask = (freqs >= 0.5) & (freqs <= 3.0)
                if gait_mask.any():
                    gait_fft = fft_mag[gait_mask]
                    gait_freqs = freqs[gait_mask]
                    dom_freq = gait_freqs[np.argmax(gait_fft)]
                    dom_power = gait_fft.max()
                    total_power = fft_mag[1:].sum() + 1e-8
                    gait_power_ratio = gait_fft.sum() / total_power
                    features.append(np.array([dom_freq, dom_power, gait_power_ratio]))
                else:
                    features.append(np.zeros(3))
            else:
                features.append(np.zeros(3))

    # 7. Gait regularity via autocorrelation of pelvis
    pelvis_signal = rot6d[:, 0, 0]
    pelvis_signal = pelvis_signal - pelvis_signal.mean()
    if T > 30:
        ac = np.correlate(pelvis_signal, pelvis_signal, mode='full')
        ac = ac[T-1:] / (ac[T-1] + 1e-8)
        n_lags = min(60, len(ac))
        features.append(ac[:n_lags])
        if n_lags < 60:
            features.append(np.zeros(60 - n_lags))
    else:
        features.append(np.zeros(60))

    # 8. Freeze-of-gait proxy: high-freq power ratio for ankles
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

    # 9. Bilateral coordination breakdown features (KEY for severe PD)
    # Compute normalized cross-correlation at lags [-5, 0, 5] for key joint pairs
    lag_vals = [-5, 0, 5]
    joint_pairs = [(1,2), (4,5), (7,8)]  # hips, knees, ankles
    pelvis_ankle_pairs = [(0,7), (0,8)]  # pelvis-L_ankle, pelvis-R_ankle
    for pair in joint_pairs + pelvis_ankle_pairs:
        for comp in range(6):  # all 6 rotation components
            s1 = rot6d[:, pair[0], comp]
            s2 = rot6d[:, pair[1], comp]
            s1 = (s1 - s1.mean()) / (s1.std() + 1e-8)
            s2 = (s2 - s2.mean()) / (s2.std() + 1e-8)
            for lag in lag_vals:
                if lag == 0:
                    cc = np.corrcoef(s1, s2)[0,1]
                else:
                    if lag > 0:
                        cc = np.corrcoef(s1[lag:], s2[:-lag])[0,1] if lag < T else 0.0
                    else:
                        cc = np.corrcoef(s1[:lag], s2[-lag:])[0,1] if -lag < T else 0.0
                features.append(np.array([cc]))

    # 10. Proximal-distal coupling: pelvis vs ankle rotation variance ratio (6-dim)
    for comp in range(6):
        pelvis_var = np.var(rot6d[:, 0, comp])
        ankle_var = (np.var(rot6d[:, 7, comp]) + np.var(rot6d[:, 8, comp])) / 2.0
        ratio = pelvis_var / (ankle_var + 1e-8)
        features.append(np.array([ratio]))

    # 11. Sequence length (normalized) — PD patients may have shorter walks
    features.append(np.array([T / 30.0, np.log1p(T)]))

    return np.concatenate([f.ravel() for f in features])


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

# Remove FocalLoss - now using sklearn with class_weight='balanced'


# ── Feature extraction ───────────────────────────────────────────────────────

@torch.no_grad()
def extract_motionclip_features(encoder: nn.Module, poses: list) -> np.ndarray:
    """
    Extract MotionCLIP embeddings using multi-window mean pooling.

    Strategy: slide 60-frame windows across each walk → extract per-window features
              → mean-pool across windows → 512-dim walk-level embedding.

    Uses simple mean pooling instead of untrainable attention pooling.

    Args:
        encoder: frozen MotionCLIPEncoder on DEVICE
        poses: list of (T_i, 72) float32 arrays
    Returns:
        (N, 512) numpy feature matrix
    """
    embs = []
    for pose in poses:
        rot6d = smpl_to_6d(pose)                               # (T, 25, 6)
        windows = get_sliding_windows(rot6d, window_size=60, step=30)  # (N_w, 60, 25, 6)

        # Extract features for all windows
        window_embs = []
        for window in windows:
            x = torch.from_numpy(window).float().unsqueeze(0).to(DEVICE)  # (1, 60, 25, 6)
            window_emb = encoder(x)  # (1, 512)
            window_embs.append(window_emb)

        # Stack and mean pool
        window_embs = torch.cat(window_embs, dim=0)  # (N_w, 512)
        pooled_emb = window_embs.mean(0, keepdim=True).cpu().numpy()  # (1, 512)
        embs.append(pooled_emb)

    return np.vstack(embs)  # (N, 512)


# ── Classifier head ──────────────────────────────────────────────────────────

# Remove ClassifierMLP and train_classifier - now using sklearn ensemble


def train_and_predict(train_poses: list, y_train: np.ndarray,
                      test_poses: list, care_pd_dir: str = None) -> np.ndarray:
    """
    Main prediction pipeline with hybrid features and sklearn ensemble.

    Path A (primary): MotionCLIP + handcrafted features → sklearn ensemble
    Path B (fallback): Handcrafted features only → sklearn ensemble

    The primary approach combines MotionCLIP embeddings with comprehensive
    handcrafted gait features, then uses a voting ensemble of GradientBoosting
    and SVC with class_weight='balanced'. This leverages both deep motion
    representations and clinical gait biomarkers.
    """
    enc = load_motionclip(care_pd_dir)

    # Extract handcrafted features for all poses
    X_tr_gait = np.stack([extract_gait_features(p) for p in train_poses])
    X_te_gait = np.stack([extract_gait_features(p) for p in test_poses])

    if enc is not None:
        print(f'[genome] Path A: Hybrid features (MotionCLIP + gait) → sklearn ensemble (device={DEVICE})', flush=True)
        # Extract MotionCLIP features
        X_tr_clip = extract_motionclip_features(enc, train_poses)
        X_te_clip = extract_motionclip_features(enc, test_poses)
        # Combine features
        X_tr = np.concatenate([X_tr_gait, X_tr_clip], axis=1)
        X_te = np.concatenate([X_te_gait, X_te_clip], axis=1)
    else:
        print(f'[genome] Path B: Handcrafted gait features only → sklearn ensemble', flush=True)
        X_tr, X_te = X_tr_gait, X_te_gait

    # Use sklearn ensemble with proper scaling and class balancing
    from sklearn.ensemble import GradientBoostingClassifier, VotingClassifier
    from sklearn.svm import SVC
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline

    # Create ensemble with class_weight='balanced'
    clf = Pipeline([
        ('scaler', StandardScaler()),
        ('ensemble', VotingClassifier([
            ('gbc', GradientBoostingClassifier(n_estimators=100, random_state=42)),
            ('svc', SVC(kernel='rbf', C=10, gamma='scale', class_weight='balanced', probability=True))
        ], voting='soft'))
    ])

    # Fit and predict
    clf.fit(X_tr, y_train)
    return clf.predict(X_te)

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