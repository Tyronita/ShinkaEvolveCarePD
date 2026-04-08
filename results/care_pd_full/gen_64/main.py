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


# ── Comprehensive gait feature extraction ──────────────────────────────────
def extract_gait_features(pose: np.ndarray) -> np.ndarray:
    """
    Extract discriminative gait features from SMPL pose with focus on PD-relevant characteristics.
    Features based on biomechanical gait analysis: asymmetry, rhythm, regularity, range of motion.
    """
    T = len(pose)
    if T == 0:
        return np.zeros(100, dtype=np.float32)

    rot6d = smpl_to_6d(pose)  # (T, 25, 6)
    flat = rot6d.reshape(T, -1)  # (T, 150)

    features = []

    # 1. Global statistics
    features.append(flat.mean(0))
    features.append(flat.std(0))
    features.append(flat.max(0) - flat.min(0))

    # 2. Velocity and acceleration statistics
    if T > 1:
        vel = np.diff(flat, axis=0)
        features.append(vel.mean(0))
        features.append(vel.std(0))
        features.append(np.abs(vel).mean(0))
        
        if T > 2:
            acc = np.diff(vel, axis=0)
            features.append(acc.mean(0))
            features.append(acc.std(0))
            features.append(np.abs(acc).mean(0))
        else:
            features.extend([np.zeros(flat.shape[1])] * 3)
    else:
        features.extend([np.zeros(flat.shape[1])] * 6)

    # 3. Left-right asymmetry (hips, knees, ankles)
    asym_pairs = [(1,2), (4,5), (7,8), (10,11)]
    for l_idx, r_idx in asym_pairs:
        if T > 0:
            diff = rot6d[:, l_idx, :] - rot6d[:, r_idx, :]
            features.append(diff.mean(0))
            features.append(diff.std(0))
            features.append(np.abs(diff).mean(0))
        else:
            features.extend([np.zeros(6)] * 3)

    # 4. Key joint range of motion (pelvis, hips, knees, ankles)
    rom_joints = [0, 1, 2, 4, 5, 7, 8]
    for j in rom_joints:
        if T > 0:
            joint_data = rot6d[:, j, :]
            rom = joint_data.max(0) - joint_data.min(0)
            features.append(rom)
        else:
            features.append(np.zeros(6))

    # 5. Stride frequency via FFT (pelvis, hips, ankles)
    freqs = np.fft.rfftfreq(T, d=1.0/30) if T > 0 else [0]
    gait_mask = (freqs >= 0.5) & (freqs <= 3.0) if T > 0 else np.array([])
    
    for j in [0, 1, 2, 7, 8]:
        if T > 10:
            for ch in range(3):
                sig = rot6d[:, j, ch] - rot6d[:, j, ch].mean()
                fft_mag = np.abs(np.fft.rfft(sig))
                
                if gait_mask.any() and len(fft_mag) > 0:
                    gait_power = fft_mag[gait_mask].sum()
                    total_power = fft_mag.sum() + 1e-8
                    dom_freq_idx = np.argmax(fft_mag[gait_mask])
                    dom_freq = freqs[gait_mask][dom_freq_idx]
                    features.append(np.array([dom_freq, gait_power/total_power]))
                else:
                    features.append(np.zeros(2))
        else:
            features.append(np.zeros(2))

    # 6. Gait regularity via pelvis autocorrelation
    if T > 1:
        pelvis_sig = rot6d[:, 0, 0] - rot6d[:, 0, 0].mean()
        ac = np.correlate(pelvis_sig, pelvis_sig, mode='full')
        ac = ac[T-1:] / (ac[T-1] + 1e-8)
        ac_mean = ac[1:min(31, len(ac))].mean() if len(ac) > 1 else 0.0
        features.append(np.array([ac_mean]))
    else:
        features.append(np.zeros(1))

    # 7. Freeze-of-gait proxy: ankle high-frequency power ratio
    for j in [7, 8]:
        if T > 10:
            sig = rot6d[:, j, 0] - rot6d[:, j, 0].mean()
            fft_mag = np.abs(np.fft.rfft(sig))
            hf_power = fft_mag[freqs > 3.0].sum() if freqs[freqs > 3.0].any() else 0.0
            lf_mask = (freqs >= 0.5) & (freqs <= 3.0)
            lf_power = fft_mag[lf_mask].sum() + 1e-8
            features.append(np.array([hf_power / lf_power]))
        else:
            features.append(np.zeros(1))

    # 8. Sequence duration
    features.append(np.array([T / 30.0, np.log1p(T)]))

    # Concatenate and clean
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
    Slide 60-frame windows with 30-frame stride → encode → mean pool.
    """
    if not poses:
        return np.array([])
        
    embs = []
    for pose in poses:
        if len(pose) == 0:
            embs.append(np.zeros((1, 512), dtype=np.float32))
            continue

        rot6d = smpl_to_6d(pose)  # (T, 25, 6)
        T = rot6d.shape[0]
        
        # Pad if too short
        if T < 60:
            pad_len = 60 - T
            left = pad_len // 2
            right = pad_len - left
            rot6d = np.pad(rot6d, ((left, right), (0,0), (0,0)), mode='edge')
            T = 60

        # Create 60-frame windows with 30-frame stride
        starts = list(range(0, max(1, T - 60 + 1), 30))
        windows = np.stack([rot6d[s:s+60] for s in starts])  # (N_w, 60, 25, 6)
        
        # Process through encoder
        x = torch.from_numpy(windows).float().to(DEVICE)  # (N_w, 60, 25, 6)
        window_embs = encoder(x)  # (N_w, 512)
        
        # Mean pool across windows
        pooled = window_embs.mean(0, keepdim=True).cpu().numpy()  # (1, 512)
        embs.append(pooled)

    result = np.nan_to_num(np.vstack(embs), nan=0.0)
    return result


def train_and_predict(train_poses: list, y_train: np.ndarray,
                      test_poses: list, care_pd_dir: str = None) -> np.ndarray:
    """
    Main prediction pipeline with physics-informed features and proper ensemble weighting.
    Uses class-weighted RF+GB ensemble with optional MotionCLIP augmentation.
    """
    # Import inside function to prevent module issues
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
    from sklearn.svm import SVC
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    
    # Compute proper class weights using inverse frequency
    n_cls = 3
    counts = np.bincount(y_train, minlength=n_cls).astype(float)
    # Replace zeros with 1.0 to avoid division by zero
    counts = np.where(counts == 0, 1.0, counts)
    class_weight = {i: counts.sum() / (n_cls * counts[i]) for i in range(n_cls)}
    
    # Apply moderate boost to severe class (class 2) - 1.2x increase
    class_weight[2] *= 1.2
    
    print(f'[genome] Using class weights: {class_weight}', flush=True)
    print(f'[genome] Training distribution: {np.bincount(y_train)}', flush=True)

    # Extract handcrafted gait features
    print('[genome] Extracting gait features...', flush=True)
    X_tr_gait = np.stack([extract_gait_features(p) for p in train_poses])
    X_te_gait = np.stack([extract_gait_features(p) for p in test_poses])
    
    # Handle NaN/inf values
    X_tr_gait = np.nan_to_num(X_tr_gait, nan=0.0, posinf=0.0, neginf=0.0)
    X_te_gait = np.nan_to_num(X_te_gait, nan=0.0, posinf=0.0, neginf=0.0)

    # Get MotionCLIP features if available
    X_tr_clip, X_te_clip = None, None
    enc = load_motionclip(care_pd_dir)
    if enc is not None:
        print('[genome] Extracting MotionCLIP features...', flush=True)
        try:
            X_tr_clip = extract_motionclip_features(enc, train_poses)
            X_te_clip = extract_motionclip_features(enc, test_poses)
            # Verify dimensions match
            if X_tr_clip.shape[0] == X_tr_gait.shape[0] and X_te_clip.shape[0] == X_te_gait.shape[0]:
                X_tr = np.concatenate([X_tr_gait, X_tr_clip], axis=1)
                X_te = np.concatenate([X_te_gait, X_te_clip], axis=1)
                print(f'[genome] Combined feature dimension: {X_tr.shape[1]}', flush=True)
            else:
                print('[genome] Shape mismatch, using gait features only', flush=True)
                X_tr, X_te = X_tr_gait, X_te_gait
        except Exception as e:
            print(f'[genome] MotionCLIP extraction failed: {e}, using gait features only', flush=True)
            X_tr, X_te = X_tr_gait, X_te_gait
    else:
        print('[genome] MotionCLIP unavailable, using gait features only', flush=True)
        X_tr, X_te = X_tr_gait, X_te_gait

    # Ensure no NaN values
    X_tr = np.nan_to_num(X_tr, nan=0.0, posinf=0.0, neginf=0.0)
    X_te = np.nan_to_num(X_te, nan=0.0, posinf=0.0, neginf=0.0)

    # Create classifiers with proper scaling and class weighting
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
    
    print('[genome] Training ensemble classifier...', flush=True)
    ensemble.fit(X_tr, y_train)
    
    # Predict
    preds = ensemble.predict(X_te)
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