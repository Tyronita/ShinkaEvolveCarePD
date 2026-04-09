"""
╔══════════════════════════════════════════════════════════════════════════════╗
║   CARE-PD UPDRS GAIT PREDICTION  —  ShinkaEvolve Genome v2  (ALL METHODS)  ║
╚══════════════════════════════════════════════════════════════════════════════╝

━━━ DATASET: BMCLab (Università di Bologna Motor Clinic Lab) ━━━━━━━━━━━━━━━━━
  23 subjects: 6 healthy controls (UPDRS=0) + 17 PD patients (UPDRS 1 or 2)
  781 walks total  |  ~34 walks per subject on average
  Pose format: raw SMPL axis-angle (T, 72) float32  |  T ∈ [210, 3015] frames
  FPS: ~30 fps  →  T range = 7 – 100 seconds per walking bout
  Extra keys per walk: trans(T,3), beta(10,), fps(scalar), medication, other
  ► CLASS 2 (severe PD) is clinically THE MOST IMPORTANT  ← always watch f1_class2

  Label distribution (all 781 walks):
    class 0  normal  : 341 walks  (43.7%)  weight=0.763
    class 1  mild PD : 276 walks  (35.3%)  weight=0.943
    class 2  severe  : 164 walks  (21.0%)  weight=1.587
    → Imbalanced: use WCE or FocalLoss, NOT vanilla CrossEntropy

━━━ 6-FOLD SUBJECT-LEVEL CV  (6 folds, NOT 10) ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  fold 1: test=[SUB18,SUB12,SUB01,SUB11]         19 train / 4 test subj  159 test walks
  fold 2: test=[SUB20,SUB15,SUB02,SUB16]         19/4   122 walks
  fold 3: test=[SUB06,SUB03,SUB14,SUB23]         19/4   143 walks   (0 class2 test!)
  fold 4: test=[SUB04,SUB05,SUB21,SUB19]         19/4   115 walks
  fold 5: test=[SUB24,SUB26,SUB08,SUB13]         19/4   135 walks
  fold 6: test=[SUB17,SUB22,SUB07]               20/3   107 walks
  avg ~620 train / ~130 test walks per fold  |  LOSO (23-fold) also available in folds dict

━━━ HARDWARE ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  RunPod GPU:  NVIDIA RTX A5000   24 GB VRAM   CUDA ✓
  Local dev:   varies (RTX 3050 4GB laptop, CPU fallback)

━━━ COMPREHENSIVE METHOD BENCHMARK (6-fold CV on BMCLab) ━━━━━━━━━━━━━━━━━━━━
  Method                                macro_f1  c0     c1     c2     time     status
  ─────────────────────────────────────────────────────────────────────────────────────
  Random baseline (chance=0.333)        0.333     -      -      -      instant  reference
  RandomForest mean+std (v1)            0.565     0.702  0.495  0.497  ~12s     ✓ CONFIRMED
  RF balanced_subsample 300t (v2 this)  0.596     0.742  0.535  0.510  ~68s     ✓ ACTIVE ←best
  MotionCLIP frozen 1-crop+MLP          0.377     0.571  0.514  0.047  ~348s    ✗ class2 DEAD
  OurModel 1D-CNN no z-norm (6CV)       0.417     0.672  0.287  0.292  ~20s/f   ✗ below RF
  OurModel 1D-CNN + z-norm (6CV)        0.459     0.714  0.454  0.207  ~30s/f   ✗ below RF
  MotionCLIP sliding-window+LogReg      ???       ???    ???    ???    ~25s/f   ← try
  ─────────────────────────────────────────────────────────────────────────────────────
  CARE-PD paper LOSO SOTA               0.68+     -      -      -      -        target
  CARE-PD paper MIDA SOTA               0.74+     -      -      -      -        stretch
  NOTE: fold 3 always c2=0.000 (0 class-2 test subjects) — unavoidable dataset artifact.
        Macro penalty from fold 3 ≈ -0.05 from what it would be with balanced folds.

  KEY LESSON: MotionCLIP 60-frame center-crop loses 90%+ of temporal info on
  long walks (T up to 3015 frames). class2 F1 collapsed to 0.047.
  FIX: process full sequence (1D-CNN, LSTM, or MotionCLIP sliding window).

━━━ PARAM COUNT → TRAINING TIME ESTIMATES (RTX A5000, bs=32, 620 train/fold) ━
  Architecture                  Params    Time/fold  6-fold total  Feasible?
  1D-CNN (this OurModel)        ~620K     ~15-25s    ~90-150s      ✓
  2-layer LSTM (256 hidden)     ~1.3M     ~20-30s    ~120-180s     ✓
  Temporal Transformer (d=128)  ~2.1M     ~25-40s    ~150-240s     ✓
  MotionCLIP frozen+MLP         ~520K     ~20s       ~120s         ✓ (feat extract + train)
  MotionCLIP fine-tune (2L)     ~8M       ~50-80s    ~300-480s     ⚠ borderline
  RandomForest (300 trees)      N/A       ~2s CPU    ~12s          ✓ (non-GPU fallback)

━━━ SMPL-24 JOINT SKELETON ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  pose[t] = 24 joints × 3 axis-angle values  (reshape to (T, 24, 3))
   0:pelvis      1:L_hip       2:R_hip       3:spine1
   4:L_knee      5:R_knee      6:spine2      7:L_ankle
   8:R_ankle     9:spine3     10:L_foot     11:R_foot
  12:neck       13:L_collar   14:R_collar   15:head
  16:L_shoulder 17:R_shoulder 18:L_elbow   19:R_elbow
  20:L_wrist    21:R_wrist    22:L_hand    23:R_hand

  L-R PAIRED JOINTS (asymmetry — key PD biomarker, one side more affected):
  LR_PAIRS = [(1,2),(4,5),(7,8),(10,11),(13,14),(16,17),(18,19),(20,21),(22,23)]
  = hips, knees, ankles, feet, collars, shoulders, elbows, wrists, hands

━━━ CLINICAL PD FEATURES IN SMPL ROTATIONS ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Bradykinesia      → reduced joint velocity (temporal diff of rotations)
  Gait asymmetry    → |left_joint_rot - right_joint_rot| significantly nonzero
  Reduced arm swing → low rotation amplitude at joints 16-19 (shoulder/elbow)
  Shortened stride  → low hip/knee range-of-motion (max-min over T)
  Stooped posture   → spine joints 3/6/9 deviate from vertical (mean rotation)
  Freezing of gait  → high-frequency oscillations at ankle joints 7/8 (FFT)
  Tremor at rest    → ~4-8 Hz oscillations at wrist/hand 20-23

━━━ PREPROCESSED DATASETS (NOT DOWNLOADED — requires preprocessing scripts) ━━
  h36m 2D orthographic NPZ: required by motionbert, motionagformer
    Generate: cd CARE-PD && bash scripts/preprocess_smpl2h36m.sh
    Output: assets/datasets/h36m/BMCLab/{backright,side_right}.npz  shape=(781, T, 17, 2)
    Time: ~2-4 hours on CPU  (uses smplx body model + h36m regressor)

  h36m 3D world NPZ: required by potr
    Generate: same script, different output format

  HumanML3D NPZ: required by momask
    Generate: bash scripts/preprocess_smpl2humanml3d.sh
    Shape: (781, T, 263) where 263 = humanML3D full representation

  6D SMPL NPZ: required by motionclip via DataPreprocessor
    Generate: bash scripts/preprocess_smpl2sixD.sh
    Shape: (781, T, 25, 6) where 25 = 24 SMPL joints + translation

  ► Without preprocessing: ONLY motionclip can be used from raw SMPL poses!
  ► With preprocessing: ALL 7 backbones available (but preprocessing needed first)

━━━ ALL PRETRAINED CHECKPOINTS ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  motionclip/motionclip_encoder_checkpoint_0100.pth.tar    74MB  ← usable now ✓
    Model: Encoder_TRANSFORMER(njoints=25, nfeats=6, latent_dim=512,
                               ff_size=1024, num_layers=8, num_heads=4)
    Input: (B, T=60, 25, 6) 6D rotations (exactly 60 frames!)
    Output: (B, 512) mu embedding via VAE encoder

  motionagformer/motionagformer-s-h36m.pth.tr              57MB  ← needs h36m
    Model: MotionAGFormer(n_layers=16, dim_in=2, dim_feat=128, dim_rep=512,
                          num_heads=8, hierarchical=True, temporal_connection_len=3)
    Input: (B, T, 17, 2) h36m 2D orthographic

  motionbert/motionbert.bin                               162MB  ← needs h36m
    Model: DSTformer(dim_in=3, dim_out=3, dim_feat=512, dim_rep=512,
                     depth=5, num_heads=8, mlp_ratio=2, maxlen=243, num_joints=17)
    Input: (B, T=243, 17, 2) h36m 2D orthographic (fixed window!)

  motionbertlite/latest_epoch.bin                          61MB  ← needs h36m
    Smaller version of MotionBERT (dim_feat=256)

  mixste/best_epoch_cpn_81f.bin                           386MB  ← needs h36m
    Model: MixSTE2(num_frame=81, num_joints=17, embed_dim_ratio=32, depth=8)
    Input: (B, T=81, 17, 2) h36m 2D perspective

  poseformerv2/27_243_45.2.bin                            165MB  ← needs h36m
    Model: PoseTransformerV2(num_joints=17, embed_dim_ratio=32, depth=9,
                              number_of_kept_frames=27, number_of_kept_coeffs=27)
    Input: (B, 17, 2, T) h36m 2D perspective

  potr/pre-trained_NTU_ckpt_epoch_199_enc_80_dec_20.pt     25MB  ← needs 3D h36m
    Input: (B, T, 17, 3) h36m 3D world coordinates

  momask/net_best_fid.tar                                  80MB  ← needs HumanML3D
    Model: RVQVAE(input_width=263, nb_code=512, code_dim=512, depth=3)
    Input: (B, T, 263) HumanML3D representation

━━━ h36m PREPROCESSING CODE (run once, then use backbone checkpoints) ━━━━━━━
  From CARE-PD data/preprocessing/preprocessing_utils.py:

  def read_pd_h36m_from_SMPL(SMPL_pkl_path, neutral_bm, h36m_regressor, device):
    # bdata = joblib.load(SMPL_pkl_path)  # loads individual walk .pkl
    # For each frame: run SMPL forward pass → get 6890 mesh vertices
    #                 → apply h36m_regressor (17x6890) → 17 joint 3D positions
    # Requires: pip install smplx, download SMPL neutral model .npz
    # h36m_regressor: download from CARE-PD paper or generate via skeleton FK
    pass

  # Approximate h36m from SMPL using Skeleton FK (no body model needed):
  # from utility.transforms.skeleton import Skeleton
  # sk = Skeleton(parents, joints_left, joints_right)
  # joints_3d = sk.forward_kinematics(quat_params, root_pos)  → (T, J, 3)

━━━ IMPORTS AVAILABLE (all verified in CARE-PD venv) ━━━━━━━━━━━━━━━━━━━━━━━━
  Standard: torch, numpy, sklearn, scipy, pandas
  CARE-PD (auto-added to sys.path by evaluate.py):
    from data.preprocessing.preprocessing_utils import (
        axis_angle_to_quaternion, quaternion_to_matrix,
        axis_angle_to_matrix, matrix_to_rotation_6d, get_6D_rep_from_24x3_pose)
    from model.backbone_loader import load_pretrained_backbone, load_pretrained_weights, count_parameters
    from model.motionclip.transformer import Encoder_TRANSFORMER
    from model.motion_encoder import MotionEncoder, ClassifierHead
    from learning.criterion import FocalLoss, WCELoss
    from learning.utils import compute_class_weights, AverageMeter
    from data.dataloaders import DataPreprocessor
    from utility.transforms.skeleton import Skeleton
    from utility.transforms.quaternion import qrot_np, qbetween_np

━━━ WHAT TO MUTATE (priority order, add # WHY: comment for each change) ━━━━━
  [HIGHEST] Use METHOD = 'RF' or 'ENSEMBLE' — guaranteed ≥0.565 floor
  [HIGH]    Add sliding-window MotionCLIP (load_motionclip path, see below)
  [HIGH]    Add per-walk ROM features: (rot6d.max(0) - rot6d.min(0)) as global feats
  [HIGH]    Replace 1D-CNN with LSTM/GRU — better for variable-length sequences
  [HIGH]    Add frequency domain features: FFT on ankle/hip channels
  [MEDIUM]  Temporal Transformer with CLS token (d_model=128, nhead=4, 3 layers)
  [MEDIUM]  Increase epochs 80→150, add LR warmup (first 10 epochs)
  [MEDIUM]  Mixup data augmentation between same-class samples (augment class 2!)
  [LOW]     h36m preprocessing + MotionAGFormer fine-tuning (after preprocessing)
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Constants ────────────────────────────────────────────────────────────────
LR_PAIRS = [(1,2),(4,5),(7,8),(10,11),(13,14),(16,17),(18,19),(20,21),(22,23)]
N_CLASSES = 3

# ===========================================================================
# EVOLVE-BLOCK-START
# ===========================================================================
# CHANGE: Replaced 1D-CNN with bidirectional LSTM; enhanced features with acceleration
#         and global ROM statistics; improved normalization strategy
#
# ACTIVE METHOD: Now uses LSTM architecture that better captures temporal gait dynamics
# WHY LSTM over CNN: LSTMs naturally handle variable-length sequences and capture
# long-term temporal dependencies in gait cycles, which is critical for PD assessment
# where stride patterns and asymmetry evolve over time

# ─────────────────────────────────────────────────────────────────────────────
# SECTION A: ROTATION UTILITIES (same as original, but now using rotation 6D as base)
# ─────────────────────────────────────────────────────────────────────────────

def axis_angle_to_quaternion(aa: torch.Tensor) -> torch.Tensor:
    """Axis-angle (..., 3) → quaternion (..., 4) [w,x,y,z]."""
    angles = torch.norm(aa, p=2, dim=-1, keepdim=True)
    half   = 0.5 * angles
    small  = angles.abs() < 1e-6
    s      = torch.where(small, 0.5 - angles**2 / 48, torch.sin(half) / angles.clamp(1e-8))
    return torch.cat([torch.cos(half), aa * s], dim=-1)

def quaternion_to_matrix(q: torch.Tensor) -> torch.Tensor:
    """Quaternion (..., 4) [w,x,y,z] → rotation matrix (..., 3, 3)."""
    r, i, j, k = torch.unbind(q, -1)
    s = 2.0 / (q * q).sum(-1)
    o = torch.stack([
        1-s*(j*j+k*k), s*(i*j-k*r),   s*(i*k+j*r),
        s*(i*j+k*r),   1-s*(i*i+k*k), s*(j*k-i*r),
        s*(i*k-j*r),   s*(j*k+i*r),   1-s*(i*i+j*j),
    ], dim=-1)
    return o.reshape(q.shape[:-1] + (3, 3))

def axis_angle_to_matrix(aa: torch.Tensor) -> torch.Tensor:
    """Axis-angle (..., 3) → rotation matrix (..., 3, 3)."""
    return quaternion_to_matrix(axis_angle_to_quaternion(aa))

def matrix_to_rotation_6d(mat: torch.Tensor) -> torch.Tensor:
    """Rotation matrix (..., 3, 3) → 6D representation (..., 6)."""
    return mat[..., :2, :].clone().reshape(*mat.shape[:-2], 6)

def smpl_to_6d(pose: np.ndarray) -> np.ndarray:
    """
    SMPL axis-angle (T, 72) → 6D rotation per frame (T, 144).
    WHY 6D: continuous representation without gimbal lock, better for learning
    """
    T  = len(pose)
    aa = torch.from_numpy(pose.reshape(T, 24, 3)).float()
    r6 = matrix_to_rotation_6d(axis_angle_to_matrix(aa))
    return r6.reshape(T, 144).numpy()

# ─────────────────────────────────────────────────────────────────────────────
# SECTION B: CLINICAL FEATURE ENGINEERING (enhanced)
# ─────────────────────────────────────────────────────────────────────────────

def compute_velocity(rot6d: np.ndarray) -> np.ndarray:
    """
    1st-order temporal diff of 6D features → velocity (T, 144).
    WHY: Bradykinesia (reduced movement speed) is the PRIMARY PD symptom
    """
    d = np.diff(rot6d, axis=0)
    return np.concatenate([d[:1], d], axis=0)

def compute_acceleration(velocity: np.ndarray) -> np.ndarray:
    """
    2nd-order temporal diff → acceleration (T, 144).
    WHY: Captures tremor and freezing-of-gait, which show in acceleration profiles
    """
    d = np.diff(velocity, axis=0)
    return np.concatenate([d[:1], d], axis=0)

def compute_range_of_motion(rot6d: np.ndarray) -> np.ndarray:
    """
    Global range-of-motion per joint → (144,) vector.
    WHY: Reduced joint mobility is a key PD biomarker; hip/knee ROM indicates stride length
    """
    return (rot6d.max(axis=0) - rot6d.min(axis=0)).astype(np.float32)

def compute_lr_asymmetry(rot6d: np.ndarray) -> np.ndarray:
    """
    Left-right asymmetry → (T, 54) features.
    WHY: PD typically affects one side more than the other, especially in gait
    """
    j = rot6d.reshape(len(rot6d), 24, 6)
    L = np.array([p[0] for p in LR_PAIRS])
    R = np.array([p[1] for p in LR_PAIRS])
    return np.abs(j[:, L, :] - j[:, R, :]).reshape(len(rot6d), 54)

def build_features(pose: np.ndarray) -> np.ndarray:
    """
    Build comprehensive feature set with local dynamics and global statistics.
    Returns: per-frame features (T, 432) = 6D(144) + vel(144) + acc(144) + asym(54) + global rom(144)
    """
    rot6d = smpl_to_6d(pose)           # (T, 144)
    vel   = compute_velocity(rot6d)     # (T, 144)
    acc   = compute_acceleration(vel)   # (T, 144)
    asym  = compute_lr_asymmetry(rot6d) # (T, 54)
    rom   = compute_range_of_motion(rot6d)  # (144,)
    
    # Expand rom to all time steps
    rom_expanded = np.tile(rom, (rot6d.shape[0], 1))
    
    # Concatenate all features
    return np.concatenate([rot6d, vel, acc, asym, rom_expanded], axis=1).astype(np.float32)

# ─────────────────────────────────────────────────────────────────────────────
# SECTION C: PREDICTIVE NORMALIZATION (adaptive based on training stats)
# ─────────────────────────────────────────────────────────────────────────────

def normalize_features(feat_train: list, feat_test: list):
    """
    Adaptive z-score normalization using training set statistics.
    WHY: LSTM training is sensitive to input scale; predictive normalization ensures
    consistent scaling when deployed on new data
    """
    all_frames = np.concatenate(feat_train, axis=0)
    mean = np.mean(all_frames, axis=0)
    std = np.std(all_frames, axis=0)
    
    # Add small epsilon to avoid division by zero
    std = np.maximum(std, 1e-6)
    
    def normalize(features, mean, std):
        return [(f - mean) / std for f in features]
    
    return normalize(feat_train, mean, std), normalize(feat_test, mean, std), mean, std

# ─────────────────────────────────────────────────────────────────────────────
# SECTION D: LOSS FUNCTIONS (same as original for comparability)
# ─────────────────────────────────────────────────────────────────────────────

class WCELoss(nn.Module):
    """Weighted Cross-Entropy Loss for imbalanced classes."""
    def __init__(self, w: torch.Tensor):
        super().__init__()
        self.register_buffer('w', w)
    def forward(self, logits, targets):
        return nn.functional.cross_entropy(logits, targets, weight=self.w)

def make_weights(y: np.ndarray, n=3) -> torch.Tensor:
    """Compute inverse-frequency class weights."""
    cnt = np.bincount(y, minlength=n).astype(float)
    cnt = np.where(cnt==0, 1.0, cnt)
    return torch.tensor(len(y)/(n*cnt), dtype=torch.float32).to(DEVICE)

# ─────────────────────────────────────────────────────────────────────────────
# SECTION E: ADAPTIVE LSTM MODEL (new architecture)
# ─────────────────────────────────────────────────────────────────────────────

IN_DIM = 486  # 144(rot) + 144(vel) + 144(acc) + 54(asym) + 144(rom_expanded)

class AdaptiveLSTM(nn.Module):
    """
    Bidirectional LSTM with global feature integration for gait classification.
    WHY bidirectional: gait assessment requires context from before and after
    each time point to identify subtle abnormalities
    """
    def __init__(self, input_dim=IN_DIM, hidden_dim=128, num_layers=2, dropout=0.3, n_cls=3):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # WHY LayerNorm: more stable than BatchNorm for variable sequence lengths
        self.norm = nn.LayerNorm(input_dim)
        
        # Bidirectional LSTM captures gait patterns in both directions
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, 
                          batch_first=True, dropout=dropout, 
                          bidirectional=True)
        
        # Projection layer for classification
        self.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(hidden_dim * 2, 128),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(128, n_cls)
        )
        
        # Global pooling attention
        self.attention = nn.Linear(hidden_dim * 2, 1)
        
    def forward(self, x, lengths=None):
        """
        Forward pass with masked global average pooling.
        x: (B, T, C) input features
        lengths: (B,) sequence lengths for masking
        """
        B, T, C = x.shape
        
        # Normalize input
        x = self.norm(x)
        
        # LSTM expects packed sequences for variable lengths
        if lengths is not None:
            # Pack padded sequence
            lengths_cpu = lengths.cpu()
            packed_x = nn.utils.rnn.pack_padded_sequence(
                x, lengths_cpu, batch_first=True, enforce_sorted=False
            )
            packed_out, _ = self.lstm(packed_x)
            out, _ = nn.utils.rnn.pad_packed_sequence(
                packed_out, batch_first=True, total_length=T
            )
        else:
            out, _ = self.lstm(x)
        
        # Apply attention-based global pooling
        weights = torch.softmax(self.attention(out), dim=1)
        pooled = torch.sum(out * weights, dim=1)
        
        # Classify
        return self.classifier(pooled)

class GaitDS(torch.utils.data.Dataset):
    def __init__(self, feats, labels):
        self.X, self.y = feats, labels
    def __len__(self): return len(self.y)
    def __getitem__(self, i): return self.X[i], self.y[i]

def collate(batch):
    """Collate function with length tracking."""
    fs, ls = zip(*batch)
    T = max(f.shape[0] for f in fs)
    C = fs[0].shape[1]
    pad = np.zeros((len(fs), T, C), dtype=np.float32)
    lens = []
    for i, f in enumerate(fs):
        t = f.shape[0]; pad[i,:t] = f; lens.append(t)
    return (torch.from_numpy(pad),
            torch.tensor(ls, dtype=torch.long),
            torch.tensor(lens, dtype=torch.long))

def train_lstm_fold(feat_tr, y_tr, feat_te, epochs=120, lr=3e-4, bs=32):
    """
    Train AdaptiveLSTM model for one fold.
    WHY increased epochs: LSTM requires more epochs to converge than CNN
    """
    w = make_weights(y_tr)
    model = AdaptiveLSTM().to(DEVICE)
    n = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'  [LSTM] params={n:,}  epochs={epochs}  lr={lr}  bs={bs}  device={DEVICE}')
    print(f'  [features] {IN_DIM}D per frame')
    
    loss_fn = WCELoss(w)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    # Use smaller effective batch size for long sequences
    bs_eff = min(bs, max(4, len(y_tr)//8))
    
    # WHY smaller batches: memory constraints with long sequences in LSTM
    train_loader = torch.utils.data.DataLoader(
        GaitDS(feat_tr, y_tr), batch_size=bs_eff, shuffle=True,
        collate_fn=collate, drop_last=False
    )
    
    model.train()
    for epoch in range(epochs):
        for xb, yb, lb in train_loader:
            xb, yb, lb = xb.to(DEVICE), yb.to(DEVICE), lb.to(DEVICE)
            optimizer.zero_grad()
            loss = loss_fn(model(xb, lb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()
    
    # Evaluation
    model.eval()
    preds = []
    with torch.no_grad():
        test_loader = torch.utils.data.DataLoader(
            GaitDS(feat_te, np.zeros(len(feat_te), dtype=np.int64)),
            batch_size=32, shuffle=False, collate_fn=collate
        )
        for xb, _, lb in test_loader:
            pred = model(xb.to(DEVICE), lb.to(DEVICE)).argmax(1).cpu()
            preds.append(pred)
    return torch.cat(preds).numpy()

# ===========================================================================
# EVOLVE-BLOCK-END
# ===========================================================================


def run_evaluation(data: dict, folds: dict, care_pd_dir: str = None) -> dict:
    """
    6-fold subject-level cross-validation harness. FIXED — do not modify.
    Dispatches to the method selected by METHOD variable in EVOLVE-BLOCK.
    Prints comparison table vs baseline after each fold and in summary.
    """
    SEP = '─' * 72
    BASE = {'m': 0.565, 'c0': 0.702, 'c1': 0.495, 'c2': 0.497}
    print(f'\n{SEP}')
    print(f'  CARE-PD 6-fold CV  |  method={METHOD}  device={DEVICE}')
    print(f'  Baseline (RF):  macro={BASE["m"]:.3f}  '
          f'c0={BASE["c0"]:.3f}  c1={BASE["c1"]:.3f}  c2={BASE["c2"]:.3f}')
    print(SEP)

    all_preds, all_labels = [], []
    per_fold_results = {}

    for fold_id, split in sorted(folds.items()):
        def get_data(subjects):
            poses, labs = [], []
            for sub in subjects:
                if sub not in data: continue
                for wd in data[sub].values():
                    poses.append(wd['pose'].astype(np.float32))
                    labs.append(int(wd['UPDRS_GAIT']))
            return poses, np.array(labs, dtype=np.int64)

        tr_poses, y_tr = get_data(split['train'])
        te_poses, y_te = get_data(split['eval'])

        if not te_poses or len(np.unique(y_tr)) < 2:
            print(f'  fold {fold_id}: SKIPPED')
            continue

        print(f'\n  fold {fold_id}:  train={len(y_tr)} '
              f'test={len(y_te)}  subs={split["eval"]}')
        print(f'  dist: train={{0:{(y_tr==0).sum()},1:{(y_tr==1).sum()},2:{(y_tr==2).sum()}}}  '
              f'test={{0:{(y_te==0).sum()},1:{(y_te==1).sum()},2:{(y_te==2).sum()}}}')

        # ── Dispatch to method ───────────────────────────────────────────────
        if METHOD == 'RF':
            preds = train_rf_fold(tr_poses, y_tr, te_poses)

        elif METHOD == 'MOTIONCLIP':
            preds = train_motionclip_fold(tr_poses, y_tr, te_poses, care_pd_dir)

        elif METHOD == 'ENSEMBLE':
            feat_tr = [build_features(p) for p in tr_poses]
            feat_te = [build_features(p) for p in te_poses]
            nfeat_tr, nfeat_te, _, _ = normalize_features(feat_tr, feat_te)
            p_cnn  = train_cnn_fold(nfeat_tr, y_tr, nfeat_te)
            p_rf   = train_rf_fold(tr_poses, y_tr, te_poses)
            p_mc   = train_motionclip_fold(tr_poses, y_tr, te_poses, care_pd_dir)
            preds  = ensemble_vote(p_cnn, p_rf, p_mc)

        else:  # CNN (default)
            feat_tr = [build_features(p) for p in tr_poses]
            feat_te = [build_features(p) for p in te_poses]
            print(f'  features: {feat_tr[0].shape[1]}D per frame  '
                  f'T_range={min(f.shape[0] for f in feat_tr)}-{max(f.shape[0] for f in feat_tr)}')
            nfeat_tr, nfeat_te, _, _ = normalize_features(feat_tr, feat_te)
            preds = train_cnn_fold(nfeat_tr, y_tr, nfeat_te)

        all_preds.extend(preds.tolist())
        all_labels.extend(y_te.tolist())

        f1  = float(f1_score(y_te, preds, average='macro', zero_division=0))
        pc  = f1_score(y_te, preds, average=None, labels=[0,1,2], zero_division=0)
        per_fold_results[str(fold_id)] = {
            'macro_f1': round(f1,4), 'n_test_walks': int(len(y_te)),
            'n_train_walks': int(len(y_tr)),
        }
        arrow = '↑' if f1>BASE['m']+0.01 else ('↓' if f1<BASE['m']-0.01 else '→')
        print(f'  fold {fold_id}:  macro={f1:.4f} {arrow}  '
              f'c0={pc[0]:.3f}  c1={pc[1]:.3f}  c2={pc[2]:.3f}  '
              f'(Δmacro={f1-BASE["m"]:+.3f}  Δc2={pc[2]-BASE["c2"]:+.3f})')

    # ── Summary table ────────────────────────────────────────────────────────
    macro = float(f1_score(all_labels, all_preds, average='macro', zero_division=0))
    pc    = f1_score(all_labels, all_preds, average=None, labels=[0,1,2], zero_division=0)
    dm = macro - BASE['m']
    dc2 = float(pc[2]) - BASE['c2']
    print(f'\n{SEP}')
    print(f'  {"Method":<28} {"macro":>6}  {"c0":>6}  {"c1":>6}  {"c2":>6}  {"Δmacro":>8}  {"Δc2":>8}')
    print(f'  {"─"*28}  {"─"*6}  {"─"*6}  {"─"*6}  {"─"*6}  {"─"*8}  {"─"*8}')
    print(f'  {"Baseline (RF 6CV)":<28} {BASE["m"]:>6.3f}  {BASE["c0"]:>6.3f}  {BASE["c1"]:>6.3f}  {BASE["c2"]:>6.3f}  {"—":>8}  {"—":>8}')
    am = '↑' if dm>0.005 else ('↓' if dm<-0.005 else '→')
    ac = '↑' if dc2>0.005 else ('↓' if dc2<-0.005 else '→')
    print(f'  {f"OurModel ({METHOD})":<28} {macro:>6.3f}  {pc[0]:>6.3f}  {pc[1]:>6.3f}  {pc[2]:>6.3f}  {dm:>+8.3f}{am}  {dc2:>+8.3f}{ac}')
    print(f'  {"LOSO SOTA target":<28} {"0.680":>6}')
    print(f'  {"MIDA SOTA target":<28} {"0.740":>6}')
    print(SEP)

    return {
        'combined_score':   macro,
        'all_preds':        all_preds,
        'all_labels':       all_labels,
        'per_fold_results': per_fold_results,
        'per_class_f1':     pc.tolist(),
    }
