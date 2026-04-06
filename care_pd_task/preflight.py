"""
CARE-PD Preflight Benchmark
============================
Runs BEFORE ShinkaEvolve starts. Tests every available method on fold 0
so you have a complete score map before evolution begins.

Methods tested:
  0. RandomForest baseline (raw axis-angle mean+std, CPU)
  1. GradientBoosting on 6D rotation mean+std (CPU)
  2. MotionCLIP frozen features → MLP (GPU if available)
  3. MotionCLIP frozen features → GradientBoosting (GPU encoder, CPU clf)
  4. 1D-CNN on 6D rotation sequences (GPU if available)

Usage (from ShinkaEvolveCarePD/ or care_pd_task/):
  python care_pd_task/preflight.py
  python care_pd_task/preflight.py --folds 0 1 2   (run multiple folds)
"""

import argparse
import collections
import os
import pickle
import sys
import time
import traceback

import numpy as np

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))


def _find_dir(name, start, levels=6):
    cur = start
    for _ in range(levels):
        c = os.path.join(cur, name)
        if os.path.isdir(c):
            return c
        p = os.path.dirname(cur)
        if p == cur:
            break
        cur = p
    raise RuntimeError(f"Cannot find '{name}' from {start}")


_CARE_PD = _find_dir("CARE-PD", _HERE)
if _CARE_PD not in sys.path:
    sys.path.insert(0, _CARE_PD)

_DATASET = os.path.join(_CARE_PD, "assets", "datasets", "BMCLab.pkl")
_FOLDS   = os.path.join(_CARE_PD, "assets", "datasets", "folds",
                         "UPDRS_Datasets", "BMCLab_6fold_participants.pkl")
_CKPT_ROOT = os.path.join(_CARE_PD, "assets", "Pretrained_checkpoints")
_SEP = "=" * 72

import torch
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------------------------
# Rotation utils (inline — no smplx dep)
# ---------------------------------------------------------------------------

def _aa_to_quat(aa):
    angles = torch.norm(aa, p=2, dim=-1, keepdim=True).clamp(min=1e-8)
    half = 0.5 * angles
    small = angles.abs() < 1e-6
    s = torch.where(small, 0.5 - angles * angles / 48, torch.sin(half) / angles)
    return torch.cat([torch.cos(half), aa * s], dim=-1)

def _quat_to_mat(q):
    r, i, j, k = torch.unbind(q, -1)
    s = 2.0 / (q * q).sum(-1)
    o = torch.stack([
        1-s*(j*j+k*k), s*(i*j-k*r),   s*(i*k+j*r),
        s*(i*j+k*r),   1-s*(i*i+k*k), s*(j*k-i*r),
        s*(i*k-j*r),   s*(j*k+i*r),   1-s*(i*i+j*j),
    ], dim=-1)
    return o.reshape(q.shape[:-1] + (3, 3))

def smpl_to_6d(pose):
    """(T, 72) → (T, 25, 6) 6D rotation + zero translation slot."""
    T  = len(pose)
    aa = torch.from_numpy(pose.reshape(T, 24, 3)).float()
    rm = _quat_to_mat(_aa_to_quat(aa))           # (T, 24, 3, 3)
    r6 = rm[..., :2, :].reshape(T, 24, 6).numpy()
    return np.concatenate([r6, np.zeros((T, 1, 6), np.float32)], 1)  # (T, 25, 6)

def pose_feat_6d(pose):
    """(T, 72) → (300,) mean+std of 6D rotation."""
    r = smpl_to_6d(pose).reshape(len(pose), -1)  # (T, 150)
    return np.concatenate([r.mean(0), r.std(0)])  # (300,)

def pose_feat_raw(pose):
    """(T, 72) → (144,) mean+std of raw axis-angle."""
    return np.concatenate([pose.mean(0), pose.std(0)])

# ---------------------------------------------------------------------------
# MotionCLIP
# ---------------------------------------------------------------------------

class _PosEnc(torch.nn.Module):
    def __init__(self, d, drop=0.1, maxlen=5000):
        super().__init__()
        self.drop = torch.nn.Dropout(drop)
        pe = torch.zeros(maxlen, d)
        pos = torch.arange(0, maxlen).float().unsqueeze(1)
        div = torch.exp(torch.arange(0, d, 2).float() * (-np.log(10000.) / d))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0).transpose(0, 1))
    def forward(self, x):
        return self.drop(x + self.pe[:x.shape[0]])

class MotionCLIPEncoder(torch.nn.Module):
    def __init__(self, njoints=25, nfeats=6, latent_dim=512,
                 ff_size=1024, num_layers=8, num_heads=4, dropout=0.1):
        super().__init__()
        self.muQ    = torch.nn.Parameter(torch.randn(1, latent_dim))
        self.sigQ   = torch.nn.Parameter(torch.randn(1, latent_dim))
        self.embed  = torch.nn.Linear(njoints * nfeats, latent_dim)
        self.pos    = _PosEnc(latent_dim, dropout)
        layer = torch.nn.TransformerEncoderLayer(
            d_model=latent_dim, nhead=num_heads,
            dim_feedforward=ff_size, dropout=dropout, activation='gelu')
        self.enc = torch.nn.TransformerEncoder(layer, num_layers=num_layers)
    def forward(self, x):
        B, T, J, F = x.shape
        x = x.permute(1, 0, 2, 3).reshape(T, B, J * F)
        x = self.embed(x)
        y = torch.zeros(B, dtype=torch.long, device=x.device)
        xseq = torch.cat([self.muQ[y].unsqueeze(0), self.sigQ[y].unsqueeze(0), x], 0)
        xseq = self.pos(xseq)
        mask = torch.ones(B, T + 2, dtype=torch.bool, device=x.device)
        return self.enc(xseq, src_key_padding_mask=~mask)[0]  # mu (B, D)


def load_motionclip():
    ckpt_path = os.path.join(_CKPT_ROOT, "motionclip",
                             "motionclip_encoder_checkpoint_0100.pth.tar")
    if not os.path.isfile(ckpt_path):
        print(f"  [!] MotionCLIP checkpoint NOT FOUND: {ckpt_path}")
        return None, None
    try:
        enc = MotionCLIPEncoder().to(DEVICE)
        ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        sd = ckpt.get('state_dict', ckpt)
        msd = enc.state_dict()
        first = next(iter(msd))
        new = collections.OrderedDict()
        for k, v in sd.items():
            if k.startswith('module.') and 'module.' not in first:
                k = k[7:]
            if k in msd:
                new[k] = v
        msd.update(new)
        enc.load_state_dict(msd, strict=True)
        enc.eval()
        for p in enc.parameters():
            p.requires_grad_(False)
        size_mb = os.path.getsize(ckpt_path) / 1e6
        print(f"  [✓] MotionCLIP loaded  ({len(new)}/{len(msd)} layers, {size_mb:.1f}MB)")
        return enc, ckpt_path
    except Exception as e:
        print(f"  [!] MotionCLIP load failed: {e}")
        return None, None


@torch.no_grad()
def encode_motionclip(enc, poses):
    embs = []
    for pose in poses:
        r = smpl_to_6d(pose)
        T = len(r)
        n = 60
        if T >= n:
            s = (T - n) // 2; r = r[s:s+n]
        else:
            r = np.tile(r, ((n+T-1)//T, 1, 1))[:n]
        x = torch.from_numpy(r).float().unsqueeze(0).to(DEVICE)
        embs.append(enc(x).cpu().numpy())
    return np.vstack(embs)

# ---------------------------------------------------------------------------
# Fold data loader
# ---------------------------------------------------------------------------

def get_fold_data(data, folds, fold_id):
    split = folds[fold_id]
    def _collect(subjects):
        poses, labels = [], []
        for sub in subjects:
            if sub not in data: continue
            for wd in data[sub].values():
                poses.append(wd['pose'].astype(np.float32))
                labels.append(int(wd['UPDRS_GAIT']))
        return poses, np.array(labels, np.int64)
    return _collect(split['train']), _collect(split['eval'])

# ---------------------------------------------------------------------------
# Classifiers
# ---------------------------------------------------------------------------

def run_rf(X_tr, y_tr, X_te):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    X_tr = sc.fit_transform(X_tr); X_te = sc.transform(X_te)
    clf = RandomForestClassifier(n_estimators=200, random_state=42,
                                 class_weight='balanced', n_jobs=-1)
    clf.fit(X_tr, y_tr)
    return clf.predict(X_te)

def run_gb(X_tr, y_tr, X_te):
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    X_tr = sc.fit_transform(X_tr); X_te = sc.transform(X_te)
    # One-vs-rest for multiclass
    from sklearn.multiclass import OneVsRestClassifier
    clf = OneVsRestClassifier(
        GradientBoostingClassifier(n_estimators=200, max_depth=4, random_state=42))
    clf.fit(X_tr, y_tr)
    return clf.predict(X_te)

def run_mlp(feat_tr, y_tr, feat_te):
    import torch.nn as nn, torch.optim as optim
    n  = 3
    w  = np.bincount(y_tr, minlength=n).astype(float)
    w  = np.where(w == 0, 1., w)
    wt = torch.tensor(w.sum()/(n*w), dtype=torch.float32).to(DEVICE)
    clf = nn.Sequential(
        nn.Linear(feat_tr.shape[1], 256), nn.GELU(), nn.Dropout(0.3),
        nn.Linear(256, 128), nn.GELU(), nn.Dropout(0.2),
        nn.Linear(128, n)).to(DEVICE)
    opt  = optim.AdamW(clf.parameters(), lr=1e-3, weight_decay=1e-3)
    loss = nn.CrossEntropyLoss(weight=wt)
    Xtr  = torch.from_numpy(feat_tr.astype(np.float32)).to(DEVICE)
    ytr  = torch.tensor(y_tr, dtype=torch.long).to(DEVICE)
    Xte  = torch.from_numpy(feat_te.astype(np.float32)).to(DEVICE)
    clf.train()
    for ep in range(200):
        idx = torch.randperm(len(Xtr))
        for s in range(0, len(Xtr), 64):
            b = idx[s:s+64]
            opt.zero_grad()
            loss(clf(Xtr[b]), ytr[b]).backward()
            opt.step()
    clf.eval()
    with torch.no_grad():
        return clf(Xte).argmax(1).cpu().numpy()

def run_cnn(train_poses, y_tr, test_poses):
    import torch.nn as nn, torch.optim as optim
    n  = 3
    w  = np.bincount(y_tr, minlength=n).astype(float)
    w  = np.where(w == 0, 1., w)
    wt = torch.tensor(w.sum()/(n*w), dtype=torch.float32).to(DEVICE)
    class Enc(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv1d(150, 128, 7, padding=3), nn.GroupNorm(8, 128), nn.GELU(),
                nn.Conv1d(128, 256, 5, padding=2), nn.GroupNorm(8, 256), nn.GELU(),
                nn.Conv1d(256, 256, 3, padding=1), nn.GroupNorm(8, 256), nn.GELU())
            self.pool = nn.AdaptiveAvgPool1d(1)
        def forward(self, x):
            return self.pool(self.net(x.unsqueeze(0))).squeeze(-1)
    enc = Enc().to(DEVICE)
    head = nn.Sequential(nn.Linear(256,128), nn.GELU(), nn.Dropout(0.3),
                         nn.Linear(128, n)).to(DEVICE)
    params = list(enc.parameters()) + list(head.parameters())
    opt = optim.AdamW(params, lr=3e-4, weight_decay=1e-4)
    crit = nn.CrossEntropyLoss(weight=wt)
    sch  = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=40)
    tensors = [torch.from_numpy(smpl_to_6d(p).reshape(len(p),-1).T).float().to(DEVICE)
               for p in train_poses]
    labels  = torch.tensor(y_tr, dtype=torch.long).to(DEVICE)
    N, bs   = len(tensors), 32
    for _ in range(40):
        idx = torch.randperm(N)
        for s in range(0, N, bs):
            batch = idx[s:s+bs]
            opt.zero_grad()
            total = torch.tensor(0., device=DEVICE)
            for i in batch:
                total += crit(head(enc(tensors[i])), labels[i:i+1])
            (total/len(batch)).backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
        sch.step()
    enc.eval(); head.eval()
    preds = []
    with torch.no_grad():
        for p in test_poses:
            x = torch.from_numpy(smpl_to_6d(p).reshape(len(p),-1).T).float().to(DEVICE)
            preds.append(int(head(enc(x)).argmax(1).item()))
    return np.array(preds)

# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def score(y_true, y_pred):
    from sklearn.metrics import f1_score, confusion_matrix
    macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
    per   = f1_score(y_true, y_pred, average=None, labels=[0,1,2], zero_division=0)
    cm    = confusion_matrix(y_true, y_pred, labels=[0,1,2])
    return macro, per, cm

def print_result(name, macro, per, cm, elapsed):
    print(f"\n  ┌─ {name}")
    print(f"  │  macro_F1   : {macro:.4f}")
    print(f"  │  F1 normal  : {per[0]:.4f}  (class 0)")
    print(f"  │  F1 mild    : {per[1]:.4f}  (class 1)")
    print(f"  │  F1 severe  : {per[2]:.4f}  (class 2) ← clinically most important")
    print(f"  │  time       : {elapsed:.1f}s")
    print(f"  │  confusion matrix (rows=true, cols=pred):")
    print(f"  │    pred→   [0]   [1]   [2]")
    for i, row in enumerate(cm):
        lbl = ['normal','mild  ','severe'][i]
        print(f"  │    true[{i}] {lbl}: {row.tolist()}")
    print(f"  └{'─'*50}")

# ---------------------------------------------------------------------------
# Main benchmark
# ---------------------------------------------------------------------------

def run_benchmark(fold_ids):
    print(_SEP)
    print("  CARE-PD PREFLIGHT BENCHMARK")
    print(_SEP)

    # Environment
    print(f"\n[env] Python  : {sys.version.split()[0]}")
    print(f"[env] PyTorch : {torch.__version__}")
    print(f"[env] Device  : {DEVICE}")
    if torch.cuda.is_available():
        print(f"[env] GPU     : {torch.cuda.get_device_name(0)}")
        print(f"[env] VRAM    : {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")

    # Checkpoint inventory
    print(f"\n[checkpoints] {_CKPT_ROOT}")
    if not os.path.isdir(_CKPT_ROOT):
        print(f"  WARNING: directory not found — MotionCLIP methods will be skipped")
    else:
        for bb in sorted(os.listdir(_CKPT_ROOT)):
            bd = os.path.join(_CKPT_ROOT, bb)
            if not os.path.isdir(bd): continue
            files = [f for f in os.listdir(bd) if os.path.isfile(os.path.join(bd, f))]
            for fn in files:
                mb = os.path.getsize(os.path.join(bd, fn)) / 1e6
                print(f"  {bb}/{fn}  ({mb:.1f} MB)")

    # Load dataset
    print(f"\n[data] Loading {_DATASET} ...")
    t = time.perf_counter()
    with open(_DATASET, 'rb') as f: data = pickle.load(f)
    with open(_FOLDS, 'rb') as f:   folds = pickle.load(f)
    print(f"[data] Loaded in {time.perf_counter()-t:.2f}s")
    n_sub = len(data)
    n_walk = sum(len(v) for v in data.values())
    labels_all = [int(wd['UPDRS_GAIT']) for ws in data.values() for wd in ws.values()]
    dist = collections.Counter(labels_all)
    print(f"[data] subjects={n_sub}  walks={n_walk}  "
          f"label_dist={{0:{dist[0]}, 1:{dist[1]}, 2:{dist[2]}}}")
    print(f"[data] Running on fold(s): {fold_ids}")

    # Load MotionCLIP once
    print(f"\n[model] Loading pretrained checkpoints ...")
    motionclip, ckpt_path = load_motionclip()

    results = {}

    for fold_id in fold_ids:
        print(f"\n{_SEP}")
        print(f"  FOLD {fold_id}")
        print(_SEP)

        (tr_poses, y_tr), (te_poses, y_te) = get_fold_data(data, folds, fold_id)
        print(f"  train: {len(tr_poses)} walks  |  test: {len(te_poses)} walks")
        print(f"  train label dist: {dict(sorted(collections.Counter(y_tr.tolist()).items()))}")
        print(f"  test  label dist: {dict(sorted(collections.Counter(y_te.tolist()).items()))}")

        fold_results = {}

        # ── Method 0: RandomForest on raw axis-angle ──────────────────────
        print(f"\n  [0] RandomForest  (raw axis-angle mean+std, 144-dim, CPU) ...")
        t0 = time.perf_counter()
        try:
            X_tr = np.stack([pose_feat_raw(p) for p in tr_poses])
            X_te = np.stack([pose_feat_raw(p) for p in te_poses])
            preds = run_rf(X_tr, y_tr, X_te)
            m, per, cm = score(y_te, preds)
            elapsed = time.perf_counter() - t0
            print_result("RandomForest / raw axis-angle", m, per, cm, elapsed)
            fold_results['rf_raw'] = m
        except Exception as e:
            print(f"  [!] FAILED: {e}"); traceback.print_exc()

        # ── Method 1: GradientBoosting on 6D rotation ────────────────────
        print(f"\n  [1] GradientBoosting  (6D rotation mean+std, 300-dim, CPU) ...")
        t0 = time.perf_counter()
        try:
            X_tr = np.stack([pose_feat_6d(p) for p in tr_poses])
            X_te = np.stack([pose_feat_6d(p) for p in te_poses])
            preds = run_gb(X_tr, y_tr, X_te)
            m, per, cm = score(y_te, preds)
            elapsed = time.perf_counter() - t0
            print_result("GradientBoosting / 6D rotation", m, per, cm, elapsed)
            fold_results['gb_6d'] = m
        except Exception as e:
            print(f"  [!] FAILED: {e}"); traceback.print_exc()

        # ── Method 2: 1D-CNN on 6D rotation ──────────────────────────────
        print(f"\n  [2] 1D-CNN  (6D rotation sequences, 150-dim, {DEVICE}) ...")
        t0 = time.perf_counter()
        try:
            preds = run_cnn(tr_poses, y_tr, te_poses)
            m, per, cm = score(y_te, preds)
            elapsed = time.perf_counter() - t0
            print_result(f"1D-CNN / 6D rotation ({DEVICE})", m, per, cm, elapsed)
            fold_results['cnn_6d'] = m
        except Exception as e:
            print(f"  [!] FAILED: {e}"); traceback.print_exc()

        # ── Method 3: MotionCLIP frozen + MLP ────────────────────────────
        if motionclip is not None:
            print(f"\n  [3] MotionCLIP frozen → MLP  ({DEVICE}) ...")
            t0 = time.perf_counter()
            try:
                f_tr = encode_motionclip(motionclip, tr_poses)
                f_te = encode_motionclip(motionclip, te_poses)
                preds = run_mlp(f_tr, y_tr, f_te)
                m, per, cm = score(y_te, preds)
                elapsed = time.perf_counter() - t0
                print_result(f"MotionCLIP frozen → MLP ({DEVICE})", m, per, cm, elapsed)
                fold_results['motionclip_mlp'] = m
            except Exception as e:
                print(f"  [!] FAILED: {e}"); traceback.print_exc()

            # ── Method 4: MotionCLIP frozen + GradientBoosting ───────────
            print(f"\n  [4] MotionCLIP frozen → GradientBoosting  ...")
            t0 = time.perf_counter()
            try:
                preds = run_gb(f_tr, y_tr, f_te)
                m, per, cm = score(y_te, preds)
                elapsed = time.perf_counter() - t0
                print_result("MotionCLIP frozen → GradientBoosting", m, per, cm, elapsed)
                fold_results['motionclip_gb'] = m
            except Exception as e:
                print(f"  [!] FAILED: {e}"); traceback.print_exc()
        else:
            print(f"\n  [3,4] MotionCLIP — SKIPPED (checkpoint not available)")

        results[fold_id] = fold_results

    # ── Summary ───────────────────────────────────────────────────────────
    print(f"\n{_SEP}")
    print("  PREFLIGHT SUMMARY")
    print(_SEP)
    method_names = {
        'rf_raw':         'RandomForest / raw axis-angle (baseline)',
        'gb_6d':          'GradientBoosting / 6D rotation',
        'cnn_6d':         f'1D-CNN / 6D rotation ({DEVICE})',
        'motionclip_mlp': f'MotionCLIP frozen → MLP ({DEVICE})',
        'motionclip_gb':  'MotionCLIP frozen → GradientBoosting',
    }
    all_methods = list(method_names.keys())
    print(f"  {'Method':<45} " + "  ".join(f"Fold{f}" for f in fold_ids) + "  Mean")
    print(f"  {'-'*45} " + "  ".join("------" for _ in fold_ids) + "  ------")
    for m in all_methods:
        scores_row = [results[f].get(m, float('nan')) for f in fold_ids]
        valid = [s for s in scores_row if not np.isnan(s)]
        mean  = np.mean(valid) if valid else float('nan')
        row   = "  ".join(f"{s:.4f}" if not np.isnan(s) else "  FAIL" for s in scores_row)
        print(f"  {method_names[m]:<45} {row}  {mean:.4f}" if not np.isnan(mean)
              else f"  {method_names[m]:<45} {row}  FAILED")

    best_method = None
    best_mean   = -1
    for m in all_methods:
        scores_row = [results[f].get(m, float('nan')) for f in fold_ids]
        valid = [s for s in scores_row if not np.isnan(s)]
        if valid and np.mean(valid) > best_mean:
            best_mean   = np.mean(valid)
            best_method = m
    if best_method:
        print(f"\n  → Best method on fold(s) {fold_ids}: "
              f"{method_names[best_method]}  ({best_mean:.4f})")
    print(f"\n  → ShinkaEvolve will start from initial.py which uses MotionCLIP frozen")
    print(f"     features by default (falls back to 1D-CNN if checkpoint unavailable).")
    print(f"\n  → Evolution will now begin. Watch for genomes that improve on these scores.")
    print(_SEP)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--folds", nargs="+", type=int, default=[1],
                        help="Fold indices to benchmark (1-indexed, default: fold 1)")
    args = parser.parse_args()
    run_benchmark(args.folds)
