"""
ShinkaEvolve evaluation harness — CARE-PD UPDRS gait prediction (Phase 1).

Fully standalone: no shinka import needed. Runs under the same venv Python
that launches shinka_run (guaranteed by the sys.executable fix in scheduler.py).

Writes:
  <results_dir>/metrics.json   — Shinka's required output
  <results_dir>/correct.json   — Shinka's required output
  care_pd_task/leaderboard.csv — persistent leaderboard across all genomes
"""

import argparse
import csv
import importlib.util
import json
import os
import pickle
import sys
import time
import traceback
from collections import Counter
from datetime import datetime, timezone

# ---------------------------------------------------------------------------
# Paths — robust to being copied into a subdirectory by Shinka
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))


def _find_dir(name: str, start: str, max_levels: int = 6) -> str:
    """Walk up ancestor directories until a sibling named `name` is found."""
    current = start
    for _ in range(max_levels):
        candidate = os.path.join(current, name)
        if os.path.isdir(candidate):
            return candidate
        parent = os.path.dirname(current)
        if parent == current:
            break
        current = parent
    raise RuntimeError(
        f"Could not locate '{name}' directory searching up from {start}"
    )


_CARE_PD = _find_dir("CARE-PD", _HERE)

# Make CARE-PD importable from evolved genomes.
# Genomes can do: from data.preprocessing.preprocessing_utils import ...
#                 from model.backbone_loader import load_pretrained_backbone
#                 from model.motionclip.transformer import Encoder_TRANSFORMER
#                 from learning.criterion import FocalLoss, WCELoss
if _CARE_PD not in sys.path:
    sys.path.insert(0, _CARE_PD)

_DATASET  = os.path.join(_CARE_PD, "assets", "datasets", "BMCLab.pkl")
_FOLDS    = os.path.join(
    _CARE_PD, "assets", "datasets", "folds", "UPDRS_Datasets",
    "BMCLab_6fold_participants.pkl",
)

_TASK_DIR        = _find_dir("care_pd_task", _HERE)
_LEADERBOARD_CSV = os.path.join(_TASK_DIR, "leaderboard.csv")

_LEADERBOARD_COLS = [
    "timestamp_utc", "genome_id", "program_path",
    "macro_f1", "f1_class0", "f1_class1", "f1_class2",
    "n_walks", "mpjpe_mm",
    "fold_f1_json", "label_dist_json", "pred_dist_json",
    "combined_score", "eval_time_s",
]

_SEP = "=" * 72


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _load_program(program_path: str):
    spec   = importlib.util.spec_from_file_location("evolved_program", program_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_results(results_dir: str, metrics: dict, correct: bool, error):
    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(results_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)
    with open(os.path.join(results_dir, "correct.json"), "w") as f:
        json.dump({"correct": correct, "error": error}, f, indent=4)


def _genome_id(program_path: str) -> str:
    parts = os.path.normpath(program_path).split(os.sep)
    for p in reversed(parts):
        if p.startswith("gen_"):
            return p
    return os.path.splitext(os.path.basename(program_path))[0]


def _append_leaderboard(row: dict):
    file_exists = os.path.isfile(_LEADERBOARD_CSV)
    try:
        with open(_LEADERBOARD_CSV, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=_LEADERBOARD_COLS)
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)
    except Exception as e:
        print(f"[leaderboard] WARNING: {e}", file=sys.stderr)


def _gpu_info() -> str:
    try:
        import torch
        if torch.cuda.is_available():
            name   = torch.cuda.get_device_name(0)
            mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            return f"CUDA  device={name}  vram={mem_gb:.1f}GB  gpus={torch.cuda.device_count()}"
        return "CUDA not available — CPU only"
    except ImportError:
        return "torch not installed"


def _dataset_summary(data: dict, folds: dict) -> str:
    n_subjects = len(data)
    n_walks    = sum(len(ws) for ws in data.values())
    labels_all = [int(wd["UPDRS_GAIT"])
                  for ws in data.values() for wd in ws.values()]
    dist = Counter(labels_all)
    return (f"subjects={n_subjects}  walks={n_walks}  "
            f"label_dist={{0:{dist[0]}, 1:{dist[1]}, 2:{dist[2]}}}  "
            f"folds={len(folds)}")


def _checkpoint_inventory() -> str:
    ckpt_root = os.path.join(_CARE_PD, "assets", "Pretrained_checkpoints")
    if not os.path.isdir(ckpt_root):
        return "  (pretrained_checkpoints dir not found)"
    lines = []
    for backbone in sorted(os.listdir(ckpt_root)):
        bd = os.path.join(ckpt_root, backbone)
        if not os.path.isdir(bd):
            continue
        files = [f for f in os.listdir(bd) if os.path.isfile(os.path.join(bd, f))]
        sizes = [os.path.getsize(os.path.join(bd, f)) / 1e6 for f in files]
        if files:
            fstr = ", ".join(f"{fn} ({sz:.1f}MB)" for fn, sz in zip(files, sizes))
            lines.append(f"  {backbone}: {fstr}")
        else:
            lines.append(f"  {backbone}: (empty)")
    return "\n".join(lines) if lines else "  (no checkpoints found)"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(program_path: str, results_dir: str):
    t0       = time.perf_counter()
    gid      = _genome_id(program_path)
    abs_prog = os.path.abspath(program_path)

    print(_SEP)
    print(f"[evaluate] genome:      {gid}")
    print(f"[evaluate] program:     {abs_prog}")
    print(f"[evaluate] results_dir: {os.path.abspath(results_dir)}")
    print(f"[evaluate] care_pd_dir: {_CARE_PD}")
    print(f"[evaluate] gpu:         {_gpu_info()}")
    print(_SEP)

    # ── Dataset ──────────────────────────────────────────────────────────────
    t_data = time.perf_counter()
    print(f"[evaluate] Loading dataset from {_DATASET} ...", flush=True)
    with open(_DATASET, "rb") as f:
        data = pickle.load(f)
    with open(_FOLDS, "rb") as f:
        folds = pickle.load(f)
    print(f"[evaluate] Dataset loaded in {time.perf_counter()-t_data:.2f}s")
    print(f"[evaluate] {_dataset_summary(data, folds)}")

    # ── Pretrained checkpoints inventory ─────────────────────────────────────
    print(f"[evaluate] Pretrained checkpoints available:")
    print(_checkpoint_inventory())
    print(_SEP)

    # ── Import genome ─────────────────────────────────────────────────────────
    print(f"[evaluate] Importing genome ...", flush=True)
    t_import = time.perf_counter()
    try:
        mod = _load_program(program_path)
    except Exception:
        err = traceback.format_exc()
        print(f"[evaluate] IMPORT FAILED after {time.perf_counter()-t_import:.2f}s")
        print(err)
        _write_results(results_dir, {"combined_score": 0.0}, False, err)
        return
    print(f"[evaluate] Import OK ({time.perf_counter()-t_import:.2f}s)")

    if not hasattr(mod, "run_evaluation"):
        err = "genome missing required function: run_evaluation(data, folds)"
        print(f"[evaluate] ERROR: {err}")
        _write_results(results_dir, {"combined_score": 0.0}, False, err)
        return

    # ── Run evaluation ────────────────────────────────────────────────────────
    print(f"[evaluate] Running run_evaluation() ...", flush=True)
    t_eval = time.perf_counter()
    try:
        result = mod.run_evaluation(data=data, folds=folds, care_pd_dir=_CARE_PD)
    except TypeError:
        result = mod.run_evaluation(data=data, folds=folds)
    except Exception:
        err = traceback.format_exc()
        elapsed = time.perf_counter() - t_eval
        print(f"[evaluate] RUN_EVALUATION FAILED after {elapsed:.2f}s")
        print(err)
        _write_results(results_dir, {"combined_score": 0.0}, False, err)
        return
    eval_time = time.perf_counter() - t0

    # ── Validate result ───────────────────────────────────────────────────────
    if not isinstance(result, dict) or "combined_score" not in result:
        err = "run_evaluation() must return a dict with 'combined_score'"
        print(f"[evaluate] ERROR: {err}  got: {type(result)}")
        _write_results(results_dir, {"combined_score": 0.0}, False, err)
        return

    score  = result.get("combined_score", 0.0)
    preds  = result.get("all_preds",  [])
    labels = result.get("all_labels", [])

    if not isinstance(score, (int, float)) or not (0.0 <= float(score) <= 1.0):
        err = f"combined_score must be float in [0,1], got: {score!r}"
        print(f"[evaluate] ERROR: {err}")
        _write_results(results_dir, {"combined_score": 0.0}, False, err)
        return
    if len(preds) == 0:
        err = "No predictions produced (all folds skipped?)"
        print(f"[evaluate] ERROR: {err}")
        _write_results(results_dir, {"combined_score": 0.0}, False, err)
        return

    # ── Metrics ───────────────────────────────────────────────────────────────
    per_class_f1 = result.get("per_class_f1", [None, None, None])
    per_fold     = result.get("per_fold_results", {})

    try:
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(labels, preds, labels=[0, 1, 2]).tolist()
    except Exception:
        cm = None

    label_dist = dict(Counter(int(x) for x in labels))
    pred_dist  = dict(Counter(int(x) for x in preds))

    def _f1(i):
        v = per_class_f1[i] if per_class_f1 and i < len(per_class_f1) else None
        return round(float(v), 4) if v is not None else None

    # ── Print full results ────────────────────────────────────────────────────
    print(_SEP)
    print(f"[evaluate] RESULTS — {gid}")
    print(f"  macro_f1        : {float(score):.4f}   (baseline 0.5650, LOSO SOTA 0.68)")
    print(f"  f1_class0_normal: {_f1(0)}   (n={label_dist.get(0, 0)} walks)")
    print(f"  f1_class1_mild  : {_f1(1)}   (n={label_dist.get(1, 0)} walks)")
    print(f"  f1_class2_severe: {_f1(2)}   (n={label_dist.get(2, 0)} walks)  ← clinically most important")
    print(f"  n_walks_total   : {len(labels)}")
    print(f"  eval_time       : {eval_time:.1f}s")
    if cm:
        print(f"  confusion_matrix:")
        print(f"    pred→  [0]   [1]   [2]")
        for i, row in enumerate(cm):
            label = ["normal", "mild  ", "severe"][i]
            print(f"    true[{i}] {label}: {row}")
    if per_fold:
        print(f"  per_fold_f1:")
        for fid, fdata in sorted(per_fold.items()):
            print(f"    fold {fid}: {fdata.get('macro_f1', '?'):.4f}"
                  f"  (test={fdata.get('n_test_walks','?')}, train={fdata.get('n_train_walks','?')})")
    print(f"  pred_dist  : {dict(sorted(pred_dist.items()))}")
    print(f"  label_dist : {dict(sorted(label_dist.items()))}")
    print(_SEP)

    # ── Write outputs ─────────────────────────────────────────────────────────
    metrics = {
        "combined_score": float(score),
        "public": {
            "macro_f1":         round(float(score), 4),
            "n_walks":          len(labels),
            "f1_class0_normal": _f1(0),
            "f1_class1_mild":   _f1(1),
            "f1_class2_severe": _f1(2),
            "per_fold_f1":      per_fold,
            "confusion_matrix": cm,
            "label_dist":       label_dist,
            "pred_dist":        pred_dist,
            "mpjpe_mm":         "N/A",
            "eval_time_s":      round(eval_time, 2),
        },
    }
    _write_results(results_dir, metrics, True, None)

    _append_leaderboard({
        "timestamp_utc":   datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "genome_id":       gid,
        "program_path":    program_path,
        "macro_f1":        round(float(score), 4),
        "f1_class0":       _f1(0) if _f1(0) is not None else "",
        "f1_class1":       _f1(1) if _f1(1) is not None else "",
        "f1_class2":       _f1(2) if _f1(2) is not None else "",
        "n_walks":         len(labels),
        "mpjpe_mm":        "N/A",
        "fold_f1_json":    json.dumps({k: v["macro_f1"] for k, v in per_fold.items()}),
        "label_dist_json": json.dumps(label_dist),
        "pred_dist_json":  json.dumps(pred_dist),
        "combined_score":  round(float(score), 4),
        "eval_time_s":     round(eval_time, 2),
    })
    print(f"[evaluate] Leaderboard updated: {_LEADERBOARD_CSV}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--program_path", type=str, default="initial.py")
    parser.add_argument("--results_dir",  type=str, default="results")
    args = parser.parse_args()
    main(args.program_path, args.results_dir)
