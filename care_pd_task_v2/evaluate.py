"""
ShinkaEvolve evaluation harness — CARE-PD v2 UPDRS gait prediction.

Fully standalone: no shinka import needed. Leaderboard stored inside care_pd_task_v2/.
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


def _find_dir(name: str, start: str, max_levels: int = 8) -> str:
    current = start
    for _ in range(max_levels):
        candidate = os.path.join(current, name)
        if os.path.isdir(candidate):
            return candidate
        parent = os.path.dirname(current)
        if parent == current:
            break
        current = parent
    raise RuntimeError(f"Could not locate '{name}' searching up from {start}")


_CARE_PD = _find_dir("CARE-PD", _HERE)

if _CARE_PD not in sys.path:
    sys.path.insert(0, _CARE_PD)

_DATASET  = os.path.join(_CARE_PD, "assets", "datasets", "BMCLab.pkl")
_FOLDS    = os.path.join(
    _CARE_PD, "assets", "datasets", "folds", "UPDRS_Datasets",
    "BMCLab_6fold_participants.pkl",
)

_TASK_DIR        = _find_dir("care_pd_task_v2", _HERE)
_LEADERBOARD_CSV = os.path.join(_TASK_DIR, "leaderboard.csv")

# Auxiliary UPDRS-labeled datasets for extra training data (test is always BMCLab only)
_AUX_DATASETS = ["T-SDU-PD", "PD-GaM", "3DGait"]

_LEADERBOARD_COLS = [
    "timestamp_utc", "genome_id", "program_path",
    "macro_f1", "f1_class0", "f1_class1", "f1_class2",
    "delta_macro", "delta_c2",
    "n_walks", "mpjpe_mm",
    "fold_f1_json", "label_dist_json", "pred_dist_json",
    "combined_score", "eval_time_s",
]

BASELINE_MACRO = 0.5646
BASELINE_C2    = 0.4966
_SEP = "=" * 72


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
            return f"CUDA  {name}  {mem_gb:.1f}GB  gpus={torch.cuda.device_count()}"
        return "CPU only (no CUDA)"
    except ImportError:
        return "torch not installed"


def _dataset_summary(data: dict, folds: dict) -> str:
    n_walks = sum(len(ws) for ws in data.values())
    labels  = [int(wd["UPDRS_GAIT"]) for ws in data.values() for wd in ws.values()]
    dist    = Counter(labels)
    trange  = [(wd["pose"].shape[0]) for ws in data.values() for wd in ws.values()]
    return (f"subjects={len(data)}  walks={n_walks}  "
            f"label_dist={{0:{dist[0]},1:{dist[1]},2:{dist[2]}}}  "
            f"T_range={min(trange)}-{max(trange)} frames  folds={len(folds)}")


def _checkpoint_inventory() -> str:
    ckpt_root = os.path.join(_CARE_PD, "assets", "Pretrained_checkpoints")
    if not os.path.isdir(ckpt_root):
        return "  (pretrained_checkpoints dir not found)"
    lines = []
    for backbone in sorted(os.listdir(ckpt_root)):
        bd = os.path.join(ckpt_root, backbone)
        if not os.path.isdir(bd):
            continue
        files = [f for f in os.listdir(bd) if os.path.isfile(os.path.join(bd, f))
                 and not f.startswith('.')]
        if files:
            fstr = "  ".join(f"{fn}({os.path.getsize(os.path.join(bd,fn))/1e6:.0f}MB)"
                              for fn in files)
            lines.append(f"  {backbone}: {fstr}")
    return "\n".join(lines) if lines else "  (no checkpoints)"


def main(program_path: str, results_dir: str):
    t0  = time.perf_counter()
    gid = _genome_id(program_path)

    print(_SEP)
    print(f"[evaluate] genome:      {gid}")
    print(f"[evaluate] program:     {os.path.abspath(program_path)}")
    print(f"[evaluate] results_dir: {os.path.abspath(results_dir)}")
    print(f"[evaluate] care_pd_dir: {_CARE_PD}")
    print(f"[evaluate] leaderboard: {_LEADERBOARD_CSV}")
    print(f"[evaluate] gpu:         {_gpu_info()}")
    print(_SEP)

    # ── Dataset ──────────────────────────────────────────────────────────────
    print(f"[evaluate] Loading dataset...", flush=True)
    with open(_DATASET, "rb") as f:
        data = pickle.load(f)
    with open(_FOLDS, "rb") as f:
        folds = pickle.load(f)
    print(f"[evaluate] {_dataset_summary(data, folds)}")

    # ── Auxiliary UPDRS datasets (extra training data, test stays BMCLab) ────
    aux_data = {}
    ds_root = os.path.join(_CARE_PD, "assets", "datasets")
    for ds_name in _AUX_DATASETS:
        pkl_path = os.path.join(ds_root, f"{ds_name}.pkl")
        if not os.path.exists(pkl_path):
            print(f"[evaluate] aux {ds_name}: NOT FOUND (skipping)")
            continue
        with open(pkl_path, "rb") as f:
            ds = pickle.load(f)
        # Remap 4-class labels → 3-class: class 3 (very severe) → class 2 (severe)
        n_walks = n_remapped = 0
        for subj in ds.values():
            for wd in subj.values():
                if wd.get("UPDRS_GAIT") is not None:
                    if int(wd["UPDRS_GAIT"]) == 3:
                        wd["UPDRS_GAIT"] = 2
                        n_remapped += 1
                    n_walks += 1
        aux_data[ds_name] = ds
        print(f"[evaluate] aux {ds_name}: {n_walks} walks  (remapped class3→2: {n_remapped})")
    if aux_data:
        total_aux = sum(sum(len(s) for s in d.values()) for d in aux_data.values())
        print(f"[evaluate] aux total: {total_aux} extra training walks from {list(aux_data.keys())}")

    # ── Checkpoints ──────────────────────────────────────────────────────────
    print(f"[evaluate] Pretrained checkpoints:")
    print(_checkpoint_inventory())
    print(_SEP)

    # ── Import genome ─────────────────────────────────────────────────────────
    print(f"[evaluate] Importing genome...", flush=True)
    try:
        mod = _load_program(program_path)
    except Exception:
        err = traceback.format_exc()
        print(f"[evaluate] IMPORT FAILED\n{err}")
        _write_results(results_dir, {"combined_score": 0.0}, False, err)
        return

    if not hasattr(mod, "run_evaluation"):
        err = "genome missing run_evaluation(data, folds, care_pd_dir=None)"
        _write_results(results_dir, {"combined_score": 0.0}, False, err)
        return

    # ── Run evaluation ────────────────────────────────────────────────────────
    print(f"[evaluate] Running run_evaluation()...", flush=True)
    t_eval = time.perf_counter()
    try:
        result = mod.run_evaluation(data=data, folds=folds, care_pd_dir=_CARE_PD, aux_data=aux_data)
    except TypeError:
        try:
            result = mod.run_evaluation(data=data, folds=folds, care_pd_dir=_CARE_PD)
        except TypeError:
            result = mod.run_evaluation(data=data, folds=folds)
    except Exception:
        err = traceback.format_exc()
        print(f"[evaluate] RUN_EVALUATION FAILED after {time.perf_counter()-t_eval:.2f}s")
        print(err)
        _write_results(results_dir, {"combined_score": 0.0}, False, err)
        return
    eval_time = time.perf_counter() - t0

    # ── Validate ──────────────────────────────────────────────────────────────
    if not isinstance(result, dict) or "combined_score" not in result:
        err = f"run_evaluation() must return dict with 'combined_score', got {type(result)}"
        _write_results(results_dir, {"combined_score": 0.0}, False, err)
        return

    score  = float(result.get("combined_score", 0.0))
    preds  = result.get("all_preds",  [])
    labels = result.get("all_labels", [])

    if not (0.0 <= score <= 1.0):
        err = f"combined_score must be in [0,1], got {score}"
        _write_results(results_dir, {"combined_score": 0.0}, False, err)
        return
    if not preds:
        err = "No predictions (all folds skipped?)"
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

    delta_macro = round(score - BASELINE_MACRO, 4)
    delta_c2    = round((_f1(2) or 0.0) - BASELINE_C2, 4)
    arrow_m     = "↑" if delta_macro > 0.005 else ("↓" if delta_macro < -0.005 else "→")
    arrow_c2    = "↑" if delta_c2 > 0.005 else ("↓" if delta_c2 < -0.005 else "→")

    # ── Print results table ───────────────────────────────────────────────────
    print(_SEP)
    print(f"[evaluate] RESULTS — {gid}")
    print(f"  {'Metric':<24} {'This genome':>10}  {'Baseline':>10}  {'Delta':>8}")
    print(f"  {'─'*24}  {'─'*10}  {'─'*10}  {'─'*8}")
    print(f"  {'macro_f1':<24} {score:>10.4f}  {BASELINE_MACRO:>10.4f}  {delta_macro:>+8.4f} {arrow_m}")
    print(f"  {'f1_class0 (normal)':<24} {(_f1(0) or 0):>10.4f}  {0.7024:>10.4f}  "
          f"{(_f1(0) or 0)-0.7024:>+8.4f}")
    print(f"  {'f1_class1 (mild PD)':<24} {(_f1(1) or 0):>10.4f}  {0.4949:>10.4f}  "
          f"{(_f1(1) or 0)-0.4949:>+8.4f}")
    print(f"  {'f1_class2 (severe PD) ◄':<24} {(_f1(2) or 0):>10.4f}  {BASELINE_C2:>10.4f}  "
          f"{delta_c2:>+8.4f} {arrow_c2}  ← most important")
    print(f"  {'eval_time_s':<24} {eval_time:>10.1f}")
    print(f"  {'n_walks':<24} {len(labels):>10}")
    if cm:
        print(f"\n  Confusion matrix (rows=true, cols=pred):")
        print(f"    pred→   [0]   [1]   [2]")
        for i, row in enumerate(cm):
            lbl = ["normal","mild  ","severe"][i]
            print(f"    true[{i}] {lbl}: {row}")
    if per_fold:
        print(f"\n  Per-fold macro_f1:")
        for fid, fd in sorted(per_fold.items()):
            f1v = fd.get('macro_f1', 0)
            arrow = "↑" if f1v > BASELINE_MACRO else ("↓" if f1v < BASELINE_MACRO - 0.02 else "→")
            print(f"    fold {fid}: {f1v:.4f} {arrow}  "
                  f"(test={fd.get('n_test_walks','?')}, train={fd.get('n_train_walks','?')})")
    print(f"\n  pred_dist : {dict(sorted(pred_dist.items()))}")
    print(f"  label_dist: {dict(sorted(label_dist.items()))}")
    print(_SEP)

    # ── Write outputs ─────────────────────────────────────────────────────────
    metrics = {
        "combined_score": score,
        "public": {
            "macro_f1":         round(score, 4),
            "n_walks":          len(labels),
            "f1_class0_normal": _f1(0),
            "f1_class1_mild":   _f1(1),
            "f1_class2_severe": _f1(2),
            "delta_macro_vs_baseline": delta_macro,
            "delta_c2_vs_baseline":    delta_c2,
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
        "macro_f1":        round(score, 4),
        "f1_class0":       _f1(0) or "",
        "f1_class1":       _f1(1) or "",
        "f1_class2":       _f1(2) or "",
        "delta_macro":     delta_macro,
        "delta_c2":        delta_c2,
        "n_walks":         len(labels),
        "mpjpe_mm":        "N/A",
        "fold_f1_json":    json.dumps({k: v["macro_f1"] for k, v in per_fold.items()}),
        "label_dist_json": json.dumps(label_dist),
        "pred_dist_json":  json.dumps(pred_dist),
        "combined_score":  round(score, 4),
        "eval_time_s":     round(eval_time, 2),
    })
    print(f"[evaluate] Leaderboard: {_LEADERBOARD_CSV}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--program_path", default="initial.py")
    parser.add_argument("--results_dir",  default="results_test")
    args = parser.parse_args()
    main(args.program_path, args.results_dir)
