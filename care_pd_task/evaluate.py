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
_DATASET  = os.path.join(_CARE_PD, "assets", "datasets", "BMCLab.pkl")
_FOLDS    = os.path.join(
    _CARE_PD, "assets", "datasets", "folds", "UPDRS_Datasets",
    "BMCLab_6fold_participants.pkl",
)

# Leaderboard lives in the care_pd_task/ dir (also found by walking up)
_TASK_DIR        = _find_dir("care_pd_task", _HERE)
_LEADERBOARD_CSV = os.path.join(_TASK_DIR, "leaderboard.csv")

_LEADERBOARD_COLS = [
    "timestamp_utc", "genome_id", "program_path",
    "macro_f1", "f1_class0", "f1_class1", "f1_class2",
    "n_walks", "mpjpe_mm",
    "fold_f1_json", "label_dist_json", "pred_dist_json",
    "combined_score", "eval_time_s",
]

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
    """Extract gen_N from the program path (Shinka stores as <results_dir>/gen_N/main.py)."""
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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(program_path: str, results_dir: str):
    t0 = time.perf_counter()

    # Load dataset + folds (cached per-process; negligible for first load ~0.5s)
    with open(_DATASET, "rb") as f:
        data = pickle.load(f)
    with open(_FOLDS, "rb") as f:
        folds = pickle.load(f)

    # Import evolved module
    try:
        mod = _load_program(program_path)
    except Exception:
        err = traceback.format_exc()
        _write_results(results_dir, {"combined_score": 0.0}, False, err)
        print(f"[ERROR] import failed:\n{err}", file=sys.stderr)
        return

    if not hasattr(mod, "run_evaluation"):
        err = "evolved program missing required function: run_evaluation(data, folds)"
        _write_results(results_dir, {"combined_score": 0.0}, False, err)
        return

    # Run evaluation — pass care_pd_dir so genomes can locate checkpoints
    try:
        result = mod.run_evaluation(data=data, folds=folds, care_pd_dir=_CARE_PD)
    except TypeError:
        # Older genomes that don't accept care_pd_dir
        result = mod.run_evaluation(data=data, folds=folds)
    except Exception:
        err = traceback.format_exc()
        _write_results(results_dir, {"combined_score": 0.0}, False, err)
        print(f"[ERROR] run_evaluation failed:\n{err}", file=sys.stderr)
        return

    # Validate
    if not isinstance(result, dict) or "combined_score" not in result:
        err = "run_evaluation() must return a dict with 'combined_score'"
        _write_results(results_dir, {"combined_score": 0.0}, False, err)
        return

    score = result.get("combined_score", 0.0)
    preds  = result.get("all_preds",  [])
    labels = result.get("all_labels", [])

    if not isinstance(score, (int, float)) or not (0.0 <= float(score) <= 1.0):
        err = f"combined_score must be float in [0,1], got: {score!r}"
        _write_results(results_dir, {"combined_score": 0.0}, False, err)
        return
    if len(preds) == 0:
        err = "No predictions produced (all folds skipped?)"
        _write_results(results_dir, {"combined_score": 0.0}, False, err)
        return

    eval_time = time.perf_counter() - t0

    # Per-class F1 and confusion matrix
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

    # Build Shinka metrics.json
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
            "mpjpe_mm":         "N/A",   # Phase 2 placeholder
            "eval_time_s":      round(eval_time, 2),
        },
    }
    _write_results(results_dir, metrics, True, None)

    # Leaderboard CSV
    _append_leaderboard({
        "timestamp_utc":   datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "genome_id":       _genome_id(program_path),
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

    print(
        f"macro_f1={score:.4f}  "
        f"f1=[{_f1(0)},{_f1(1)},{_f1(2)}]  "
        f"n_walks={len(labels)}  eval_time={eval_time:.1f}s"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--program_path", type=str, default="initial.py")
    parser.add_argument("--results_dir",  type=str, default="results")
    args = parser.parse_args()
    main(args.program_path, args.results_dir)
