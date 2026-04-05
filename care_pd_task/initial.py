"""
CARE-PD: UPDRS Gait Score Prediction from SMPL Pose Sequences
Task: 3-class classification (UPDRS_GAIT in {0, 1, 2}) on BMCLab dataset.
Data: raw SMPL pose (frames, 72) per walk. 6-fold subject-level CV.
Fitness: macro-F1 (higher is better).
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score


# EVOLVE-BLOCK-START
def extract_features(pose: np.ndarray) -> np.ndarray:
    """
    Convert a variable-length pose sequence to a fixed-size feature vector.

    Args:
        pose: np.ndarray of shape (T, 72) — SMPL joint rotation params,
              T = number of frames (varies per walk, ~210-3015).

    Returns:
        1D feature vector.
    """
    # Baseline: mean and std over time axis → (144,)
    mean_feat = pose.mean(axis=0)       # (72,)
    std_feat  = pose.std(axis=0)        # (72,)
    return np.concatenate([mean_feat, std_feat])


def train_and_predict(X_train, y_train, X_test):
    """
    Train a classifier on X_train/y_train and return predictions on X_test.

    Args:
        X_train: (n_train, n_features)
        y_train: (n_train,) int labels in {0, 1, 2}
        X_test:  (n_test, n_features)

    Returns:
        y_pred: (n_test,) int predictions
    """
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    clf = RandomForestClassifier(
        n_estimators=100, random_state=42, n_jobs=-1, class_weight="balanced"
    )
    clf.fit(X_train, y_train)
    return clf.predict(X_test)
# EVOLVE-BLOCK-END


def run_evaluation(data: dict, folds: dict, care_pd_dir: str = None) -> dict:
    """
    Run 6-fold subject-level CV.  Called by evaluate.py.

    Returns dict with:
        combined_score  — macro-F1 across all folds (primary fitness)
        all_preds       — flat list of predictions
        all_labels      — flat list of ground-truth labels
        per_fold_results — dict keyed by fold_id with per-fold macro-F1 breakdown
        per_class_f1    — list [f1_class0, f1_class1, f1_class2]
    """
    all_preds, all_labels = [], []
    per_fold_results = {}

    for fold_id, split in folds.items():
        def get_subject_data(subjects):
            X, y = [], []
            for sub in subjects:
                if sub not in data:
                    continue
                for walk_data in data[sub].values():
                    pose  = walk_data["pose"].astype(np.float32)
                    label = int(walk_data["UPDRS_GAIT"])
                    X.append(extract_features(pose))
                    y.append(label)
            return np.array(X), np.array(y)

        X_train, y_train = get_subject_data(split["train"])
        X_test,  y_test  = get_subject_data(split["eval"])

        if len(X_test) == 0 or len(np.unique(y_train)) < 2:
            continue

        preds = train_and_predict(X_train, y_train, X_test)
        all_preds.extend(preds.tolist())
        all_labels.extend(y_test.tolist())

        fold_f1 = float(f1_score(y_test, preds, average="macro", zero_division=0))
        per_fold_results[str(fold_id)] = {
            "macro_f1":      round(fold_f1, 4),
            "n_test_walks":  int(len(y_test)),
            "n_train_walks": int(len(y_train)),
        }

    macro_f1     = float(f1_score(all_labels, all_preds, average="macro",  zero_division=0))
    per_class_f1 = f1_score(all_labels, all_preds, average=None, labels=[0, 1, 2], zero_division=0)

    return {
        "combined_score":   macro_f1,
        "all_preds":        all_preds,
        "all_labels":       all_labels,
        "per_fold_results": per_fold_results,
        "per_class_f1":     per_class_f1.tolist(),
    }
