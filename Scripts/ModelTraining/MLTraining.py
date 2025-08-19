# ════════════════════════════════════════════════════════════════════════════
#  MLTraining.py  — training + calibration
# ════════════════════════════════════════════════════════════════════════════
"""
Helpers for training NB / KNN / MLP / XGB base learners, a stacking ensemble,
*calibrating* the final probabilities, and evaluating the models.
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, Any

import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             classification_report, confusion_matrix, ConfusionMatrixDisplay)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import StackingClassifier
from sklearn.calibration import CalibratedClassifierCV
from xgboost import XGBClassifier

# ─────────────────────────────────────────────────────────────────────────────
CV = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
# ─────────────────────────────────────────────────────────────────────────────
def _make_dir(path: str | Path):
    Path(path).mkdir(parents=True, exist_ok=True)

from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# ─────────────────────────────────────────────────────────────────────────────
def calibrate_prefit(estimator, X_cal, y_cal, method: str = "sigmoid") -> CalibratedClassifierCV:
    """
    Wrap a *pre-fitted* estimator with Platt ('sigmoid') or 'isotonic' calibration,
    trained on a dedicated calibration set that should reflect deployment prevalence.
    """
    # sklearn version shim
    try:
        cal = CalibratedClassifierCV(estimator=estimator, method=method, cv="prefit")
    except TypeError:
        # older/newer API alias
        cal = CalibratedClassifierCV(base_estimator=estimator, method=method, cv="prefit")
    cal.fit(X_cal, y_cal)
    return cal



# NB --------------------------------------------------------------------------
def train_NB(
    X_train, y_train, *,
    cv=CV, num_cores: int = -1, name_target_subclass: str,
):
    print(f"\n[NB] training → {name_target_subclass}")
    param_grid = {"var_smoothing": np.logspace(0, -9, 10)}
    search = RandomizedSearchCV(
        GaussianNB(),
        param_distributions=param_grid,
        cv=cv, n_jobs=num_cores, verbose=1, random_state=42
    ).fit(X_train, y_train)
    nb = search.best_estimator_
    nb.fit(X_train, y_train)
    return nb




# KNN -------------------------------------------------------------------------
def train_KNN(
    X_train, y_train, *,
    cv=CV, num_cores: int = -1, name_target_subclass: str,
):
    print(f"\n[KNN] training → {name_target_subclass}")
    knn_pipe = Pipeline([
        ("knn", KNeighborsClassifier())
    ])
    param_grid = {
        "knn__n_neighbors": [15, 30],
        "knn__weights": ["uniform", "distance"],
        "knn__algorithm": ["auto", "ball_tree", "kd_tree", "brute"],
        "knn__leaf_size": [30, 45, 60],
        "knn__p": [1, 2],
    }
    search = RandomizedSearchCV(
        knn_pipe,
        param_distributions=param_grid,
        cv=cv, n_jobs=num_cores, verbose=1, random_state=42
    ).fit(X_train, y_train)
    knn = search.best_estimator_
    knn.fit(X_train, y_train)
    return knn



# MLP -------------------------------------------------------------------------
def train_MLP(
    X_train, y_train, *,
    cv=CV, num_cores: int = -1, name_target_subclass: str,
):
    print(f"\n[MLP] training → {name_target_subclass}")
    mlp_pipe = Pipeline([
        ("mlp", MLPClassifier())
    ])
    param_grid = {
        "mlp__hidden_layer_sizes": [(50,50,50), (50,100,50), (100,)],
        "mlp__activation": ["tanh", "relu"],
        "mlp__solver": ["sgd", "adam"],
        "mlp__alpha": [0.0001, 0.05],
        "mlp__learning_rate": ["constant", "adaptive"],
        "mlp__early_stopping": [True],
    }
    search = RandomizedSearchCV(
        mlp_pipe,
        param_distributions=param_grid,
        cv=cv, n_jobs=num_cores, verbose=1, random_state=42
    ).fit(X_train, y_train)
    mlp = search.best_estimator_
    mlp.fit(X_train, y_train)
    return mlp




# XGB -------------------------------------------------------------------------
def train_XGB(
    X_train, y_train, *,
    cv=CV, num_cores: int = -1, name_target_subclass: str,
):
    print(f"\n[XGB] training → {name_target_subclass}")
    param_grid = {
        "booster": ["gbtree"],
        "max_depth": [3, 6, 9, 12, 15],
        "gamma": [0, 0.3, 0.7],
        "eta": [0.01, 0.1, 0.2, 0.3],
        "max_delta_step": [0, 1, 3],
        "subsample": [0.7, 1],
        "colsample_bytree": [0.5, 0.7, 1],
        "reg_lambda": [40, 100, 200],
        "reg_alpha": [0, 0.5, 1],
        "objective": ["binary:logistic"],
        "eval_metric": ["logloss"],
        "n_estimators": [100, 250, 500],
        "min_child_weight": [0, 1, 5, 10],
        "tree_method": ["auto"],
    }
    search = RandomizedSearchCV(
        XGBClassifier(),
        param_distributions=param_grid,
        cv=cv, scoring="neg_log_loss",
        n_jobs=num_cores, verbose=1, random_state=42
    ).fit(X_train, y_train)
    xgb = search.best_estimator_
    xgb.fit(X_train, y_train)
    return xgb

def _unwrap_for_stacking(model):
    try:
        from sklearn.calibration import CalibratedClassifierCV
        if isinstance(model, CalibratedClassifierCV):
            # prefer .estimator (newer sklearn); fallback to .base_estimator
            return getattr(model, "estimator", getattr(model, "base_estimator", model))
        return model
    except Exception:
        return model

import numpy as np
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss, log_loss

from typing import Optional, Tuple
from sklearn.calibration import CalibratedClassifierCV

def get_raw_estimator(calibrated) -> Optional[object]:
    """
    If *calibrated* is a CalibratedClassifierCV with cv='prefit',
    return its underlying estimator (handles sklearn API variants).
    """
    if isinstance(calibrated, CalibratedClassifierCV):
        if hasattr(calibrated, "estimator"):      # modern sklearn
            return calibrated.estimator
        if hasattr(calibrated, "base_estimator"): # older sklearn
            return calibrated.base_estimator
    return None

def proba_both(calibrated, X) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return (uncalibrated_proba, calibrated_proba) for P(y=1).
    """
    raw = get_raw_estimator(calibrated)
    if raw is None:
        raise ValueError("Provided model is not a prefit CalibratedClassifierCV.")
    y_proba_uncal = raw.predict_proba(X)[:, 1]
    y_proba_cal   = calibrated.predict_proba(X)[:, 1]
    return y_proba_uncal, y_proba_cal


def plot_calibration_curve(
    y_true,
    proba_list,
    clf_names,
    *,
    n_bins=15,
    strategy="quantile",
    ax=None,
    title="Calibration (Reliability) Curve",
    annotate_metrics=True,
):
    """
    y_true:       array-like (0/1)
    proba_list:   list of arrays of shape (n_samples,) with predicted P(y=1)
    clf_names:    list of labels for legend (same length as proba_list)
    n_bins:       number of bins for calibration_curve
    strategy:     'uniform' or 'quantile'
    """
    assert len(proba_list) == len(clf_names), "Names must match prob arrays"

    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 5))

    # Perfect calibration reference
    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1, label="Perfect")

    # Plot each model curve
    curves_labels = []
    for p, name in zip(proba_list, clf_names):
        # Guard: clip to (0,1)
        p = np.clip(p, 1e-9, 1 - 1e-9)
        frac_pos, mean_pred = calibration_curve(y_true, p, n_bins=n_bins, strategy=strategy)

        # Plot curve
        ax.plot(mean_pred, frac_pos, marker="o", linewidth=1, label=name)

        if annotate_metrics:
            try:
                bs  = brier_score_loss(y_true, p)
                ll  = log_loss(y_true, p)
                curves_labels.append((name, bs, ll))
            except Exception:
                pass

    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives")
    ax.set_title(title)
    ax.legend(loc="best")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True, linestyle=":", linewidth=0.5)

    # Optional: print metrics to console for quick check
    if annotate_metrics and curves_labels:
        print("\nCalibration metrics:")
        for name, bs, ll in curves_labels:
            print(f"  {name:>13s}  |  Brier: {bs:.4f}   LogLoss: {ll:.4f}")

    return ax


# ------------------------------------------------------------------ I/O helper
# ─────────────────────────────────────────────────────────────────────────────
# small I/O utilities
# ─────────────────────────────────────────────────────────────────────────────

def save_models(models: dict, out_dir: str | Path, tag: str) -> None:
    """
    Dump each estimator in *models* to
        <out_dir>/<tag>_<key>.joblib

    Example
    -------
    >>> save_models(
    ...     {"NB": nb, "Stacked": stack},
    ...     out_dir="Models/Broad_B_Naive",
    ...     tag="Broad_B_Naive",
    ... )
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for key, mdl in models.items():
        joblib.dump(mdl, out_dir / f"{tag}_{key}.joblib")



# ─────────────────────────────────────────────────────────────────────────────
# Evaluation helper exposed to users / notebooks
# ─────────────────────────────────────────────────────────────────────────────
def evaluate_classifier(
    model,
    X_test,
    y_test,
    *,
    average: str | None = None,
    plot_cm: bool = False,
    cm_path: str | Path | None = None,
) -> Dict[str, Any]:
    if average is None:
        average = "binary" if len(np.unique(y_test)) == 2 else "weighted"

    y_pred = model.predict(X_test)
    metrics: Dict[str, Any] = {
        "accuracy":  accuracy_score (y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average=average, zero_division=0),
        "recall":    recall_score   (y_test, y_pred, average=average, zero_division=0),
        "f1":        f1_score       (y_test, y_pred, average=average, zero_division=0),
        "report":    classification_report(y_test, y_pred, zero_division=0),
    }

    if plot_cm:
        cm_path = Path(cm_path or "confusion_matrix.png")
        cm_path.parent.mkdir(parents=True, exist_ok=True)
        cm = confusion_matrix(y_test, y_pred)
        ConfusionMatrixDisplay(cm).plot(cmap="Blues")
        plt.savefig(cm_path, dpi=300, bbox_inches="tight")
        plt.close()
        metrics["cm_path"] = str(cm_path)

    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# CSV appender
# ─────────────────────────────────────────────────────────────────────────────
def append_metrics_csv(
    metrics_df: pd.DataFrame,
    csv_path: str | Path,
    *,
    drop_report: bool = True,
) -> None:
    csv_path = Path(csv_path)
    if drop_report and "report" in metrics_df.columns:
        metrics_df = metrics_df.drop(columns=["report"])

    header = not csv_path.exists()
    metrics_df.to_csv(csv_path, mode="a", header=header, index=False)
