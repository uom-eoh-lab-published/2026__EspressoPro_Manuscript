# ════════════════════════════════════════════════════════════════════════════
#  MLTraining.py — training + calibration + evaluation
#    + XGB additive contributions on POSITIVE-CLASS margin (robust direction)
#    + True SHAP (shap.TreeExplainer) values + beeswarm plotting
#    + Platt calibration plot helper with LogLoss/Brier in legend
#    + Export per-class pre/post Platt metrics
#    + Minimal embedding plots for per-class train data (pos vs rest) + legend
#      - user-selectable embedding source: adata.obsm / adata.obs / train_df cols
#      - robust palette lookup (handles underscores/spaces/case)
# ════════════════════════════════════════════════════════════════════════════

from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, Optional, Tuple, Union, List

import re
import warnings
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


from matplotlib.patches import Patch
import matplotlib.colors as mcolors

from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, ConfusionMatrixDisplay,
)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import brier_score_loss, log_loss

from xgboost import XGBClassifier


# ─────────────────────────────────────────────────────────────────────────────
# CV DEFAULT
# ─────────────────────────────────────────────────────────────────────────────
CV = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)


# ─────────────────────────────────────────────────────────────────────────────
# Small I/O utilities
# ─────────────────────────────────────────────────────────────────────────────
def _make_dir(path: Union[str, Path]) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def save_models(models: dict, out_dir: Union[str, Path], tag: str) -> None:
    """
    Dump each estimator in *models* to:
        <out_dir>/<tag>_<key>.joblib
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for key, mdl in models.items():
        joblib.dump(mdl, out_dir / f"{tag}_{key}.joblib")


# ─────────────────────────────────────────────────────────────────────────────
# General convenience helpers (for main script)
# ─────────────────────────────────────────────────────────────────────────────
def safe_name(s: str) -> str:
    s = str(s).strip()
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^A-Za-z0-9_\-\.]+", "", s)
    return s if s else "Unknown"


def norm_feats(names) -> pd.Index:
    """
    Normalize feature names to facilitate matching across naming variants.
    """
    s = pd.Index(map(str, names))
    s = (
        s.str.strip()
         .str.lower()
         .str.replace(r"[ _/]+", "-", regex=True)
         .str.replace(r"-+", "-", regex=True)
         .str.strip("-")
    )
    return s


def attach_celltype(df: pd.DataFrame, ad: "Any", field: str) -> pd.DataFrame:
    """
    Attach an AnnData.obs label column to a dataframe by index alignment.
    Requires df.index to align with ad.obs_names.
    """
    if not hasattr(ad, "obs") or field not in ad.obs:
        raise KeyError(f"'{field}' not found in AnnData.obs")

    # NOTE: we keep underscores to be consistent with earlier pipeline outputs,
    # but palette lookup is robust to underscores/spaces/case.
    lab = (
        ad.obs[field]
          .astype("string")
          .str.strip()
          .str.replace(r"\s+", "_", regex=True)
    )

    out = df.copy()
    out["Celltype"] = pd.Categorical(lab.reindex(out.index))
    if out["Celltype"].isna().any():
        missing = int(out["Celltype"].isna().sum())
        warnings.warn(f"{missing} rows got NaN Celltype after reindex; check barcode alignment.")
    return out


def check_finite(df: pd.DataFrame, tag: str) -> None:
    arr = df.to_numpy()
    if not np.isfinite(arr).all():
        bad = np.where(~np.isfinite(arr))
        raise ValueError(f"Non-finite values found in {tag} features at positions {bad}")


# ─────────────────────────────────────────────────────────────────────────────
# Wrapper unwrapping helpers
# ─────────────────────────────────────────────────────────────────────────────
def unwrap_estimator(m):
    """
    Best-effort unwrap for common sklearn wrappers (SearchCV, CalibratedClassifierCV, etc.).
    Returns an object that should expose model-specific attributes (e.g. get_booster()).
    """
    if m is None:
        return None

    # SearchCV objects (RandomizedSearchCV / GridSearchCV)
    if hasattr(m, "best_estimator_"):
        try:
            return m.best_estimator_
        except Exception:
            pass

    # CalibratedClassifierCV and other wrappers
    return getattr(m, "estimator", None) or getattr(m, "base_estimator", None) or m


def unwrap_for_stacking(model):
    """
    If a base learner is calibrated, return the raw estimator.
    (StackingClassifier expects estimators with predict_proba/predict.)
    """
    try:
        if isinstance(model, CalibratedClassifierCV):
            return getattr(model, "estimator", getattr(model, "base_estimator", model))
        return model
    except Exception:
        return model


# ─────────────────────────────────────────────────────────────────────────────
# Calibration helpers
# ─────────────────────────────────────────────────────────────────────────────
def calibrate_prefit(estimator, X_cal, y_cal, method: str = "sigmoid") -> CalibratedClassifierCV:
    """
    Wrap a *pre-fitted* estimator with Platt ('sigmoid') or 'isotonic' calibration,
    trained on a dedicated calibration set.
    """
    try:
        cal = CalibratedClassifierCV(estimator=estimator, method=method, cv="prefit")
    except TypeError:
        cal = CalibratedClassifierCV(base_estimator=estimator, method=method, cv="prefit")
    cal.fit(X_cal, y_cal)
    return cal


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
    n_bins: int = 15,
    strategy: str = "quantile",
    ax=None,
    title: str = "Calibration (Reliability) Curve",
    annotate_metrics: bool = True,
):
    """
    Generic reliability curve helper.
    """
    assert len(proba_list) == len(clf_names), "Names must match prob arrays"

    if ax is None:
        _, ax = plt.subplots(figsize=(5, 5))

    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1, label="Perfect")

    curves_labels = []
    for p, name in zip(proba_list, clf_names):
        p = np.clip(np.asarray(p, dtype=float), 1e-9, 1 - 1e-9)
        frac_pos, mean_pred = calibration_curve(y_true, p, n_bins=n_bins, strategy=strategy)
        ax.plot(mean_pred, frac_pos, marker="o", linewidth=1, label=name)

        if annotate_metrics:
            try:
                bs = brier_score_loss(y_true, p)
                ll = log_loss(y_true, p)
                curves_labels.append((name, bs, ll))
            except Exception:
                pass

    ax.set_xlabel("Mean predicted probability", fontsize=16)
    ax.set_ylabel("Fraction of positives", fontsize=16)
    ax.set_title(title, fontsize=16)

    ax.tick_params(axis="both", labelsize=14)
    ax.legend(loc="best", fontsize=14)

    ax.grid(True, linestyle=":", linewidth=0.5)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    if annotate_metrics and curves_labels:
        print("\nCalibration metrics:")
        for name, bs, ll in curves_labels:
            print(f"  {name:>13s}  |  Brier: {bs:.4f}   LogLoss: {ll:.4f}")

    return ax


def plot_platt_calibration_on_test(
    y_true_bin: np.ndarray,
    p_raw: np.ndarray,
    p_platt: Optional[np.ndarray],
    *,
    title: str,
    out_png_dev: Optional[Union[str, Path]] = None,
    out_png_rel: Optional[Union[str, Path]] = None,
    n_bins: int = 15,
) -> Tuple[float, float, float, float, bool]:
    """
    Reliability diagram evaluated on TEST with strict plot order:
      1) Ideal
      2) RAW
      3) Platt (if available) on top

    Legend includes TEST LogLoss + Brier for RAW and Platt.

    Returns:
      (ll_raw, br_raw, ll_platt, br_platt, platt_available)
    """
    y_true_bin = np.asarray(y_true_bin).astype(int)

    p_raw = np.clip(np.asarray(p_raw, dtype=float), 1e-9, 1 - 1e-9)
    p_pl = None if p_platt is None else np.clip(np.asarray(p_platt, dtype=float), 1e-9, 1 - 1e-9)

    ll_raw = float(log_loss(y_true_bin, p_raw))
    br_raw = float(brier_score_loss(y_true_bin, p_raw))

    if p_pl is not None:
        ll_pl = float(log_loss(y_true_bin, p_pl))
        br_pl = float(brier_score_loss(y_true_bin, p_pl))
        pl_avail = True
    else:
        ll_pl = float("nan")
        br_pl = float("nan")
        pl_avail = False

    fig = plt.figure(figsize=(5, 5))
    ax = plt.gca()

    # Ideal
    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1.0, color="gray", label="Ideal")

    # RAW
    frac_pos_raw, mean_pred_raw = calibration_curve(y_true_bin, p_raw, n_bins=n_bins, strategy="quantile")
    ax.plot(
        mean_pred_raw, frac_pos_raw,
        marker="o", linewidth=1.5,
        label=f"RAW (LogLoss={ll_raw:.3f}, Brier={br_raw:.3f})"
    )

    # PLATT on top
    if pl_avail:
        frac_pos_pl, mean_pred_pl = calibration_curve(y_true_bin, p_pl, n_bins=n_bins, strategy="quantile")
        ax.plot(
            mean_pred_pl, frac_pos_pl,
            marker="o", linewidth=1.5,
            label=f"Platt (LogLoss={ll_pl:.3f}, Brier={br_pl:.3f})"
        )

    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives")
    ax.set_title(title)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc="lower right")
    ax.grid(True, linestyle=":", linewidth=0.5)
    plt.tight_layout()

    def _save(fig_, out_):
        if out_ is None:
            return
        out_ = Path(out_)
        out_.parent.mkdir(parents=True, exist_ok=True)
        fig_.savefig(out_, dpi=300, bbox_inches="tight")

    _save(fig, out_png_dev)
    _save(fig, out_png_rel)
    plt.close(fig)

    return ll_raw, br_raw, ll_pl, br_pl, pl_avail


def export_platt_metrics_csv(
    rows: List[Dict[str, Any]],
    *,
    out_dev: Optional[Union[str, Path]] = None,
    out_rel: Optional[Union[str, Path]] = None,
    filename: str = "Single_classes_metrics_pre_and_post_platt_calibration.csv",
) -> Optional[pd.DataFrame]:
    """
    Export per-class TEST logloss/brier metrics (RAW vs Platt) as a CSV.
    """
    if not rows:
        return None
    df = pd.DataFrame(rows).sort_values("class_name")

    if out_dev is not None:
        p = Path(out_dev) / filename
        p.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(p, index=False)

    if out_rel is not None:
        p = Path(out_rel) / filename
        p.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(p, index=False)

    return df


# ─────────────────────────────────────────────────────────────────────────────
# XGB POSITIVE-CLASS MARGIN CONTRIBUTIONS (ROBUST DIRECTION)
# ─────────────────────────────────────────────────────────────────────────────
def xgb_margin_contribution_summary(
    xgb_estimator,
    X: pd.DataFrame,
    *,
    sanity_check: bool = True,
) -> pd.DataFrame:
    """
    Robust per-feature contribution summaries for a fitted XGBoost *binary* model.

    Uses XGBoost Booster pred_contribs on the *margin* (log-odds) output, which corresponds
    to the positive class (label=1) for binary:logistic models.
    """
    base = unwrap_estimator(xgb_estimator)
    if not hasattr(base, "get_booster"):
        raise TypeError(f"Expected an XGBoost sklearn estimator with get_booster(); got: {type(base)}")

    booster = base.get_booster()

    import xgboost as xgb
    dmat = xgb.DMatrix(X.to_numpy(dtype=float), feature_names=list(X.columns))

    contrib = booster.predict(dmat, pred_contribs=True, output_margin=True)
    contrib = np.asarray(contrib)

    if contrib.ndim != 2 or contrib.shape[1] != (X.shape[1] + 1):
        raise ValueError(f"Unexpected contrib shape: {contrib.shape}; expected (n, p+1)")

    if sanity_check:
        try:
            margin_from_contrib = contrib.sum(axis=1)
            margin_direct = booster.predict(dmat, output_margin=True)
            max_diff = float(np.max(np.abs(margin_from_contrib - margin_direct)))
            if max_diff > 1e-4:
                warnings.warn(f"[XGB contrib check] contrib sum != margin (max abs diff={max_diff:.3e})")
        except Exception:
            pass

    contrib_feat = contrib[:, :-1]
    Xv = X.to_numpy(dtype=float)

    mean_abs = np.mean(np.abs(contrib_feat), axis=0)
    mean_signed = np.mean(contrib_feat, axis=0)
    frac_pos = np.mean(contrib_feat > 0, axis=0)

    corrs: List[float] = []
    for j in range(contrib_feat.shape[1]):
        xj = Xv[:, j]
        sj = contrib_feat[:, j]
        if np.std(xj) == 0 or np.std(sj) == 0:
            corrs.append(np.nan)
        else:
            corrs.append(float(np.corrcoef(xj, sj)[0, 1]))

    df = pd.DataFrame({
        "feature": X.columns.astype(str),
        "mean_abs_contrib": mean_abs.astype(float),
        "mean_signed_contrib": mean_signed.astype(float),
        "frac_samples_contrib_positive": frac_pos.astype(float),
        "corr_feature_value_vs_contrib": np.array(corrs, dtype=float),
    })

    df["direction_hint"] = np.where(
        df["mean_signed_contrib"] > 0,
        "higher_standardized_value_pushes_TOWARD_class",
        np.where(df["mean_signed_contrib"] < 0, "higher_standardized_value_pushes_AWAY_from_class", "neutral")
    )
    return df


# ─────────────────────────────────────────────────────────────────────────────
# TRUE SHAP (shap.TreeExplainer) VALUES + BEESWARM PLOTTING
# ─────────────────────────────────────────────────────────────────────────────
def xgb_shap_values(
    xgb_estimator,
    X: pd.DataFrame,
    *,
    class_index: int = 1,
) -> Tuple[np.ndarray, Any]:
    """
    Compute true SHAP values using shap.TreeExplainer for an XGBoost binary model.
    """
    try:
        import shap
    except ImportError as e:
        raise ImportError("shap is not installed. Install with: pip install shap") from e

    base = unwrap_estimator(xgb_estimator)
    if not hasattr(base, "get_booster"):
        raise TypeError(f"Expected an XGBoost sklearn estimator with get_booster(); got: {type(base)}")

    explainer = shap.TreeExplainer(base)
    shap_vals = explainer.shap_values(X)

    if isinstance(shap_vals, list):
        if class_index < 0 or class_index >= len(shap_vals):
            raise ValueError(f"class_index={class_index} out of range for SHAP list of length {len(shap_vals)}")
        shap_vals = shap_vals[class_index]

    return np.asarray(shap_vals), explainer.expected_value


def xgb_shap_mean_abs_and_corr(
    xgb_estimator,
    X: pd.DataFrame,
    *,
    class_index: int = 1,
) -> pd.DataFrame:
    """
    Compute mean(|SHAP|) + corr(feature_value, SHAP) per feature.
    """
    sv, _ = xgb_shap_values(xgb_estimator, X, class_index=class_index)
    if sv.ndim != 2 or sv.shape[1] != X.shape[1]:
        raise ValueError(f"Unexpected SHAP shape {sv.shape} for X shape {X.shape}")

    Xv = X.to_numpy(dtype=float)
    mean_abs = np.mean(np.abs(sv), axis=0)

    corrs: List[float] = []
    for j in range(sv.shape[1]):
        xj = Xv[:, j]
        sj = sv[:, j]
        if np.std(xj) == 0 or np.std(sj) == 0:
            corrs.append(np.nan)
        else:
            corrs.append(float(np.corrcoef(xj, sj)[0, 1]))

    return pd.DataFrame({
        "feature": X.columns.astype(str),
        "mean_abs_shap": mean_abs.astype(float),
        "corr_feature_value_vs_shap": np.asarray(corrs, dtype=float),
    })


def plot_xgb_shap_beeswarm(
    shap_values: np.ndarray,
    X: pd.DataFrame,
    *,
    title: str = "SHAP Beeswarm",
    max_display: int = 30,
    out_path: Union[str, Path, None] = None,
    figsize: Tuple[float, float] = (6, 7),
) -> None:
    """
    Create a SHAP beeswarm (summary dot) plot, and save if out_path provided.
    """
    try:
        import shap
    except ImportError as e:
        raise ImportError("shap is not installed. Install with: pip install shap") from e

    shap.summary_plot(
        shap_values,
        X,
        plot_type="dot",
        max_display=max_display,
        show=False,
    )
    fig = plt.gcf()
    fig.set_size_inches(figsize[0], figsize[1])

    plt.title(title, fontsize=16)

    ax = plt.gca()
    ax.tick_params(axis="both", labelsize=14)

    plt.tight_layout()

    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Palette + embedding plotting helpers (PATCHED)
# ─────────────────────────────────────────────────────────────────────────────
def _norm_label(s: str) -> str:
    return re.sub(r"\s+", " ", str(s).strip())


def palette_color(
    custom_palette: Dict[str, str],
    celltype: str,
    cls_safe: str,
    default: str = "#1f77b4",
) -> str:
    """
    Robust palette lookup that matches:
      - exact raw label
      - normalized raw label (strip/collapse spaces)
      - safe_name variants (underscores <-> spaces)
      - case-insensitive matches on all of the above
    """
    if not custom_palette:
        return default

    pal_norm = {_norm_label(k): v for k, v in custom_palette.items()}
    pal_lower = {_norm_label(k).lower(): v for k, v in custom_palette.items()}

    raw = str(celltype)
    raw_n = _norm_label(raw)
    safe = str(cls_safe)
    safe_n = _norm_label(safe)

    candidates = [
        raw, raw_n,
        safe, safe_n,
        raw_n.replace("_", " "),
        safe_n.replace("_", " "),
        raw_n.replace(" ", "_"),
        safe_n.replace(" ", "_"),
    ]

    for k in candidates:
        if k in custom_palette:
            return custom_palette[k]
        if k in pal_norm:
            return pal_norm[k]

    for k in candidates:
        kl = _norm_label(k).lower()
        if kl in pal_lower:
            return pal_lower[kl]

    return default


def validate_mpl_color(c: str, fallback: str = "#1f77b4") -> str:
    try:
        mcolors.to_rgba(c)
        return c
    except Exception:
        return fallback


def _get_embedding_coords_for_barcodes(
    barcodes: np.ndarray,
    *,
    adata=None,
    df: Optional[pd.DataFrame] = None,
    embedding_source: str = "adata_obsm",   # "adata_obsm" | "adata_obs" | "train_df"
    obsm_key: str = "X_umap",
    obs_x: str = "UMAP_1",
    obs_y: str = "UMAP_2",
    df_x: str = "UMAP_1",
    df_y: str = "UMAP_2",
    require_all: bool = True,
) -> Tuple[Optional[np.ndarray], bool]:
    """
    Return (xy, ok) with xy aligned to the given barcodes.

    embedding_source options:
      - "adata_obsm": use adata.obsm[obsm_key]
      - "adata_obs":  use adata.obs[[obs_x, obs_y]]
      - "train_df":   use df[[df_x, df_y]]
    """
    idx = pd.Index(barcodes)

    def _finish(out: pd.DataFrame) -> Tuple[Optional[np.ndarray], bool]:
        if require_all:
            if out.notna().all(axis=1).all():
                return out.to_numpy(dtype=float), True
            return None, False
        keep = out.notna().all(axis=1)
        if keep.sum() == 0:
            return None, False
        return out.loc[keep].to_numpy(dtype=float), True

    # adata.obsm
    if embedding_source == "adata_obsm":
        if adata is None or not hasattr(adata, "obsm"):
            return None, False
        if obsm_key not in adata.obsm:
            warnings.warn(
                f"Embedding key '{obsm_key}' not found in adata.obsm. "
                f"Available keys: {list(getattr(adata, 'obsm', {}).keys())}"
            )
            return None, False
        um = np.asarray(adata.obsm[obsm_key])
        if um.ndim != 2 or um.shape[1] < 2:
            warnings.warn(f"adata.obsm['{obsm_key}'] has shape {um.shape}; expected (n, >=2).")
            return None, False
        um_df = pd.DataFrame(
            um[:, :2],
            index=pd.Index(getattr(adata, "obs_names", getattr(adata, "obs", pd.DataFrame()).index)),
            columns=["UMAP1", "UMAP2"],
        )
        return _finish(um_df.reindex(idx))

    # adata.obs
    if embedding_source == "adata_obs":
        if adata is None or not hasattr(adata, "obs"):
            return None, False
        if obs_x not in adata.obs.columns or obs_y not in adata.obs.columns:
            warnings.warn(
                f"Embedding columns '{obs_x}', '{obs_y}' not found in adata.obs. "
                f"Available: {list(getattr(adata, 'obs', pd.DataFrame()).columns)[:20]}..."
            )
            return None, False
        um_df = adata.obs[[obs_x, obs_y]].copy()
        um_df.columns = ["UMAP1", "UMAP2"]
        um_df.index = pd.Index(getattr(adata, "obs_names", um_df.index))
        return _finish(um_df.reindex(idx))

    # train df
    if embedding_source == "train_df":
        if df is None:
            return None, False
        if df_x not in df.columns or df_y not in df.columns:
            warnings.warn(
                f"Embedding columns '{df_x}', '{df_y}' not found in train_df. "
                f"Available: {list(df.columns)[:20]}..."
            )
            return None, False
        out = df.reindex(idx)[[df_x, df_y]].copy()
        out.columns = ["UMAP1", "UMAP2"]
        return _finish(out)

    warnings.warn(f"Unknown embedding_source='{embedding_source}'.")
    return None, False


def save_class_train_umap_pngs(
    *,
    celltype: str,
    cls_safe: str,
    barcodes: np.ndarray,
    y_bin: np.ndarray,
    custom_palette: Dict[str, str],
    out_dir_dev: Optional[Union[str, Path]] = None,
    out_dir_rel: Optional[Union[str, Path]] = None,
    adata_train=None,
    train_df: Optional[pd.DataFrame] = None,
    # embedding controls
    embedding_source: str = "adata_obsm",
    obsm_key: str = "X_umap",
    obs_x: str = "UMAP_1",
    obs_y: str = "UMAP_2",
    df_x: str = "UMAP_1",
    df_y: str = "UMAP_2",
    require_all: bool = True,
    # styling
    neg_color: str = "#A3A3A3",
    point_size: float = 10.0,
    outline: Tuple[float, float] = (0.0, 0.0),  # (size_mult, alpha) e.g. (1.6, 0.12)
    debug: bool = False,
) -> None:
    """
    Saves:
      - <cls_safe>_Class_Train_data.png
      - <cls_safe>_Class_Train_data_legend.png

    Plot:
      - minimal embedding, no axis
      - positives colored by custom_palette lookup (robust)
      - negatives colored by neg_color
      - optional black outline behind positives (outline=(size_mult, alpha))
    """
    xy, ok = _get_embedding_coords_for_barcodes(
        barcodes,
        adata=adata_train,
        df=train_df,
        embedding_source=embedding_source,
        obsm_key=obsm_key,
        obs_x=obs_x,
        obs_y=obs_y,
        df_x=df_x,
        df_y=df_y,
        require_all=require_all,
    )
    if not ok or xy is None:
        warnings.warn(f"No embedding coords for {cls_safe}; skipping Class_Train_data plots.")
        return

    y_bin = np.asarray(y_bin).astype(int)
    if xy.shape[0] != y_bin.shape[0]:
        warnings.warn(f"Embedding rows != y rows for {cls_safe}; skipping.")
        return

    pos_mask = (y_bin == 1)
    neg_mask = ~pos_mask
    n_pos = int(pos_mask.sum())
    n_neg = int(neg_mask.sum())

    # PATCH: use the correct helper names (palette_color / validate_mpl_color)
    pos_color = validate_mpl_color(
        palette_color(custom_palette, celltype=celltype, cls_safe=cls_safe, default="#1f77b4"),
        fallback="#1f77b4",
    )
    neg_color = validate_mpl_color(neg_color, fallback="#A3A3A3")

    if debug:
        print(
            f"[EMBED] celltype='{celltype}' cls_safe='{cls_safe}' "
            f"source='{embedding_source}' obsm_key='{obsm_key}' "
            f"pos={n_pos} neg={n_neg} pos_color='{pos_color}' neg_color='{neg_color}'"
        )
        if pos_color == "#1f77b4":
            warnings.warn(
                f"[Palette] No palette match for celltype='{celltype}' (cls_safe='{cls_safe}'). Using default."
            )

    def _save_fig(fig_, out_dir_, fname_):
        if out_dir_ is None:
            return
        out_dir_ = Path(out_dir_)
        out_dir_.mkdir(parents=True, exist_ok=True)
        fig_.savefig(out_dir_ / fname_, dpi=300, bbox_inches="tight", pad_inches=0.02)

    # --- main plot ---
    fig = plt.figure(figsize=(5.5, 4.5))
    ax = plt.gca()

    # Negatives first
    if n_neg > 0:
        ax.scatter(
            xy[neg_mask, 0], xy[neg_mask, 1],
            s=point_size,
            c=neg_color,
            linewidths=0,
            alpha=1.0,
            rasterized=True,
            zorder=1,
        )

    # Optional outline behind positives
    size_mult, out_alpha = float(outline[0]), float(outline[1])
    if n_pos > 0 and size_mult > 0 and out_alpha > 0:
        ax.scatter(
            xy[pos_mask, 0], xy[pos_mask, 1],
            s=point_size * size_mult,
            c="black",
            linewidths=0,
            alpha=out_alpha,
            rasterized=True,
            zorder=2,
        )

    # Positives on top
    if n_pos > 0:
        ax.scatter(
            xy[pos_mask, 0], xy[pos_mask, 1],
            s=point_size,
            c=pos_color,
            linewidths=0,
            alpha=1.0,
            rasterized=True,
            zorder=3,
        )

    ax.set_axis_off()
    ax.set_aspect("equal", adjustable="datalim")
    plt.tight_layout(pad=0.0)

    _save_fig(fig, out_dir_dev, f"{cls_safe}_Class_Train_data.png")
    _save_fig(fig, out_dir_rel, f"{cls_safe}_Class_Train_data.png")
    plt.close(fig)

    # --- legend ---
    handles = [
        Line2D(
            [0], [0],
            marker="o",
            linestyle="",
            markersize=10,
            markerfacecolor=pos_color,
            markeredgecolor="black",
            markeredgewidth=0.6,
            label=f"{celltype} (N={n_pos})",
        ),
        Line2D(
            [0], [0],
            marker="o",
            linestyle="",
            markersize=10,
            markerfacecolor=neg_color,
            markeredgecolor="black",
            markeredgewidth=0.6,
            label=f"Rest (N={n_neg})",
        ),
    ]

    fig = plt.figure(figsize=(4.0, 2.3))
    ax = plt.gca()
    ax.axis("off")
    ax.legend(
        handles=handles,
        loc="center",
        frameon=False,
        ncol=1,
        fontsize=12,
        handletextpad=0.8,
        borderaxespad=0.0,
    )
    plt.tight_layout(pad=0.0)

    _save_fig(fig, out_dir_dev, f"{cls_safe}_Class_Train_data_legend.png")
    _save_fig(fig, out_dir_rel, f"{cls_safe}_Class_Train_data_legend.png")
    plt.close(fig)

# ─────────────────────────────────────────────────────────────────────────────
# Training functions
# ─────────────────────────────────────────────────────────────────────────────
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


def train_KNN(
    X_train, y_train, *,
    cv=CV, num_cores: int = -1, name_target_subclass: str,
):
    print(f"\n[KNN] training → {name_target_subclass}")
    knn_pipe = Pipeline([("knn", KNeighborsClassifier())])
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


def train_MLP(
    X_train, y_train, *,
    cv=CV, num_cores: int = -1, name_target_subclass: str,
):
    print(f"\n[MLP] training → {name_target_subclass}")
    mlp_pipe = Pipeline([("mlp", MLPClassifier())])
    param_grid = {
        "mlp__hidden_layer_sizes": [(50, 50, 50), (50, 100, 50), (100,)],
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


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation helper exposed to notebooks / scripts
# ─────────────────────────────────────────────────────────────────────────────
def evaluate_classifier(
    model,
    X_test,
    y_test,
    *,
    average: str | None = None,
    plot_cm: bool = False,
    cm_path: Union[str, Path, None] = None,
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


def append_metrics_csv(
    metrics_df: pd.DataFrame,
    csv_path: Union[str, Path],
    *,
    drop_report: bool = True,
) -> None:
    csv_path = Path(csv_path)
    if drop_report and "report" in metrics_df.columns:
        metrics_df = metrics_df.drop(columns=["report"])

    header = not csv_path.exists()
    metrics_df.to_csv(csv_path, mode="a", header=header, index=False)
