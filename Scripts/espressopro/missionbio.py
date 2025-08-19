# -*- coding: utf-8 -*-
"""MissionBio-specific integration functions."""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def suggest_cluster_celltype_identity(
    sample,
    annotation,
    min_frequency_threshold: float = 0.3,
    *,
    rewrite: bool = False,
    out_key: str = "annotated_clusters",
    use_mixed_below_threshold: bool = True,
):
    """
    Suggest a cell-type identity for each cluster from per-cell labels.

    Parameters
    ----------
    sample : missionbio.mosaic.Sample
        Sample with a protein assay; must support sample.protein.get_labels() and row_attrs.
    annotation : array-like or str
        Per-cell labels or the name of a row_attrs field.
    min_frequency_threshold : float
        Minimum within-cluster frequency for treating the top label as dominant.
    rewrite : bool
        If True, write per-cell labels to sample.protein.row_attrs[out_key].
    out_key : str
        Row attribute name used when rewrite=True.
    use_mixed_below_threshold : bool
        If True and no label meets the threshold, use "Mixed"; if False, use the top label anyway.

    Returns
    -------
    dict
        Mapping: cluster -> {"suggested_celltype","frequency","confidence","total_cells"}.
    pd.DataFrame
        Cluster × celltype frequency table.
    (optional) missionbio.mosaic.Sample
        Returned only when rewrite=True (same object, modified in place).
    """
    # cluster labels from the Sample
    if not hasattr(sample, "protein") or not hasattr(sample.protein, "get_labels"):
        raise TypeError("Expected a MissionBio Sample with protein.get_labels().")
    cluster_labels = sample.protein.get_labels()

    # resolve per-cell annotation
    if isinstance(annotation, str):
        if not hasattr(sample.protein, "row_attrs") or annotation not in sample.protein.row_attrs:
            raise KeyError(f"'{annotation}' not found in sample.protein.row_attrs")
        ann_vec = np.asarray(sample.protein.row_attrs[annotation])
    else:
        ann_vec = np.asarray(annotation)

    if len(ann_vec) != len(cluster_labels):
        raise ValueError(
            f"annotation length ({len(ann_vec)}) != number of cells ({len(cluster_labels)})"
        )

    df = pd.DataFrame(
        {"cluster": cluster_labels, "celltype_detailed_refined": ann_vec.astype(object)}
    )

    # counts and frequencies
    cluster_celltype_counts = (
        df.groupby(["cluster", "celltype_detailed_refined"]).size().reset_index(name="count")
    )
    cluster_totals = df.groupby("cluster").size().reset_index(name="total")
    frequency_df = cluster_celltype_counts.merge(cluster_totals, on="cluster", how="left")
    frequency_df["frequency"] = frequency_df["count"] / frequency_df["total"].replace(0, np.nan)

    # pivot: cluster × celltype
    pivot_df = (
        frequency_df.pivot(index="cluster", columns="celltype_detailed_refined", values="frequency")
        .fillna(0.0)
    )

    # suggestions
    cluster_suggestions = {}
    for cluster in pivot_df.index:
        row = pivot_df.loc[cluster]
        max_frequency = float(row.max())
        dominant_celltype = str(row.idxmax())

        suggested = (
            dominant_celltype
            if max_frequency >= min_frequency_threshold
            else ("Mixed" if use_mixed_below_threshold else dominant_celltype)
        )
        confidence = "High" if max_frequency >= 0.7 else ("Medium" if max_frequency >= min_frequency_threshold else "Low")
        total_cells = int(cluster_totals.loc[cluster_totals["cluster"] == cluster, "total"].iloc[0])

        cluster_suggestions[cluster] = {
            "suggested_celltype": suggested,
            "frequency": max_frequency,
            "confidence": confidence,
            "total_cells": total_cells,
        }

    if not rewrite:
        return cluster_suggestions, pivot_df

    # rewrite per-cell labels from cluster suggestion
    cluster_to_label = {cl: info["suggested_celltype"] for cl, info in cluster_suggestions.items()}
    annotated = df["cluster"].map(cluster_to_label).to_numpy(dtype=object)
    sample.protein.row_attrs[out_key] = annotated
    return sample


def print_cluster_suggestions(cluster_suggestions: dict) -> None:
    """Pretty-print the mapping produced by suggest_cluster_celltype_identity()."""
    print("Cluster Cell Type Suggestions:")
    print("=" * 60)
    for cluster, info in cluster_suggestions.items():
        print(f"Cluster {cluster}:")
        print(f"  Suggested: {info['suggested_celltype']}")
        print(f"  Frequency: {info['frequency']:.2%}")
        print(f"  Confidence: {info['confidence']}")
        print(f"  Total cells: {info['total_cells']}")
        print()


def visualize_cluster_celltype_frequencies(pivot_df: pd.DataFrame, figsize=(12, 8)) -> None:
    """Heatmap of cell-type frequencies per cluster."""
    if pivot_df.empty:
        print("[visualize] Empty pivot table; nothing to plot.")
        return

    plt.figure(figsize=figsize)
    sns.heatmap(
        pivot_df,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        cbar_kws={"label": "Frequency"},
        linewidths=0.5,
    )
    plt.title("Cell Type Frequency per Cluster", fontsize=14, fontweight="bold")
    plt.xlabel("Cell Type (Detailed)", fontsize=12)
    plt.ylabel("Cluster", fontsize=12)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()
