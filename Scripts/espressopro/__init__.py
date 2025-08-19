# -*- coding: utf-8 -*-
"""EspressoPro — modular cell type annotation pipeline."""

from __future__ import annotations

# Core
from .core import (
    load_models,
    get_package_data_path,
    get_default_models_path,
    get_default_data_path,
    download_models,
    ensure_models_available,
)

# Prediction / scoring
from .prediction import (
    stack_prediction,
    generate_predictions,
    audit_feature_overlap,
    add_best_localised_tracks,
    add_consensus_weighted_tracks,
)

# Annotation
from .annotation import (
    voting_annotator,
    Broad_Annotation,
    Simplified_Annotation,
    Detailed_Annotation,
    annotate_data,
    Normalise_protein_data,
    Scale_protein_data,
    mark_small_clusters,
    mark_mixed_clusters,
    refine_labels_by_knn_consensus,
    clear_annotation,
    score_mixed_clusters,
)

# Optional: annotate_counts_matrix (only if defined)
try:
    from .annotation import annotate_counts_matrix  # type: ignore
    _HAS_ANNOTATE_COUNTS = True
except Exception:
    _HAS_ANNOTATE_COUNTS = False

# MissionBio
from .missionbio import (
    suggest_cluster_celltype_identity,
    print_cluster_suggestions,
    visualize_cluster_celltype_frequencies,
)

# Markers
from .markers import add_mast_annotation, add_signature_annotation

# Constants
from .constants import (
    SIMPLIFIED_CLASSES,
    DETAILED_CLASSES,
    SIMPLIFIED_PARENT_MAP,
    DETAILED_PARENT_MAP,
)

__version__ = "1.0.0"

__all__ = [
    # Core
    "load_models",
    "get_package_data_path",
    "get_default_models_path",
    "get_default_data_path",
    "download_models",
    "ensure_models_available",
    # Prediction / scoring
    "stack_prediction",
    "generate_predictions",
    "audit_feature_overlap",
    "add_best_localised_tracks",
    "add_consensus_weighted_tracks",
    # Annotation
    "voting_annotator",
    "Broad_Annotation",
    "Simplified_Annotation",
    "Detailed_Annotation",
    "annotate_data",
    "Normalise_protein_data",
    "Scale_protein_data",
    "mark_small_clusters",
    "refine_labels_by_knn_consensus",
    "clear_annotation",
    "score_mixed_clusters",
    "mark_mixed_clusters",
    # MissionBio
    "suggest_cluster_celltype_identity",
    "print_cluster_suggestions",
    "visualize_cluster_celltype_frequencies",
    # Markers
    "add_mast_annotation",
    "add_signature_annotation",
    # Constants
    "SIMPLIFIED_CLASSES",
    "DETAILED_CLASSES",
    "SIMPLIFIED_PARENT_MAP",
    "DETAILED_PARENT_MAP",
]

if _HAS_ANNOTATE_COUNTS:
    __all__.append("annotate_counts_matrix")
