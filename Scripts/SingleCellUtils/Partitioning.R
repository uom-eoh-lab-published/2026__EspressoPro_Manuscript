#' Partitioning Utility Functions
#' 
#' Helper functions for train/test/calibration splits with stratified
#' leverage-based sketching and one-vs-rest barcode generation.
#' 
#' Fixed version that rebuilds meta.features from actual data rownames
#' and is compatible with Seurat v5.

library(tidyverse)
library(Seurat)
library(ComplexHeatmap)
library(viridis)

# Custom color palette matching Python harmonisation
CUSTOM_PALETTE <- c(
  'B Memory' = "#68D827", 'B Naive' = '#1C511D', 'CD14 Mono' = "#D27CE3",
  'CD16 Mono' = "#8D43CD", 'CD4 T Memory' = "#C1AF93", 'CD4 T Naive' = "#C99546",
  'CD8 T Memory' = "#6B3317", 'CD8 T Naive' = "#4D382E", 'ErP' = "#D1235A",
  'Erythroblast' = "#F30A1A", 'GMP' = "#C5E4FF", 'HSC_MPP' = '#0079ea',
  'Immature B' = "#91FF7B", 'LMPP' = "#17BECF", 'MAIT' = "#BCBD22",
  'Myeloid progenitor' = "#AEC7E8", 'NK CD56 bright' = "#F3AC1F",
  'NK CD56 dim' = "#FBEF0D", 'magma' = "#9DC012", 'Pro-B' = "#66BB6A",
  'Small' = "#292929", 'cDC1' = "#76A7CB", 'cDC2' = "#16D2E3", 'gdT' = "#EDB416",
  'pDC' = "#69FFCB", 'CD4 CTL' = "#D7D2CB", 'MEP' = "#E364B0", 'Pre-B' = "#2DBD67",
  'Pre-Pro-B' = '#92AC8E', 'EoBaMaP' = "#728245", 'MkP' = "#69424D",
  'Stroma' = "#727272", 'Macrophage' = "#5F4761", 'ILC' = "#F7CF94", 'dnT' = "#504423",
  'HSC' = '#0079ea', 'MPP' = '#0079ea',  # Aliases for HSC_MPP
  'Gamma delta T' = "#EDB416",  # Alias for gdT
  'Double negative T' = "#504423"  # Alias for dnT
)

#' Rebuild meta.features from actual data features
#'
#' @param obj Seurat object
#' @return Cleaned Seurat object with correct meta.features
clean_seurat_object <- function(obj) {
  # Remove sketch assay if present
  if ("sketch" %in% Assays(obj)) {
    obj[["sketch"]] <- NULL
  }
  
  # Fix meta.features for all assays by rebuilding from actual data
  for (assay_name in Assays(obj)) {
    assay <- obj[[assay_name]]
    
    # Get actual features from data slot
    features_in_data <- rownames(assay@data)
    
    # Rebuild meta.features to match exactly
    new_meta_features <- data.frame(
      name = features_in_data,
      row.names = features_in_data,
      stringsAsFactors = FALSE
    )
    
    # Preserve any additional columns if they exist for these features
    if (ncol(assay@meta.features) > 0 && nrow(assay@meta.features) > 0) {
      existing_cols <- setdiff(colnames(assay@meta.features), "name")
      for (col in existing_cols) {
        if (col %in% colnames(assay@meta.features)) {
          # Only keep rows that exist in actual data
          valid_rows <- intersect(rownames(assay@meta.features), features_in_data)
          if (length(valid_rows) > 0) {
            new_meta_features[[col]] <- NA
            new_meta_features[valid_rows, col] <- assay@meta.features[valid_rows, col]
          }
        }
      }
    }
    
    assay@meta.features <- new_meta_features
    
    # Also ensure counts matches if present
    if (nrow(assay@counts) > 0) {
      features_in_counts <- rownames(assay@counts)
      if (!identical(features_in_counts, features_in_data)) {
        warning(sprintf("Counts features don't match data features in assay %s", assay_name))
        # Keep only features present in data
        common_features <- intersect(features_in_counts, features_in_data)
        if (length(common_features) > 0) {
          assay@counts <- assay@counts[common_features, , drop = FALSE]
        }
      }
    }
    
    # Update scale.data if present
    if (nrow(assay@scale.data) > 0) {
      features_in_scale <- rownames(assay@scale.data)
      valid_scale_features <- intersect(features_in_scale, features_in_data)
      if (length(valid_scale_features) > 0) {
        assay@scale.data <- assay@scale.data[valid_scale_features, , drop = FALSE]
      } else {
        assay@scale.data <- matrix(nrow = 0, ncol = 0)
      }
    }
    
    obj[[assay_name]] <- assay
  }
  
  return(obj)
}

#' Safe subset that handles meta.features issues
#'
#' @param obj Seurat object
#' @param cells Cell barcodes to keep
#' @return Subsetted and cleaned Seurat object
safe_subset <- function(obj, cells) {
  # Clean before subset
  obj <- clean_seurat_object(obj)
  
  # Ensure cells exist
  cells <- intersect(cells, colnames(obj))
  if (length(cells) == 0) {
    stop("No valid cells to subset")
  }
  
  # Perform subset
  obj_sub <- tryCatch({
    subset(obj, cells = cells)
  }, error = function(e) {
    # If subset fails, try manual subsetting
    warning(sprintf("Standard subset failed, attempting manual subset: %s", e$message))
    
    # Manual subset for each assay
    for (assay_name in Assays(obj)) {
      assay <- obj[[assay_name]]
      
      # Subset data
      if (ncol(assay@data) > 0) {
        assay@data <- assay@data[, cells, drop = FALSE]
      }
      
      # Subset counts
      if (ncol(assay@counts) > 0) {
        assay@counts <- assay@counts[, cells, drop = FALSE]
      }
      
      # Subset scale.data if present
      if (ncol(assay@scale.data) > 0 && any(cells %in% colnames(assay@scale.data))) {
        assay@scale.data <- assay@scale.data[, cells[cells %in% colnames(assay@scale.data)], drop = FALSE]
      }
      
      obj[[assay_name]] <- assay
    }
    
    # Subset metadata
    obj@meta.data <- obj@meta.data[cells, , drop = FALSE]
    
    obj
  })
  
  # Clean after subset to rebuild meta.features
  obj_sub <- clean_seurat_object(obj_sub)
  
  return(obj_sub)
}

#' Calculate leverage scores manually for robust sampling
#'
#' @param mat Numeric matrix (features x cells)
#' @param n_sample Number of cells to sample
#' @param seed Random seed
#' @return Vector of selected cell indices
leverage_sample <- function(mat, n_sample, seed = 123) {
  set.seed(seed)
  
  n_cells <- ncol(mat)
  n_sample <- min(n_sample, n_cells)
  
  if (n_sample == n_cells) {
    return(seq_len(n_cells))
  }
  
  # Compute leverage scores via SVD
  tryCatch({
    if (n_cells <= nrow(mat)) {
      # Tall matrix: standard SVD
      svd_res <- svd(t(mat), nu = 0, nv = min(50, n_cells))
      leverage <- rowSums(svd_res$v^2)
    } else {
      # Wide matrix: use random projection to avoid memory issues
      n_proj <- min(100, nrow(mat))
      proj_mat <- matrix(rnorm(nrow(mat) * n_proj), nrow = nrow(mat))
      proj_mat <- proj_mat / sqrt(rowSums(proj_mat^2))
      projected <- t(mat) %*% proj_mat
      leverage <- rowSums(projected^2)
    }
    
    # Normalize to probabilities
    leverage <- pmax(leverage, 1e-10)  # Avoid zeros
    leverage <- leverage / sum(leverage)
    
    # Sample without replacement
    sample(n_cells, size = n_sample, prob = leverage, replace = FALSE)
    
  }, error = function(e) {
    warning(sprintf("Leverage calculation failed, using uniform sampling: %s", e$message))
    sample(n_cells, size = n_sample, replace = FALSE)
  })
}

#' Preprocess Seurat object: merge labels, filter low-count classes
#'
#' @param obj Seurat object
#' @param label_col Column name for cell type labels
#' @param merge_labels Character vector of labels to merge (e.g., c("HSC", "MPP", "LMPP"))
#' @param merge_to Target label name (default: "HSC_MPP")
#' @param min_cells Minimum cells required per class
#' @return Filtered Seurat object
preprocess_object <- function(obj, label_col, merge_labels = NULL, 
                              merge_to = "HSC_MPP", min_cells = 50) {
  # Clean object first
  obj <- clean_seurat_object(obj)
  
  # Extract and clean labels
  labels <- as.character(obj[[label_col]][, 1])
  labels[is.na(labels)] <- "NA"
  
  # Merge specified labels
  if (!is.null(merge_labels)) {
    labels[labels %in% merge_labels] <- merge_to
  }
  
  # Standardize naming conventions
  labels <- gsub("Gamma delta T", "gdT", labels, fixed = TRUE)
  labels <- gsub("Double negative T", "dnT", labels, fixed = TRUE)
  
  obj[[label_col]] <- factor(labels)
  
  # Filter by minimum cells
  class_counts <- table(obj[[label_col]])
  valid_classes <- names(class_counts[class_counts >= min_cells])
  
  removed_classes <- setdiff(names(class_counts), valid_classes)
  if (length(removed_classes) > 0) {
    message(sprintf("Removed %d classes with < %d cells: %s", 
                    length(removed_classes), min_cells, 
                    paste(removed_classes, collapse = ", ")))
  }
  
  message(sprintf("Retained %d/%d classes with >= %d cells", 
                  length(valid_classes), length(class_counts), min_cells))
  
  cells_to_keep <- WhichCells(obj, expression = !!sym(label_col) %in% valid_classes)
  obj_filtered <- safe_subset(obj, cells_to_keep)
  
  return(obj_filtered)
}

#' Stratified leverage-based sketching per label (robust version)
#'
#' @param obj Seurat object
#' @param label_col Column name for stratification
#' @param assay Assay name
#' @param prop_sketch Proportion to retain (0-1)
#' @param seed Random seed
#' @param use_seurat_sketch Use Seurat's SketchData (FALSE recommended)
#' @return List with train_cells and holdout_cells vectors
stratified_sketch <- function(obj, label_col, assay, prop_sketch = 0.80, 
                              seed = 123, use_seurat_sketch = FALSE) {
  # Clean object first
  obj <- clean_seurat_object(obj)
  DefaultAssay(obj) <- assay
  
  labels <- obj[[label_col]][, 1]
  labels[is.na(labels)] <- "NA"
  
  train_cells <- character()
  holdout_cells <- character()
  
  for (lbl in unique(labels)) {
    cells_lbl <- colnames(obj)[labels == lbl]
    n_cells <- length(cells_lbl)
    if (n_cells == 0) next
    
    # Subset using safe method
    obj_sub <- safe_subset(obj, cells_lbl)
    
    # Normalize
    obj_sub <- NormalizeData(obj_sub, normalization.method = "CLR", 
                             margin = 2, assay = assay, verbose = FALSE)
    obj_sub <- FindVariableFeatures(obj_sub, assay = assay, verbose = FALSE)
    
    # Calculate target sketch size
    n_sketch <- max(1, round(n_cells * prop_sketch))
    
    # Always use manual leverage-based sampling (most stable)
    var_features <- VariableFeatures(obj_sub)
    if (length(var_features) == 0) {
      var_features <- rownames(obj_sub)
    }
    
    data_mat <- GetAssayData(obj_sub, slot = "data", assay = assay)[var_features, , drop = FALSE]
    sampled_idx <- leverage_sample(as.matrix(data_mat), n_sketch, seed)
    train_lbl <- cells_lbl[sampled_idx]
    
    holdout_lbl <- setdiff(cells_lbl, train_lbl)
    
    train_cells <- c(train_cells, train_lbl)
    holdout_cells <- c(holdout_cells, holdout_lbl)
    
    message(sprintf("  %s: %d train, %d holdout", lbl, 
                    length(train_lbl), length(holdout_lbl)))
  }
  
  list(train = train_cells, holdout = holdout_cells)
}

#' Perform two-stage split: Train/Test then Train/Calibration
#'
#' @param obj Seurat object
#' @param label_col Label column
#' @param assay Assay name
#' @param train_prop Train proportion (first split, default 0.80)
#' @param cal_prop Calibration proportion (second split on training data, default 0.90)
#' @param seed Random seed
#' @param use_seurat_sketch Use Seurat's SketchData (default FALSE for stability)
#' @return List with train, cal, test Seurat objects
two_stage_split <- function(obj, label_col, assay, 
                            train_prop = 0.80, cal_prop = 0.90, 
                            seed = 123, use_seurat_sketch = FALSE) {
  message("\n--- Stage 1: Train/Test Split ---")
  
  # Clean object before starting
  obj <- clean_seurat_object(obj)
  
  # Stage 1: Train/Test
  split1 <- stratified_sketch(obj, label_col, assay, train_prop, seed, use_seurat_sketch)
  obj_train_full <- safe_subset(obj, split1$train)
  obj_test <- safe_subset(obj, split1$holdout)
  
  message("\n--- Stage 2: Train/Calibration Split ---")
  
  # Stage 2: Train/Calibration
  split2 <- stratified_sketch(obj_train_full, label_col, assay, cal_prop, seed, use_seurat_sketch)
  obj_train <- safe_subset(obj_train_full, split2$train)
  obj_cal <- safe_subset(obj_train_full, split2$holdout)
  
  message("\n--- Normalizing splits ---")
  
  # Normalize all splits with error handling
  obj_train <- tryCatch({
    NormalizeData(obj_train, normalization.method = "CLR", 
                  margin = 2, assay = assay, verbose = FALSE)
  }, error = function(e) {
    warning(sprintf("Normalization failed for train split: %s", e$message))
    obj_train
  })
  
  obj_cal <- tryCatch({
    NormalizeData(obj_cal, normalization.method = "CLR", 
                  margin = 2, assay = assay, verbose = FALSE)
  }, error = function(e) {
    warning(sprintf("Normalization failed for cal split: %s", e$message))
    obj_cal
  })
  
  obj_test <- tryCatch({
    NormalizeData(obj_test, normalization.method = "CLR", 
                  margin = 2, assay = assay, verbose = FALSE)
  }, error = function(e) {
    warning(sprintf("Normalization failed for test split: %s", e$message))
    obj_test
  })
  
  message(sprintf("\nFinal sizes - Train: %d, Cal: %d, Test: %d",
                  ncol(obj_train), ncol(obj_cal), ncol(obj_test)))
  
  list(train = obj_train, cal = obj_cal, test = obj_test)
}

#' Validate partition overlap (should be 0%)
#'
#' @param splits List with train, cal, test Seurat objects
#' @return Dataframe with overlap statistics
validate_partition_overlap <- function(splits) {
  train_cells <- colnames(splits$train)
  cal_cells <- colnames(splits$cal)
  test_cells <- colnames(splits$test)
  
  overlap_df <- tibble(
    Comparison = c("Train vs Cal", "Train vs Test", "Cal vs Test"),
    Overlap_N = c(
      length(intersect(train_cells, cal_cells)),
      length(intersect(train_cells, test_cells)),
      length(intersect(cal_cells, test_cells))
    ),
    Overlap_Pct = c(
      100 * length(intersect(train_cells, cal_cells)) / length(train_cells),
      100 * length(intersect(train_cells, test_cells)) / length(train_cells),
      100 * length(intersect(cal_cells, test_cells)) / length(cal_cells)
    ),
    Total_Cells = c(
      length(train_cells) + length(cal_cells),
      length(train_cells) + length(test_cells),
      length(cal_cells) + length(test_cells)
    )
  )
  
  # Check for overlaps
  if (any(overlap_df$Overlap_N > 0)) {
    warning("⚠️  Partition overlap detected!")
  } else {
    message("✓ No overlap between partitions (0%)")
  }
  
  return(overlap_df)
}

#' Generate one-vs-rest barcodes (balanced for training, unbalanced for test)
#'
#' @param obj Seurat object
#' @param label_col Label column
#' @param assay Assay name
#' @param downsample_threshold Not used (kept for compatibility)
#' @param is_test Logical: if TRUE, keeps all cells unbalanced (for test set)
#' @param seed Random seed
#' @param use_seurat_sketch Use Seurat's SketchData (default FALSE for stability)
#' @return Named list of dataframes with Positive/Negative cell pairs per class
generate_ovr_barcodes <- function(obj, label_col, assay, 
                                  downsample_threshold = 50, 
                                  is_test = FALSE, seed = 123,
                                  use_seurat_sketch = FALSE) {
  # Clean object first
  obj <- clean_seurat_object(obj)
  DefaultAssay(obj) <- assay
  
  labels <- obj[[label_col]][, 1]
  labels[is.na(labels)] <- "NA"
  
  barcodes_list <- list()
  
  for (lbl in unique(labels)) {
    one_ids <- colnames(obj)[labels == lbl]
    rest_ids <- colnames(obj)[labels != lbl]
    
    n_pos <- length(one_ids)
    n_neg <- length(rest_ids)
    
    if (n_neg == 0) {
      warning(sprintf("Class '%s' has no negative examples, skipping", lbl))
      next
    }
    
    # Positive class handling
    if (is_test) {
      # Keep all positive cells for test set (unbalanced)
      one_ids_final <- one_ids
      message(sprintf("  %s: Keeping all %d positive cells (unbalanced mode)", lbl, n_pos))
    } else {
      # Training set: balance via leverage-based downsampling
      obj_one <- safe_subset(obj, one_ids)
      
      obj_one <- NormalizeData(obj_one, normalization.method = "CLR",
                               margin = 2, assay = assay, verbose = FALSE)
      obj_one <- FindVariableFeatures(obj_one, assay = assay, verbose = FALSE)
      
      n_target <- min(n_pos, n_neg)
      
      # Manual leverage sampling
      var_features <- VariableFeatures(obj_one)
      if (length(var_features) == 0) var_features <- rownames(obj_one)
      data_mat <- GetAssayData(obj_one, slot = "data", assay = assay)[var_features, , drop = FALSE]
      sampled_idx <- leverage_sample(as.matrix(data_mat), n_target, seed)
      one_ids_final <- one_ids[sampled_idx]
      
      message(sprintf("  %s: %d -> %d positive cells (balanced training)", lbl, n_pos, length(one_ids_final)))
    }
    
    # Negative class handling
    if (is_test) {
      # Keep all negative cells for test (unbalanced)
      rest_ids_final <- rest_ids
      message(sprintf("  %s: Keeping all %d negative cells (unbalanced mode)", lbl, n_neg))
    } else {
      # Training set: balance negative class via leverage-based downsampling
      obj_rest <- safe_subset(obj, rest_ids)
      
      obj_rest <- NormalizeData(obj_rest, normalization.method = "CLR",
                                margin = 2, assay = assay, verbose = FALSE)
      obj_rest <- FindVariableFeatures(obj_rest, assay = assay, verbose = FALSE)
      
      n_target <- min(length(one_ids_final), n_neg)
      
      # Manual leverage sampling
      var_features <- VariableFeatures(obj_rest)
      if (length(var_features) == 0) var_features <- rownames(obj_rest)
      data_mat <- GetAssayData(obj_rest, slot = "data", assay = assay)[var_features, , drop = FALSE]
      sampled_idx <- leverage_sample(as.matrix(data_mat), n_target, seed)
      rest_ids_final <- rest_ids[sampled_idx]
      
      message(sprintf("  %s: %d -> %d negative cells (balanced training)", lbl, n_neg, length(rest_ids_final)))
    }
    
    # Store as dataframe with NA padding if needed
    if (is_test && length(one_ids_final) != length(rest_ids_final)) {
      # Unbalanced: pad shorter vector with NA
      max_len <- max(length(one_ids_final), length(rest_ids_final))
      length(one_ids_final) <- max_len
      length(rest_ids_final) <- max_len
    }
    
    barcodes_list[[lbl]] <- data.frame(
      Positive = one_ids_final,
      Negative = rest_ids_final,
      stringsAsFactors = FALSE
    )
  }
  
  message(sprintf("Generated OVR barcodes for %d classes", length(barcodes_list)))
  barcodes_list
}

#' Validate split quality with correlation heatmap (Seurat v5 compatible)
#'
#' @param obj_train Training Seurat object
#' @param obj_holdout Test/Cal Seurat object
#' @param label_col Label column
#' @param assay Assay name
#' @param title Plot title
#' @return ComplexHeatmap object
validate_split_correlation <- function(obj_train, obj_holdout, label_col, assay, 
                                       title = "Train vs Holdout Correlation") {
  
  # Use AggregateExpression for Seurat v5
  avg_train <- AggregateExpression(
    obj_train, 
    group.by = label_col, 
    assays = assay,
    slot = "data",
    return.seurat = FALSE,
    verbose = FALSE
  )
  
  avg_holdout <- AggregateExpression(
    obj_holdout, 
    group.by = label_col,
    assays = assay, 
    slot = "data",
    return.seurat = FALSE,
    verbose = FALSE
  )
  
  # Extract matrix from the list structure
  if (is.list(avg_train)) {
    avg_train <- avg_train[[assay]]
  }
  if (is.list(avg_holdout)) {
    avg_holdout <- avg_holdout[[assay]]
  }
  
  # Convert to numeric matrix
  avg_train <- as.matrix(avg_train)
  avg_holdout <- as.matrix(avg_holdout)
  
  # Replace underscores with dashes in column names (Seurat v5 behavior)
  colnames(avg_train) <- gsub("_", "-", colnames(avg_train))
  colnames(avg_holdout) <- gsub("_", "-", colnames(avg_holdout))
  
  # Find common classes between train and holdout
  common_classes <- intersect(colnames(avg_train), colnames(avg_holdout))
  
  if (length(common_classes) == 0) {
    warning("No common classes between train and holdout sets")
    return(NULL)
  }
  
  message(sprintf("Computing correlation for %d common classes", length(common_classes)))
  
  # Subset to common classes
  avg_train <- avg_train[, common_classes, drop = FALSE]
  avg_holdout <- avg_holdout[, common_classes, drop = FALSE]
  
  # Compute correlation
  cor_mat <- cor(avg_train, avg_holdout)
  
  # Create heatmap with magma colormap
  ht <- Heatmap(
    cor_mat, 
    cluster_rows = FALSE, 
    cluster_columns = FALSE,
    name = "Correlation",
    column_title = title,
    col = viridis::magma(100), 
    column_title_gp = gpar(fontsize = 16),
    row_names_gp = gpar(fontsize = 14),
    column_names_gp = gpar(fontsize = 14),
    heatmap_legend_param = list(
      legend_direction = "horizontal", 
      legend_width = unit(4, "cm")),
    cell_fun = function(j, i, x, y, width, height, fill) {
      if (i <= ncol(cor_mat) && j <= nrow(cor_mat)) {
        if (rownames(cor_mat)[i] == colnames(cor_mat)[j]) {
          grid.rect(x = x, y = y, width = width, height = height,
                    gp = gpar(col = "black", fill = NA, lwd = 2))
        }
      }
    }
  )
  
  return(ht)
}

#' Save split objects as h5ad files
#'
#' @param splits List with train, cal, test objects
#' @param atlas_name Name of atlas
#' @param original_filename Original h5ad filename
#' @param output_dir Output directory path
#' @param assay Assay name
save_splits_h5ad <- function(splits, atlas_name, original_filename, 
                             output_dir, assay) {
  dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)
  
  for (split_name in c("train", "cal", "test")) {
    split_obj <- splits[[split_name]]
    
    # Clean before saving
    split_obj <- clean_seurat_object(split_obj)
    
    out_file <- file.path(
      output_dir,
      sub("\\.h5ad$", paste0("_", str_to_title(split_name), ".h5ad"), 
          basename(original_filename))
    )
    
    message(sprintf("Saving %s split to: %s", split_name, out_file))
    
    tryCatch({
      sceasy::convertFormat(
        obj = split_obj,
        outFile = out_file,
        from = "seurat",
        to = "anndata",
        assay = assay,
        main_layer = "counts"
      )
      message(sprintf("  Successfully saved %s split", split_name))
    }, error = function(e) {
      warning(sprintf("Failed to save %s split: %s", split_name, e$message))
    })
  }
}

#' Save OVR barcodes to CSV files
#'
#' @param barcodes_list Named list of barcode dataframes
#' @param output_dir Output directory path
#' @param prefix Filename prefix (e.g., "training" or "testing")
save_ovr_barcodes <- function(barcodes_list, output_dir, prefix = "training") {
  dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)
  
  iwalk(barcodes_list, function(df, lbl) {
    safe_lbl <- gsub("[^A-Za-z0-9_.-]+", "_", lbl)
    out_file <- file.path(output_dir, 
                          sprintf("Barcodes_%s_class_%s.csv", prefix, safe_lbl))
    write.csv(df, file = out_file, row.names = TRUE)
  })
  
  message(sprintf("Saved %d barcode files to: %s", length(barcodes_list), output_dir))
}