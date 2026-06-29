source("CRF/CTree.R")

# CRF: RF-like ensemble for CTree-style splits
CRF <- function(
    data, target, environment, features,
    n.trees = 200,
    sample.fraction = 1,
    replace = TRUE,
    
    sample_nodes = TRUE,
    mtry = "rf",
    
    lambda1 = 5,
    lambda2 = 5,
    min.leaf.size = 3,
    eps.improvement = 1e-2,
    n.thresh = NULL,
    
    seed = NULL,
    
    relax.constraint = FALSE,
    
    compute.oob = TRUE,
    
    importance = TRUE,
    importance.type = c("both", "permutation", "gini"),
    n.perm = 1,
    parallel.importance = TRUE,
    
    parallel = TRUE,
    n.cores = NULL,
    
    # Rho computation
    # If compute_rho_full=TRUE -> computes rho on each tree's INBAG sample (default behavior requested)
    compute_rho_full = FALSE,
    # If compute_rho_full_data=TRUE -> additionally computes rho on FULL data (separate argument)
    compute_rho_full_data = FALSE,
    # OOB rho (each tree evaluated on its own OOB subset; weighted by OOB size)
    compute_rho_oob  = FALSE,
    rho_min_oob = 5,
    rho_parallel = parallel,
    rho_n.cores  = n.cores,

    compute_diversity = FALSE,
    diversity_data = NULL,

    verbose = FALSE
) {
  # Create CRF: random-forest-like ensemble of CTrees with invariance constraints
  # Arguments:
  # - data: data frame containing the dataset.
  # - target: name of the target column (Y) as a string (binary/categorical classification).
  # - environment: name of the environmental variable (Z) as a string (categorical environment indicator).
  # - features: character vector of feature column names eligible for splitting (typically numeric/ordered).
  # - n.trees: number of trees to train in the ensemble.
  # - sample.fraction: fraction of rows sampled to grow each tree (1 = full bootstrap size; <1 = subsampling).
  # - replace: logical; if TRUE, sample rows with replacement (bootstrap). If FALSE, sample without replacement.
  # - sample_nodes: logical; if TRUE, sample a subset of features at each split node (RF-style node feature sampling).
  #                if FALSE, consider all features at every split node.
  # - mtry: number of features considered at each split node when sample_nodes=TRUE.
  #         Options: "rf" (default, floor(sqrt(p))), "all", or a single integer in [1, p].
  # - lambda1: weighting parameter for the conditional mutual information penalty term (constraint (i)).
  # - lambda2: weighting parameter for the environment entropy change penalty term (constraint (ii)).
  # - min.leaf.size: minimum number of observations allowed in each child node after a split.
  # - eps.improvement: minimum required improvement in the split loss to accept a split (early stopping).
  # - n.thresh: number of candidate thresholds evaluated per feature for splitting.
  #             If NULL, all observed values are considered; otherwise quantile-based candidates are used.
  # - seed: optional integer; random seed for reproducibility.
  # - compute.oob: logical; if TRUE, compute out-of-bag (OOB) predictions and OOB error on training data.
  # - importance: logical; if TRUE, compute feature importance measures and store in $importance.
  # - importance.type: "permutation", "gini", or "both".
  # - n.perm: number of permutation repeats per feature per tree for permutation importance.
  # - parallel: logical; if TRUE, train trees in parallel using a PSOCK cluster.
  # - n.cores: number of worker processes to use when parallel=TRUE.
  # - compute_rho_full: if TRUE, compute rho metrics averaged over trees using each tree's INBAG sample.
  # - compute_rho_full_data: if TRUE, also compute rho metrics averaged over trees using FULL data.
  # - compute_rho_oob: if TRUE, compute rho metrics averaged over trees using each tree's OOB sample (weighted).
  # - rho_min_oob: minimum OOB size per tree to include it in rho_oob aggregation.
  # - verbose: logical; if TRUE, print progress messages during training.
  
  ## ---- checks ----
  if (!is.data.frame(data)) stop("data must be a data.frame.")
  if (!target %in% names(data)) stop("target column not found in data.")
  if (!environment %in% names(data)) stop("environment column not found in data.")
  if (!all(features %in% names(data))) stop("Some features not found in data.")
  if (n.trees < 1) stop("n.trees must be >= 1.")
  if (sample.fraction <= 0) stop("sample.fraction must be > 0.")
  if (!replace && sample.fraction > 1) stop("If replace=FALSE, sample.fraction must be <= 1.")
  if (!is.numeric(n.perm) || length(n.perm) != 1 || n.perm < 1) stop("n.perm must be >= 1.")
  
  ## CTree script must provide these
  required_tree_fns <- c(
    "best_split_overall", "best_split", "loss_function",
    "conditional_entropy", "conditional_mutual_information",
    "mutual_information", "entropy"
  )
  missing_fns <- required_tree_fns[!vapply(required_tree_fns, exists, logical(1), mode = "function")]
  if (length(missing_fns) > 0) {
    stop("Missing required functions from CTree script: ",
         paste(missing_fns, collapse = ", "),
         ". Source CTree.R first.")
  }
  
  n <- nrow(data)
  p <- length(features)
  classes <- sort(unique(as.character(data[[target]])))
  
  ## mtry: only relevant if sample_nodes=TRUE
  mtry_eff <- if (!sample_nodes) p else .resolve_mtry(mtry, p)
  
  ## tree sample size
  samp_size <- if (replace) ceiling(sample.fraction * n) else floor(sample.fraction * n)
  samp_size <- max(1L, min(n, samp_size))
  
  ## RNG: parallel-safe reproducibility
  RNGkind("L'Ecuyer-CMRG")
  if (!is.null(seed)) set.seed(seed)
  
  ## per-tree seeds (sequential vs parallel reproducible)
  tree_seeds <- sample.int(.Machine$integer.max, n.trees)
  
  ## per-tree seeds for permutation importance
  imp_seeds <- if (importance && (match.arg(importance.type) %in% c("both", "permutation"))) {
    sample.int(.Machine$integer.max, n.trees)
  } else {
    NULL
  }
  
  ## -------- train trees --------
  train_one <- function(b) {
    .train_one_tree_rf(
      b = b,
      data = data, n = n,
      samp_size = samp_size, replace = replace,
      target = target, environment = environment, features = features,
      classes = classes,
      lambda1 = lambda1, lambda2 = lambda2,
      min.leaf.size = min.leaf.size,
      eps.improvement = eps.improvement,
      n.thresh = n.thresh,
      sample_nodes = sample_nodes,
      mtry = mtry_eff,
      seed = tree_seeds[b],
      relax.constraint= relax.constraint,
      verbose = verbose
    )
  }
  
  if (parallel && n.trees > 1) {
    if (is.null(n.cores)) n.cores <- max(1L, parallel::detectCores(logical = FALSE))
    n.cores <- max(1L, min(as.integer(n.cores), n.trees))
    
    if (n.cores <= 1) {
      results <- lapply(seq_len(n.trees), train_one)
    } else {
      cl <- parallel::makeCluster(n.cores)
      on.exit(parallel::stopCluster(cl), add = TRUE)
      
      parallel::clusterEvalQ(cl, { library(dplyr) })
      
      parallel::clusterExport(
        cl,
        varlist = c("data", "n", "samp_size", "replace",
                    "target", "environment", "features", "classes",
                    "lambda1", "lambda2", "min.leaf.size",
                    "eps.improvement", "n.thresh",
                    "sample_nodes", "mtry_eff", "tree_seeds", "verbose"),
        envir = environment()
      )
      
      parallel::clusterExport(
        cl,
        varlist = c(".train_one_tree_rf", "grow_tree_rf",
                    ".leaf_object",
                    ".resolve_mtry", ".gini_impurity",
                    ".predict_tree_prob_rowwise", ".traverse_tree_prob"),
        envir = .GlobalEnv
      )
      
      parallel::clusterExport(cl, varlist = required_tree_fns, envir = .GlobalEnv)
      
      results <- parallel::parLapply(cl, seq_len(n.trees), function(b) {
        .train_one_tree_rf(
          b = b,
          data = data, n = n,
          samp_size = samp_size, replace = replace,
          target = target, environment = environment, features = features,
          classes = classes,
          lambda1 = lambda1, lambda2 = lambda2,
          min.leaf.size = min.leaf.size,
          eps.improvement = eps.improvement,
          n.thresh = n.thresh,
          sample_nodes = sample_nodes,
          mtry = mtry_eff,
          seed = tree_seeds[b],
          relax.constraint= relax.constraint,
          verbose = verbose
        )
      })
    }
  } else {
    results <- lapply(seq_len(n.trees), train_one)
  }
  
  ## unpack
  trees     <- lapply(results, `[[`, "tree")
  inbag_idx <- lapply(results, `[[`, "inbag")
  oob_idx   <- lapply(results, `[[`, "oob")
  
  ## split-based importance (always collected during growth)
  per_tree_gini    <- lapply(results, `[[`, "imp_gini")
  per_tree_loss    <- lapply(results, `[[`, "imp_custom_loss")
  per_tree_splits  <- lapply(results, `[[`, "split_counts")
  per_tree_used01  <- lapply(results, `[[`, "split_used")  # ALWAYS computed
  
  sum_gini   <- Reduce(`+`, per_tree_gini)
  sum_loss   <- Reduce(`+`, per_tree_loss)
  sum_splits <- Reduce(`+`, per_tree_splits)
  sum_used   <- Reduce(`+`, per_tree_used01)
  
  mean_gini <- sum_gini / n.trees
  mean_loss <- sum_loss / n.trees
  
  ## ALWAYS computed: inclusion rate (fraction of trees where feature used at least once)
  inclusion_rate <- as.numeric(sum_used[features]) / n.trees
  
  forest <- list(
    trees = trees,
    target = target,
    environment = environment,
    features = features,
    classes = classes,
    n.trees = n.trees,
    
    sample.fraction = sample.fraction,
    replace = replace,
    
    sample_nodes = sample_nodes,
    mtry = mtry_eff,
    
    lambda1 = lambda1,
    lambda2 = lambda2,
    min.leaf.size = min.leaf.size,
    eps.improvement = eps.improvement,
    n.thresh = n.thresh,
    
    inbag_idx = inbag_idx,
    oob_idx = oob_idx,
    
    seed = seed,
    tree_seeds = tree_seeds,
    
    # Split-based importance; extended with inclusion stats, but keeps existing fields.
    importance_split = data.frame(
      Feature = features,
      TimesUsed = as.numeric(sum_splits[features]),
      TreesUsed = as.numeric(sum_used[features]),
      InclusionRate = as.numeric(inclusion_rate),
      MeanDecreaseGini = as.numeric(mean_gini[features]),
      MeanDecreaseCustomLoss = as.numeric(mean_loss[features]),
      row.names = features,
      check.names = FALSE
    )
  )
  class(forest) <- "CRF"
  
  ## OOB predictions (probability aggregation; standard RF behavior)
  if (compute.oob) {
    forest$oob <- oob_predict_CRF(forest, data, type = "class")
  }
  
  ## Rho metrics (optional)
  ## - forest$rho_full: INBAG by default (each tree evaluated on its own inbag sample)
  ## - forest$rho_full_data: optional additional FULL-data variant (separate switch)
  if (isTRUE(compute_rho_full)) {
    forest$rho_full <- rho_forest_inbag(
      forest = forest,
      data = data,
      target = target,
      environment = environment,
      parallel = rho_parallel,
      n.cores = rho_n.cores
    )
  } else {
    forest$rho_full <- NULL
  }
  
  if (isTRUE(compute_rho_full_data)) {
    forest$rho_full_data <- rho_forest_full_data(
      forest = forest,
      data = data,
      target = target,
      environment = environment,
      parallel = rho_parallel,
      n.cores = rho_n.cores
    )
  } else {
    forest$rho_full_data <- NULL
  }
  
  if (isTRUE(compute_rho_oob)) {
    forest$rho_oob <- rho_forest_oob(
      forest = forest,
      data = data,
      target = target,
      environment = environment,
      min_oob = rho_min_oob,
      parallel = rho_parallel,
      n.cores = rho_n.cores
    )
  } else {
    forest$rho_oob <- NULL
  }
  
  ## permutation importance (RF-style MeanDecreaseAccuracy)
  if (importance) {
    importance.type <- match.arg(importance.type)
    
    if (importance.type %in% c("both", "permutation")) {
      perm_imp <- permutation_importance_CRF(
        forest = forest,
        data = data,
        n.perm = n.perm,
        seed_vec = imp_seeds,
        parallel = parallel.importance,
        n.cores = n.cores,
        verbose = verbose
      )
      forest$importance <- merge(
        forest$importance_split,
        perm_imp,
        by = "Feature",
        all.x = TRUE,
        sort = FALSE
      )
      rownames(forest$importance) <- forest$importance$Feature
    } else {
      forest$importance <- forest$importance_split
    }
  } else {
    forest$importance <- NULL
  }

  ## Diversity metrics (evaluated on held-out test data)
  if (isTRUE(compute_diversity)) {
    if (is.null(diversity_data)) stop("diversity_data must be provided when compute_diversity = TRUE.")
    forest$diversity <- compute_diversity_CRF(forest, diversity_data, per_tree_used01)
  } else {
    forest$diversity <- NULL
  }

  forest
}



# Predict from the forest       
predict.CRF <- function(object, newdata,
                             type = c("class", "prob", "votes"),
                             trees = NULL, ...) {
  type <- match.arg(type)
  
  if (missing(newdata) || is.null(newdata)) stop("Please provide newdata.")
  if (!is.data.frame(newdata)) stop("newdata must be a data.frame.")
  
  if (is.null(trees)) {
    tree_idx <- seq_along(object$trees)
  } else {
    tree_idx <- as.integer(trees)
    tree_idx <- tree_idx[tree_idx >= 1 & tree_idx <= length(object$trees)]
    if (length(tree_idx) == 0) stop("trees selection is empty/invalid.")
  }
  
  n <- nrow(newdata)
  classes <- object$classes
  k <- length(classes)
  
  prob_sum <- matrix(0, nrow = n, ncol = k, dimnames = list(NULL, classes))
  votes <- matrix(0L, nrow = n, ncol = k, dimnames = list(NULL, classes))
  
  for (b in tree_idx) {
    probs_b <- .predict_tree_prob_rowwise(object$trees[[b]], newdata, classes)
    prob_sum <- prob_sum + probs_b
    
    hard_b <- classes[max.col(probs_b, ties.method = "first")]
    col_idx <- match(hard_b, classes)
    votes[cbind(seq_len(n), col_idx)] <- votes[cbind(seq_len(n), col_idx)] + 1L
  }
  
  if (type == "votes") return(votes)
  
  prob_avg <- prob_sum / length(tree_idx)
  if (type == "prob") return(prob_avg)
  
  classes[max.col(prob_avg, ties.method = "first")]
}


# OOB predictions on training data   
oob_predict_CRF <- function(object, data, type = c("class", "prob")) {
  type <- match.arg(type)
  
  n <- nrow(data)
  classes <- object$classes
  k <- length(classes)
  
  prob_sum <- matrix(0, nrow = n, ncol = k, dimnames = list(NULL, classes))
  counts <- integer(n)
  
  for (b in seq_along(object$trees)) {
    oob <- object$oob_idx[[b]]
    if (length(oob) == 0) next
    
    probs_b <- .predict_tree_prob_rowwise(object$trees[[b]], data[oob, , drop = FALSE], classes)
    prob_sum[oob, ] <- prob_sum[oob, ] + probs_b
    counts[oob] <- counts[oob] + 1L
  }
  
  has_oob <- counts > 0
  
  prob <- matrix(NA_real_, nrow = n, ncol = k, dimnames = list(NULL, classes))
  prob[has_oob, ] <- prob_sum[has_oob, , drop = FALSE] / counts[has_oob]
  
  pred_class <- rep(NA_character_, n)
  pred_class[has_oob] <- classes[max.col(prob[has_oob, , drop = FALSE], ties.method = "first")]
  
  y_true <- as.character(data[[object$target]])
  oob_err <- if (any(has_oob)) mean(pred_class[has_oob] != y_true[has_oob]) else NA_real_
  
  if (type == "prob") {
    return(list(prob = prob, counts = counts, oob_error = oob_err, n_oob_pred = sum(has_oob)))
  }
  
  list(
    pred = pred_class,
    prob = prob,
    counts = counts,
    oob_error = oob_err,
    n_oob_pred = sum(has_oob),
    confusion = if (any(has_oob)) table(truth = y_true[has_oob], pred = pred_class[has_oob]) else NULL
  )
}



# print() convenience                
print.CRF <- function(x, ...) {
  cat("CRF\n")
  cat("  n.trees           :", x$n.trees, "\n")
  cat("  row sampling      :", if (x$replace) "bootstrap (replace=TRUE)" else "subsample (replace=FALSE)", "\n")
  cat("  sample.fraction   :", x$sample.fraction, "\n")
  cat("  sample_nodes      :", x$sample_nodes, "\n")
  cat("  mtry (per node)   :", x$mtry, "of", length(x$features), "features\n")
  cat("  lambda1, lambda2  :", x$lambda1, ", ", x$lambda2, "\n", sep = "")
  cat("  min.leaf.size     :", x$min.leaf.size, "\n")
  cat("  eps.improvement   :", x$eps.improvement, "\n")
  if (!is.null(x$n.thresh)) cat("  n.thresh          :", x$n.thresh, "\n")
  if (!is.null(x$oob) && !is.null(x$oob$oob_error)) cat("  OOB error         :", x$oob$oob_error, "\n")
  if (!is.null(x$importance)) cat("  importance        : available in $importance\n")
  if (!is.null(x$rho_full)) cat("  rho_full (inbag)  : available in $rho_full\n")
  if (!is.null(x$rho_full_data)) cat("  rho_full_data     : available in $rho_full_data\n")
  if (!is.null(x$rho_oob)) cat("  rho_oob           : available in $rho_oob\n")
  invisible(x)
}


# Permutation importance (RF-style MeanDecreaseAccuracy)       
permutation_importance_CRF <- function(forest, data,
                                            n.perm = 1,
                                            seed_vec = NULL,
                                            parallel = TRUE,
                                            n.cores = NULL,
                                            verbose = FALSE) {
  features <- forest$features
  p <- length(features)
  
  one_tree_perm <- function(b) {
    if (!is.null(seed_vec)) set.seed(seed_vec[b])
    
    oob <- forest$oob_idx[[b]]
    if (length(oob) == 0) {
      return(list(inc = setNames(numeric(p), features), weight = 0L))
    }
    
    dat_oob <- data[oob, , drop = FALSE]
    y_true <- as.character(dat_oob[[forest$target]])
    
    probs_base <- .predict_tree_prob_rowwise(forest$trees[[b]], dat_oob, forest$classes)
    pred_base <- forest$classes[max.col(probs_base, ties.method = "first")]
    err_base <- mean(pred_base != y_true)
    
    inc <- numeric(p); names(inc) <- features
    
    for (j in seq_len(p)) {
      f <- features[j]
      inc_j <- 0
      
      for (r in seq_len(n.perm)) {
        dat_perm <- dat_oob
        dat_perm[[f]] <- sample(dat_perm[[f]], replace = FALSE)
        
        probs_perm <- .predict_tree_prob_rowwise(forest$trees[[b]], dat_perm, forest$classes)
        pred_perm <- forest$classes[max.col(probs_perm, ties.method = "first")]
        err_perm <- mean(pred_perm != y_true)
        
        inc_j <- inc_j + (err_perm - err_base)
      }
      
      inc[j] <- inc_j / n.perm
    }
    
    list(inc = inc, weight = length(oob))
  }
  
  if (parallel && forest$n.trees > 1) {
    if (is.null(n.cores)) n.cores <- max(1L, parallel::detectCores(logical = FALSE))
    n.cores <- max(1L, min(as.integer(n.cores), forest$n.trees))
    
    if (n.cores <= 1) {
      res <- lapply(seq_len(forest$n.trees), one_tree_perm)
    } else {
      cl <- parallel::makeCluster(n.cores)
      on.exit(parallel::stopCluster(cl), add = TRUE)
      
      parallel::clusterExport(
        cl,
        varlist = c("forest", "data", "n.perm", "seed_vec", "features", "p",
                    ".predict_tree_prob_rowwise", ".traverse_tree_prob"),
        envir = environment()
      )
      
      res <- parallel::parLapply(cl, seq_len(forest$n.trees), one_tree_perm)
    }
  } else {
    res <- lapply(seq_len(forest$n.trees), one_tree_perm)
  }
  
  weights <- vapply(res, `[[`, numeric(1), "weight")
  total_w <- sum(weights)
  
  if (total_w == 0) {
    mda <- setNames(rep(NA_real_, p), features)
  } else {
    inc_weighted_sum <- Reduce(`+`, Map(function(x, w) x$inc * w, res, weights))
    mda <- inc_weighted_sum / total_w
  }
  
  data.frame(
    Feature = features,
    MeanDecreaseAccuracy = as.numeric(mda[features]),
    row.names = features,
    check.names = FALSE
  )
}


## Helpers
.train_one_tree_rf <- function(b,
                               data, n, samp_size, replace,
                               target, environment, features,
                               classes,
                               lambda1, lambda2,
                               min.leaf.size, eps.improvement, n.thresh,
                               sample_nodes, mtry,
                               seed,
                               relax.constraint=FALSE,
                               verbose = FALSE) {
  set.seed(seed)
  
  idx <- sample.int(n, size = samp_size, replace = replace)
  oob <- setdiff(seq_len(n), unique(idx))
  dat_b <- data[idx, , drop = FALSE]
  
  imp_env <- new.env(parent = emptyenv())
  imp_env$gini <- setNames(numeric(length(features)), features)
  imp_env$custom_loss <- setNames(numeric(length(features)), features)
  imp_env$split_counts <- setNames(integer(length(features)), features)
  
  # Always compute: feature used at least once in this tree (0/1)
  imp_env$split_used <- setNames(integer(length(features)), features)
  
  tree_b <- grow_tree_rf(
    data = dat_b,
    target = target,
    environment = environment,
    features = features,
    classes = classes,
    lambda1 = lambda1,
    lambda2 = lambda2,
    min.leaf.size = min.leaf.size,
    loss_val = Inf,
    eps.improvement = eps.improvement,
    n.thresh = n.thresh,
    sample_nodes = sample_nodes,
    mtry = mtry,
    imp_env = imp_env,
    relax.constraint= relax.constraint,
    verbose = verbose
  )
  
  list(
    tree = tree_b,
    inbag = idx,
    oob = oob,
    imp_gini = imp_env$gini,
    imp_custom_loss = imp_env$custom_loss,
    split_counts = imp_env$split_counts,
    split_used = imp_env$split_used
  )
}


grow_tree_rf <- function(data, target, environment, features, classes,
                         lambda1, lambda2,
                         min.leaf.size,
                         loss_val,
                         eps.improvement,
                         n.thresh,
                         sample_nodes,
                         mtry,
                         imp_env,
                         relax.constraint=FALSE,
                         verbose = FALSE) {
  if (nrow(data) == 0) return(.leaf_object(data, target, classes))
  
  # Stop if pure / no features / relaxation sentinel
  if (length(unique(data[[target]])) == 1 || length(features) == 0 || lambda2 < 0) {
    return(.leaf_object(data, target, classes))
  }
  
  # Node-wise candidate feature sampling (RF behavior)
  if (sample_nodes) {
    mtry_eff <- .resolve_mtry(mtry, length(features))
    cand_features <- if (mtry_eff >= length(features)) features else sample(features, mtry_eff, replace = FALSE)
  } else {
    cand_features <- features
  }
  
  split <- best_split_overall(
    data = data,
    target = target,
    environment = environment,
    features = cand_features,
    lambda1 = lambda1,
    lambda2 = lambda2,
    min.leaf.size = min.leaf.size,
    n.thresh = n.thresh
  )
  
  if (is.null(split$feature) || !is.finite(split$loss)) {
    return(.leaf_object(data, target, classes))
  }
  
  new_loss <- split$loss
  
  relax_used <-  !relax.constraint
  if (is.finite(loss_val)) {
    if (abs(new_loss - loss_val) < eps.improvement || new_loss > loss_val) {
      if (lambda2 >= 0) {
        if (verbose) cat("No meaningful improvement; relaxing penalty #2 once.\n")
        
        split2 <- best_split_overall(
          data = data,
          target = target,
          environment = environment,
          features = cand_features,
          lambda1 = lambda1,
          lambda2 = lambda2/10, # relaxed
          min.leaf.size = min.leaf.size,
          n.thresh = n.thresh
        )
        if (is.null(split2$feature) || !is.finite(split2$loss)) {
          return(.leaf_object(data, target, classes))
        }
        
        split <- split2
        new_loss <- split2$loss
        relax_used <- TRUE
      } else {
        return(.leaf_object(data, target, classes))
      }
    }
  }
  
  feat <- split$feature
  thr  <- split$threshold
  
  x <- data[[feat]]
  left_idx  <- which(x < thr)
  right_idx <- which(x >= thr)
  
  if (length(left_idx) < min.leaf.size || length(right_idx) < min.leaf.size) {
    return(.leaf_object(data, target, classes))
  }
  
  left  <- data[left_idx,  , drop = FALSE]
  right <- data[right_idx, , drop = FALSE]
  
  # Inclusion indicator (used at least once in this tree)
  imp_env$split_used[feat] <- 1L
  
  # ---------- Split-based importance ----------
  n_node <- nrow(data)
  g_parent <- .gini_impurity(data[[target]], classes)
  g_left   <- .gini_impurity(left[[target]], classes)
  g_right  <- .gini_impurity(right[[target]], classes)
  
  w_left  <- nrow(left) / n_node
  w_right <- nrow(right) / n_node
  g_gain <- g_parent - (w_left * g_left + w_right * g_right)
  
  imp_env$gini[feat] <- imp_env$gini[feat] + n_node * g_gain
  imp_env$split_counts[feat] <- imp_env$split_counts[feat] + 1L
  
  mi_node <- as.numeric(mutual_information(data, target, environment))
  if (!is.finite(mi_node)) mi_node <- 0
  loss_unsplit <- entropy(data[[target]]) + lambda1 * mi_node
  
  loss_gain <- max(0, loss_unsplit - split$loss)
  imp_env$custom_loss[feat] <- imp_env$custom_loss[feat] + n_node * loss_gain
  # ------------------------------------------
  
  if (verbose) cat("Split: ", feat, " < ", thr, " (nL=", nrow(left), ", nR=", nrow(right), ")\n", sep = "")
  
  lambda2_child <- if (relax_used & relax.constraint) -1 else lambda2
  
  list(
    feature = feat,
    threshold = thr,
    nodes = list(
      left  = grow_tree_rf(left,  target, environment, features, classes,
                           lambda1, lambda2_child,
                           min.leaf.size,
                           loss_val = new_loss,
                           eps.improvement = eps.improvement,
                           n.thresh = n.thresh,
                           sample_nodes = sample_nodes,
                           mtry = mtry,
                           imp_env = imp_env,relax.constraint=relax.constraint,
                           verbose = verbose),
      right = grow_tree_rf(right, target, environment, features, classes,
                           lambda1, lambda2_child,
                           min.leaf.size,
                           loss_val = new_loss,
                           eps.improvement = eps.improvement,
                           n.thresh = n.thresh,
                           sample_nodes = sample_nodes,
                           mtry = mtry,
                           imp_env = imp_env,relax.constraint=relax.constraint,
                           verbose = verbose)
    )
  )
}


.leaf_object <- function(data, target, classes) {
  y <- as.character(data[[target]])
  tab <- table(factor(y, levels = classes))
  counts <- as.numeric(tab)
  names(counts) <- classes
  
  total <- sum(counts)
  prob <- if (total > 0) counts / total else rep(1 / length(classes), length(classes))
  names(prob) <- classes
  
  list(
    leaf = TRUE,
    n = nrow(data),
    counts = counts,
    prob = prob,
    class = names(prob)[which.max(prob)]
  )
}

.gini_impurity <- function(y, classes) {
  y <- as.character(y)
  tab <- table(factor(y, levels = classes))
  p <- as.numeric(tab) / sum(tab)
  1 - sum(p^2)
}

.resolve_mtry <- function(mtry, p) {
  if (is.null(mtry) || (is.character(mtry) && length(mtry) == 1 && mtry == "rf")) {
    return(max(1L, floor(sqrt(p))))
  }
  if (is.character(mtry) && length(mtry) == 1 && mtry == "all") {
    return(as.integer(p))
  }
  if (!is.numeric(mtry) || length(mtry) != 1) {
    stop("mtry must be NULL, 'rf', 'all', or a single numeric value.")
  }
  m <- as.integer(mtry)
  max(1L, min(p, m))
}

.traverse_tree_prob <- function(tree, row_list) {
  if (is.list(tree) && isTRUE(tree$leaf)) return(tree)
  if (!is.list(tree)) return(tree)
  
  feat <- tree$feature
  thr  <- tree$threshold
  val  <- row_list[[feat]]
  
  if (is.na(val)) return(.traverse_tree_prob(tree$nodes$right, row_list))
  if (val < thr) .traverse_tree_prob(tree$nodes$left, row_list) else .traverse_tree_prob(tree$nodes$right, row_list)
}

.predict_tree_prob_rowwise <- function(tree, newdata, classes) {
  n <- nrow(newdata)
  k <- length(classes)
  out <- matrix(0, nrow = n, ncol = k, dimnames = list(NULL, classes))
  
  if (is.list(tree) && isTRUE(tree$leaf)) {
    out[,] <- rep(tree$prob[classes], each = n)
    return(out)
  }
  
  for (i in seq_len(n)) {
    row_list <- as.list(newdata[i, , drop = FALSE])
    leaf <- .traverse_tree_prob(tree, row_list)
    out[i, ] <- leaf$prob[classes]
  }
  out
}


## Rho metrics averaged over trees (leaf-weighted)
## - inbag: each tree evaluated on its inbag (default "full" rho)
## - full:  each tree evaluated on full data (optional extra output)
## - oob:   each tree evaluated on its oob (weighted)

leaf_id_one <- function(tree, row_list, path = "") {
  if (is.list(tree) && isTRUE(tree$leaf)) return(path)
  if (!is.list(tree)) return(path)
  
  feat <- tree$feature
  thr  <- tree$threshold
  val  <- row_list[[feat]]
  
  if (is.na(val)) return(leaf_id_one(tree$nodes$right, row_list, paste0(path, "R")))
  if (val < thr) leaf_id_one(tree$nodes$left,  row_list, paste0(path, "L"))
  else          leaf_id_one(tree$nodes$right, row_list, paste0(path, "R"))
}

leaf_id_tree <- function(tree, data) {
  n <- nrow(data)
  ids <- character(n)
  for (i in seq_len(n)) {
    ids[i] <- leaf_id_one(tree, as.list(data[i, , drop = FALSE]), path = "")
  }
  ids
}

rho_i_tree <- function(tree, data, target = "y", environment = "z") {
  leaf_ids <- leaf_id_tree(tree, data)
  leaves <- unique(leaf_ids)
  N <- nrow(data)
  
  mi_total <- 0
  for (lf in leaves) {
    idx <- which(leaf_ids == lf)
    dat_leaf <- data[idx, , drop = FALSE]
    mi_leaf <- as.numeric(mutual_information(dat_leaf, target, environment))
    if (!is.finite(mi_leaf) || is.na(mi_leaf)) mi_leaf <- 0
    mi_total <- mi_total + (nrow(dat_leaf) / N) * mi_leaf
  }
  unname(as.numeric(mi_total))
}

rho_ii_tree <- function(tree, data, environment = "z") {
  leaf_ids <- leaf_id_tree(tree, data)
  leaves <- unique(leaf_ids)
  N <- nrow(data)
  
  h_root <- as.numeric(entropy(data[[environment]]))
  h_leaves <- 0
  for (lf in leaves) {
    idx <- which(leaf_ids == lf)
    dat_leaf <- data[idx, , drop = FALSE]
    h_leaf <- as.numeric(entropy(dat_leaf[[environment]]))
    h_leaves <- h_leaves + (nrow(dat_leaf) / N) * h_leaf
  }
  unname(as.numeric(h_root - h_leaves))
}

.maybe_parallel_apply_rho <- function(X, FUN, parallel = TRUE, n.cores = NULL,
                                      export = character(0),
                                      envir_export = parent.frame()) {
  # Sequential path
  if (!parallel || length(X) <= 1) {
    mat_list <- lapply(X, FUN)
    mat <- do.call(rbind, mat_list)
    if (is.null(dim(mat))) mat <- matrix(mat, nrow = 1)
    if (!is.null(colnames(mat))) colnames(mat) <- gsub("\\.", "_", colnames(mat))
    return(mat)
  }
  
  if (is.null(n.cores)) n.cores <- max(1L, parallel::detectCores(logical = FALSE))
  n.cores <- max(1L, min(as.integer(n.cores), length(X)))
  if (n.cores <= 1) {
    mat_list <- lapply(X, FUN)
    mat <- do.call(rbind, mat_list)
    if (is.null(dim(mat))) mat <- matrix(mat, nrow = 1)
    if (!is.null(colnames(mat))) colnames(mat) <- gsub("\\.", "_", colnames(mat))
    return(mat)
  }
  
  cl <- parallel::makeCluster(n.cores)
  on.exit(parallel::stopCluster(cl), add = TRUE)
  
  # Export ONLY from the caller environment where 'forest'/'data' actually exist
  if (length(export) > 0) {
    parallel::clusterExport(cl, varlist = export, envir = envir_export)
  }
  
  mat_list <- parallel::parLapply(cl, X, FUN)
  mat <- do.call(rbind, mat_list)
  if (is.null(dim(mat))) mat <- matrix(mat, nrow = 1)
  if (!is.null(colnames(mat))) colnames(mat) <- gsub("\\.", "_", colnames(mat))
  mat
}


# INBAG rho: mean across trees (each tree evaluated on its own inbag sample)
rho_forest_inbag <- function(forest, data, target = "y", environment = "z",
                             parallel = TRUE, n.cores = NULL) {
  T <- length(forest$trees)
  
  one_tree <- function(b) {
    dat_eval <- data[forest$inbag_idx[[b]], , drop = FALSE]
    tr <- forest$trees[[b]]
    c(
      rho_i  = as.numeric(rho_i_tree(tr, dat_eval, target = target, environment = environment)),
      rho_ii = as.numeric(rho_ii_tree(tr, dat_eval, environment = environment))
    )
  }
  
  mat <- .maybe_parallel_apply_rho(
    X = seq_len(T),
    FUN = one_tree,
    parallel = parallel,
    n.cores = n.cores,
    export = c("forest", "data", "target", "environment",
               "leaf_id_one", "leaf_id_tree", "rho_i_tree", "rho_ii_tree",
               "mutual_information", "entropy"),
    envir_export = environment()
  )
  
  if (is.null(colnames(mat)) || !all(c("rho_i", "rho_ii") %in% colnames(mat))) {
    stop("rho (inbag) computation failed. Got columns: ", paste(colnames(mat), collapse = ", "))
  }
  
  list(
    rho.i = mean(mat[, "rho_i"]),
    rho.ii = mean(mat[, "rho_ii"]),
    per_tree = mat,
    eval = "inbag"
  )
}

# FULL-data rho: mean across trees (each tree evaluated on full data)
rho_forest_full_data <- function(forest, data, target = "y", environment = "z",
                                 parallel = TRUE, n.cores = NULL) {
  T <- length(forest$trees)
  
  one_tree <- function(b) {
    tr <- forest$trees[[b]]
    c(
      rho_i  = as.numeric(rho_i_tree(tr, data, target = target, environment = environment)),
      rho_ii = as.numeric(rho_ii_tree(tr, data, environment = environment))
    )
  }
  
  mat <- .maybe_parallel_apply_rho(
    X = seq_len(T),
    FUN = one_tree,
    parallel = parallel,
    n.cores = n.cores,
    export = c("forest", "data", "target", "environment",
               "leaf_id_one", "leaf_id_tree", "rho_i_tree", "rho_ii_tree",
               "mutual_information", "entropy"),
    envir_export = environment()
  )
  
  if (is.null(colnames(mat)) || !all(c("rho_i", "rho_ii") %in% colnames(mat))) {
    stop("rho (full-data) computation failed. Got columns: ", paste(colnames(mat), collapse = ", "))
  }
  
  list(
    rho.i = mean(mat[, "rho_i"]),
    rho.ii = mean(mat[, "rho_ii"]),
    per_tree = mat,
    eval = "full"
  )
}

# OOB rho: weighted average across trees by OOB size
rho_forest_oob <- function(forest, data, target = "y", environment = "z",
                           min_oob = 5,
                           parallel = TRUE, n.cores = NULL) {
  if (is.null(forest$oob_idx)) stop("forest$oob_idx not found; cannot compute OOB rho.")
  T <- length(forest$trees)
  
  one_tree <- function(b) {
    oob <- forest$oob_idx[[b]]
    if (length(oob) < min_oob) return(c(rho_i = NA_real_, rho_ii = NA_real_, weight = 0))
    dat_eval <- data[oob, , drop = FALSE]
    tr <- forest$trees[[b]]
    c(
      rho_i  = as.numeric(rho_i_tree(tr, dat_eval, target = target, environment = environment)),
      rho_ii = as.numeric(rho_ii_tree(tr, dat_eval, environment = environment)),
      weight = as.numeric(length(oob))
    )
  }
  
  mat <- .maybe_parallel_apply_rho(
    X = seq_len(T),
    FUN = one_tree,
    parallel = parallel,
    n.cores = n.cores,
    export = c("forest", "data", "target", "environment", "min_oob",
               "leaf_id_one", "leaf_id_tree", "rho_i_tree", "rho_ii_tree",
               "mutual_information", "entropy"),
    envir_export = environment()
  )
  
  if (is.null(colnames(mat)) || !all(c("rho_i", "rho_ii", "weight") %in% colnames(mat))) {
    stop("rho (oob) computation failed. Got columns: ", paste(colnames(mat), collapse = ", "))
  }
  
  w <- mat[, "weight"]
  ok <- w > 0 & is.finite(w)
  if (!any(ok)) return(list(rho.i = NA_real_, rho.ii = NA_real_, per_tree = mat, eval = "oob"))
  
  list(
    rho.i = sum(mat[ok, "rho_i"]  * w[ok]) / sum(w[ok]),
    rho.ii = sum(mat[ok, "rho_ii"] * w[ok]) / sum(w[ok]),
    per_tree = mat,
    eval = "oob"
  )
}


# Pairwise tree diversity measures evaluated on a held-out test set.
# Returns mean pairwise disagreement rate, double-fault rate, and Jaccard
# similarity of feature inclusion sets, averaged over all tree pairs.
compute_diversity_CRF <- function(forest, test_data, per_tree_used01) {
  y_true   <- as.character(test_data[[forest$target]])
  n_test   <- nrow(test_data)
  T        <- forest$n.trees
  features <- forest$features

  # Per-tree hard predictions: n_test x T character matrix
  preds <- sapply(seq_len(T), function(b) predict.CRF(forest, test_data, trees = b))

  # Binary prediction matrix (1 = predicted as second class level)
  pred_bin <- (preds == forest$classes[2]) + 0L

  # Correct/wrong indicator matrices (n_test x T)
  correct  <- preds == y_true
  wrong    <- (!correct) + 0L

  # Pairwise disagreement via cross-product (T x T):
  # agree[b1,b2] = n obs where both predict class1 + n obs where both predict class2
  agree_mat    <- crossprod(pred_bin) + crossprod(1L - pred_bin)
  disagree_mat <- (n_test - agree_mat) / n_test

  # Pairwise double-fault (both wrong) via cross-product (T x T)
  dfault_mat <- crossprod(wrong) / n_test

  # Pairwise Jaccard similarity of feature inclusion sets (T x p -> T x T)
  used_mat  <- do.call(rbind, lapply(per_tree_used01, function(v) as.integer(v[features] > 0)))
  inter_mat <- tcrossprod(used_mat)
  row_sums  <- rowSums(used_mat)
  union_mat <- outer(row_sums, row_sums, "+") - inter_mat
  all_zero  <- outer(row_sums == 0L, row_sums == 0L, "&")
  jaccard_mat <- ifelse(all_zero, 1, inter_mat / pmax(union_mat, 1L))

  ut <- upper.tri(disagree_mat)
  list(
    mean_disagreement = mean(disagree_mat[ut]),
    mean_double_fault = mean(dfault_mat[ut]),
    mean_jaccard      = mean(jaccard_mat[ut])
  )
}
