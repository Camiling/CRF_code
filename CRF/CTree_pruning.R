# CTree pruning + K-fold CV for cp selection 

# MAIN (prune at a chosen cp)
# Prunes a CTree 'tree' using rpart-style cost–complexity with threshold 'cp'.
# Repeatedly prune the set of nodes whose α equals the current weakest-link; continue while the minimum α <= cp.
CTree_prune_cp <- function(tree, data, target, environment,
                                lambda1, lambda2, cp, tol = 1e-12, verbose=FALSE) {
  # If empty tree, return 
  if (rt_is_leaf(tree)) return(tree)
  
  continue = TRUE
  # Traverse tree with weakest-link pruning (find each internal node's complexity parameter - alpha). Continue until we reach the treshold.
  while (continue) {
    amap <- rt_alpha_map(tree, data, target, environment, lambda1, lambda2) # alpha of each subtree in current tree + description of path
    if (nrow(amap) == 0L) break() # End of tree
    a_min <- min(amap$alpha, na.rm = TRUE) # The smallest alpha
    if (!is.finite(a_min) || a_min > cp + tol) break() # Stop pruning when smallest alpha is larger than the provided cp threshold
    
    # prune all nodes tied at current minimal α (weakest-link step)
    candidates <- amap$path[is.finite(amap$alpha) & (amap$alpha <= a_min + tol)] # Those to be removed (path format: "L", "LR", etc.)
    keep <- rt_minimal_prefix_set(candidates) # drop descendants when an ancestor is pruned.
    
    # Prune tree; removing all descendants of pruned nodes
    tree <- rt_prune_at_paths(tree, data, target, environment, keep)
    if (rt_is_leaf(tree)) continue = FALSE # Stop when end of tree is reached; otherwise continue to top 
  }
  
  # propagate λ's for convenience
  if (is.list(tree)) {
    tree$lambda1 <- lambda1
    tree$lambda2 <- lambda2
  }
  tree
}


# MAIN (K-fold CV to select cp)
# Row-wise K-fold CV (no refitting of CTree inside folds).
# For each fold:
#   - use the SAME CTree structure (argument 'tree'),
#   - build cp path (weakest-link alphas on full TRAIN data),
#   - evaluate each cp on validation via your normalized loss,
#   - pick cp by "min" or "1se" (choose simplest tree within 1-SE).
RT_cv_select_cp <- function(tree, data, target, environment, features,
                            lambda1, lambda2,
                            K = 5, seed = 1, rule = c("min","1se"),
                            min.leaf.size = 3, eps.improvement = 1e-2, n.thresh = NULL, verbose=FALSE) {
  rule <- match.arg(rule)
  if (K < 2 || K > nrow(data)) stop("K must be in [2, nrow(data)].")
  
  set.seed(seed)
  ridx <- sample.int(nrow(data))
  fold_id <- rep_len(1:K, nrow(data))[order(order(ridx))]
  
  fold_tr <- vector("list", K)
  fold_te <- vector("list", K)
  
  for (k in 1:K) {
    fold_tr[[k]] <- data[fold_id != k, , drop = FALSE]
    fold_te[[k]] <- data[fold_id == k, , drop = FALSE]
  }
  
  # CP grid from the given tree on the full data
  cp_grid <- rt_cp_path(tree, data, target, environment, lambda1, lambda2)
  if (length(cp_grid) == 0L) cp_grid <- 0
  
  # Evaluate on validation sets using your normalized full loss
  xerror <- numeric(length(cp_grid))
  xstd   <- numeric(length(cp_grid))
  for (j in seq_along(cp_grid)) {
    cp <- cp_grid[j]
    errs <- numeric(K)
    for (k in 1:K) {
      tr <- fold_tr[[k]]
      te <- fold_te[[k]]
      # prune the same tree using the training part of fold k
      pruned_k <- CTree_prune_cp(tree, tr, target, environment, lambda1, lambda2, cp)
      denom <- rt_root_leaf_loss_full(te, target, environment, lambda1, lambda2)  # H(Y)+λ1 I(Y;Z) # the loss at the root: lambda2 terms cancels out
      errs[k] <- rt_tree_loss_full(pruned_k, te, target, environment, lambda1, lambda2) / denom
    }
    xerror[j] <- mean(errs)
    xstd[j]   <- stats::sd(errs) / sqrt(K)
    if(verbose) cat('Finished on all', K, 'folds for cp=',cp  ,'(value number', j, 'out of',length(cp_grid),  '). Error:', xerror[j],'\n')
  }
  
  # Choose cp by rpart rule
  imin <- which.min(xerror)
  if (rule == "min") {
    tol <- .Machine$double.eps^0.5
    ok <- which(abs(xerror - xerror[imin]) <= tol)
    cp_chosen <- max(cp_grid[ok])                # simplest among minima
  } else {
    thr <- xerror[imin] + xstd[imin]
    ok <- which(xerror <= thr)
    cp_chosen <- max(cp_grid[ok])                # simplest within 1-SE band
  }
  
  list(cp = cp_chosen, cptable = data.frame(CP = cp_grid, xerror = xerror, xstd = xstd))
}


# OPTIONAL convenience: prune with CV inside CTree
# Call this from the end of CTree(...) if prune=TRUE and you want automatic cp:
RT_prune_with_cv <- function(tree, data, target, environment, features,
                             lambda1, lambda2,
                             K = 5, seed = 1, rule = "1se",
                             min.leaf.size = 3, eps.improvement = 1e-2, n.thresh = NULL, verbose=FALSE) {
  if(verbose) cat('Starting ', K, '-fold CV to select complexity parameter. \n')
  sel <- RT_cv_select_cp(tree, data, target, environment, features,
                         lambda1, lambda2, K, seed, rule,
                         min.leaf.size, eps.improvement, n.thresh, verbose=verbose)
  if(verbose) cat('CV Finished. Optimal cp: ', sel$cp, '. Pruning at this value. \n')
  out <- CTree_prune_cp(tree, data, target, environment, lambda1, lambda2, cp = sel$cp)
  # Attach diagnostics
  if (is.list(out)) {
    out$cp <- sel$cp
    out$cptable <- sel$cptable
  } else {
    attr(out, "cp") <- sel$cp
    attr(out, "cptable") <- sel$cptable
  }
  out
}


# --- Structure helpers --------------------------------------------------------
rt_is_leaf <- function(node) {
  !is.list(node) || is.null(node$feature) || is.null(node$nodes)
}

rt_partition <- function(node, data) {
  left  <- dplyr::filter(data, .data[[node$feature]] <  node$threshold)
  right <- dplyr::filter(data, .data[[node$feature]] >= node$threshold)
  list(left = left, right = right)
}

rt_majority_label <- function(data, target) {
  if (nrow(data) == 0L) return(NA_character_)
  tab <- table(data[[target]])
  if (length(tab) == 0L) return(NA_character_)
  names(tab)[which.max(tab)]
}

rt_count_splits <- function(tree) {
  if (rt_is_leaf(tree)) return(0L)
  1L + rt_count_splits(tree$nodes$left) + rt_count_splits(tree$nodes$right)
}

# --- Base (count-weighted) loss pieces used for α at a node -------------------
# base(node-as-leaf at t) = n_t * [ H(Y|t) + λ1 I(Y;Z|t) - λ2 H(Z|t) ]
rt_leaf_base_counts <- function(data, target, environment, lambda1, lambda2) {
  n <- nrow(data)
  if (n == 0L) return(0)
  Hy  <- entropy(data[[target]]);               if (!is.finite(Hy )) Hy  <- 0
  Iyz <- mutual_information(data, target, environment); if (!is.finite(Iyz)) Iyz <- 0
  Hz  <- entropy(data[[environment]]);          if (!is.finite(Hz )) Hz  <- 0
  n * (Hy + lambda1 * Iyz - lambda2 * Hz)
}

# base(subtree under t) = sum over leaves ℓ: n_ℓ * [ H(Y|ℓ) + λ1 I(Y;Z|ℓ) - λ2 H(Z|ℓ) ]
rt_subtree_base_counts_and_leaves <- function(tree, data, target, environment, lambda1, lambda2) {
  if (rt_is_leaf(tree)) {
    return(list(base = rt_leaf_base_counts(data, target, environment, lambda1, lambda2),
                leaves = 1L))
  }
  parts <- rt_partition(tree, data)
  L <- rt_subtree_base_counts_and_leaves(tree$nodes$left,  parts$left,  target, environment, lambda1, lambda2)
  R <- rt_subtree_base_counts_and_leaves(tree$nodes$right, parts$right, target, environment, lambda1, lambda2)
  list(base = L$base + R$base, leaves = L$leaves + R$leaves)
}

# α at internal node t on its local data
rt_node_alpha_at <- function(node, data, target, environment, lambda1, lambda2) {
  leaf_base <- rt_leaf_base_counts(data, target, environment, lambda1, lambda2)
  sub <- rt_subtree_base_counts_and_leaves(node, data, target, environment, lambda1, lambda2)
  if (sub$leaves <= 1L) return(Inf)
  (leaf_base - sub$base) / (sub$leaves - 1)
}

# Map of all internal-node alphas (path can be: "", "L", "R", "LL", ... etc.)
rt_alpha_map <- function(tree, data, target, environment, lambda1, lambda2, path = "") {
  if (rt_is_leaf(tree)) return(data.frame(path = character(0), alpha = numeric(0)))
  # The alpha of current internal node
  a_here <- rt_node_alpha_at(tree, data, target, environment, lambda1, lambda2)
  # Find next partition
  parts <- rt_partition(tree, data)
  # Traverse left and right branches to compute alpha for each internal node
  left_df  <- rt_alpha_map(tree$nodes$left,  parts$left,  target, environment, lambda1, lambda2, paste0(path, "L"))
  right_df <- rt_alpha_map(tree$nodes$right, parts$right, target, environment, lambda1, lambda2, paste0(path, "R"))
  rbind(data.frame(path = path, alpha = a_here), left_df, right_df)
}

# Keep only minimal prefixes from a set of paths (if "L" is kept, drop "LL","LR",...)
rt_minimal_prefix_set <- function(paths) {
  paths <- unique(paths)
  if (length(paths) == 0L) return(paths)
  ord <- order(nchar(paths), paths)
  keep <- character(0)
  for (p in paths[ord]) {
    if (!any(startsWith(p, keep))) keep <- c(keep, p)
  }
  keep
}

# Data that reach a given path
rt_data_for_path <- function(tree, data, path) {
  if (path == "" || rt_is_leaf(tree)) return(data)
  head <- substring(path, 1, 1)
  tail <- substring(path, 2)
  parts <- rt_partition(tree, data)
  if (head == "L") rt_data_for_path(tree$nodes$left,  parts$left,  tail) else
    rt_data_for_path(tree$nodes$right, parts$right, tail)
}

# Replace subtree at 'path' with 'new_subtree' (path "" replaces root)
rt_replace_subtree_at_path <- function(tree, path, new_subtree) {
  if (path == "") return(new_subtree)
  dirs <- strsplit(path, "", fixed = TRUE)[[1]]
  chain <- vector("list", length(dirs) + 1L)
  cur <- tree
  chain[[1]] <- cur
  for (i in seq_along(dirs)) {
    cur <- if (dirs[i] == "L") cur$nodes$left else cur$nodes$right
    chain[[i + 1L]] <- cur
  }
  # rebuild upward
  out <- new_subtree
  for (i in length(dirs):1) {
    parent <- chain[[i]]
    if (dirs[i] == "L") parent$nodes$left <- out else parent$nodes$right <- out
    out <- parent
  }
  out
}

# Prune a set of node paths in one pass (assumes minimal-prefix set)
rt_prune_at_paths <- function(tree, data, target, environment, paths) {
  if (length(paths) == 0L) return(tree)
  if (any(paths == "")) {
    # prune root to its majority leaf
    return(rt_majority_label(data, target))
  }
  # do independent replacements (disjoint subtrees by construction)
  for (p in paths) {
    df_p <- rt_data_for_path(tree, data, p)
    leaf <- rt_majority_label(df_p, target)
    tree <- rt_replace_subtree_at_path(tree, p, leaf)
  }
  tree
}

# Build cp path (weakest-link alphas) on a given train set
rt_cp_path <- function(tree, data, target, environment, lambda1, lambda2, tol = 1e-12) {
  Tcur <- tree
  cps <- c()
  repeat {
    amap <- rt_alpha_map(Tcur, data, target, environment, lambda1, lambda2)
    if (nrow(amap) == 0L) break
    a_min <- min(amap$alpha, na.rm = TRUE)
    if (!is.finite(a_min)) break
    cps <- c(cps, a_min)
    # prune one weakest-link stage (all ties at this a_min), then continue
    paths_min <- amap$path[is.finite(amap$alpha) & (amap$alpha <= a_min + tol)]
    paths_min <- rt_minimal_prefix_set(paths_min)
    Tcur <- rt_prune_at_paths(Tcur, data, target, environment, paths_min)
    if (rt_count_splits(Tcur) == 0L) break
  }
  sort(unique(cps))
}

# Union cp grid across folds (each computed on its training data)
rt_union_cp_grid <- function(trees, train_list, target, environment, lambda1, lambda2) {
  allcps <- c()
  for (i in seq_along(trees)) {
    cp_i <- rt_cp_path(trees[[i]], train_list[[i]], target, environment, lambda1, lambda2)
    allcps <- c(allcps, cp_i)
  }
  allcps <- sort(unique(allcps[is.finite(allcps)]))
  if (length(allcps) == 0L) 0 else allcps
}

# Fit CTree on a training frame; if no split, return a single leaf (majority class)
rt_fit_CTree <- function(tr, target, environment, features, lambda1, lambda2,
                              min.leaf.size, eps.improvement, n.thresh) {
  model <- CTree(tr, target, environment, features,
                      lambda1 = lambda1, lambda2 = lambda2,
                      min.leaf.size = min.leaf.size, eps.improvement = eps.improvement,
                      cv_selection = FALSE, n.thresh = n.thresh)
  # Proper internal node?
  if (is.list(model) && !is.null(model$feature) && !is.null(model$nodes)) return(model)
  # Otherwise, make a single leaf predicting the majority class on 'tr'
  rt_majority_label(tr, target)
}

# Collect per-leaf data frames for a tree (iterative; avoids nested defs)
rt_collect_leaves_data <- function(tree, data) {
  out <- list()
  nds <- list(tree)
  dfs <- list(data)
  while (length(nds) > 0) {
    nd <- nds[[length(nds)]]
    df <- dfs[[length(dfs)]]
    nds <- nds[-length(nds)]
    dfs <- dfs[-length(dfs)]
    if (rt_is_leaf(nd)) {
      out[[length(out) + 1L]] <- df
    } else {
      parts <- rt_partition(nd, df)
      nds[[length(nds) + 1L]] <- nd$nodes$right
      dfs[[length(dfs) + 1L]] <- parts$right
      nds[[length(nds) + 1L]] <- nd$nodes$left
      dfs[[length(dfs) + 1L]] <- parts$left
    }
  }
  out
}

# Your normalized full loss on a dataset (for CV/training evaluation)
# R(T) = Σ_leaves p_l ( H(Y|l) + λ1 I(Y;Z|l) ) + λ2 ( H(Z) - Σ_leaves p_l H(Z|l) )
rt_tree_loss_full <- function(tree, data, target, environment, lambda1, lambda2) {
  N <- nrow(data); if (N == 0L) return(0)
  leaves <- rt_collect_leaves_data(tree, data)
  if (length(leaves) == 0L) leaves <- list(data)
  
  leaf_sum <- 0
  Hz_root <- if (lambda2 == 0) 0 else entropy(data[[environment]])
  Hz_weighted <- 0
  
  for (lf in leaves) {
    nl <- nrow(lf); if (nl == 0L) next
    wl <- nl / N
    Hy  <- entropy(lf[[target]]);                 if (!is.finite(Hy )) Hy  <- 0
    Iyz <- mutual_information(lf, target, environment); if (!is.finite(Iyz)) Iyz <- 0
    leaf_sum <- leaf_sum + wl * (Hy + lambda1 * Iyz)
    if (lambda2 != 0) {
      Hzl <- entropy(lf[[environment]]);         if (!is.finite(Hzl)) Hzl <- 0
      Hz_weighted <- Hz_weighted + wl * Hzl
    }
  }
  leaf_sum + lambda2 * (Hz_root - Hz_weighted)
}

# Single-leaf baseline for scaling relative error (λ2 cancels at the root-as-leaf)
rt_root_leaf_loss_full <- function(data, target, environment, lambda1, lambda2) {
  Hy  <- entropy(data[[target]]);                 if (!is.finite(Hy )) Hy  <- 0
  Iyz <- mutual_information(data, target, environment); if (!is.finite(Iyz)) Iyz <- 0
  denom <- Hy + lambda1 * Iyz
  if (!is.finite(denom) || denom <= 0) 1e-12 else denom
}
