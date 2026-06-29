source("functions/help_funcs/convert_to_rpart.R")

# Must do this for the initial run:
#reticulate::install_miniconda()
library(reticulate)
library(AnchorRegression)
use_miniconda("r-reticulate", required = TRUE)

reticulate::py_run_string("
import warnings
warnings.filterwarnings(
    'ignore',
    category=FutureWarning,
    message='.*DataFrame concatenation with empty or all-NA entries.*'
)
")

# Must install this the first time:
#py_install("pandas", envname = "r-reticulate")
#py_install(c("anchorboosting", "lightgbm"), envname = "r-reticulate", pip = TRUE)
#py_install(c("scikit-learn", "joblib"), envname = "r-reticulate", pip = TRUE)


# Core apply_* functions
# Expect:
#   tr: train data (y,z factors),
#   te: test data  (y,z factors),
#   features: predictor names.

# 1) CTree 
apply_CTree_core <- function(tr, te, features,
                                  lambdas = c(10,10),
                                  min.leaf.size = 30,
                                  n.thresh = 40,
                                  eps.improvement=1e-2,
                                  prune = TRUE,
                                  K.prune = 10, relax.constraint=F, 
                                  compute.acc.sd = F) {
  # If we are to run CTree for multiple lambdas
  if(length(lambdas)>2){
    res=list(accuracy_train = NULL, accuracy_test = NULL, auc_train = NULL, auc_test = NULL, 
             accuracy_test_sd = NULL,
             rho.i = NULL, rho.ii = NULL, time=NULL, lambdas=matrix(0,nrow(lambdas),2), 
             included_vars  = list(), model = list(), rpart_object = list(), importance = list()
    )
    for(j in 1:nrow(lambdas)){
      res.tmp = apply_CTree_core(tr, te, features, lambdas=lambdas[j,], 
                                      min.leaf.size = min.leaf.size, n.thresh=n.thresh, 
                                      prune=prune, K.prune = K.prune,eps.improvement=eps.improvement, 
                                      relax.constraint=relax.constraint, compute.acc.sd = compute.acc.sd)
      res$accuracy_train = c(res$accuracy_train, res.tmp$accuracy_train)
      res$accuracy_test  = c(res$accuracy_test, res.tmp$accuracy_test)
      res$auc_train      = c(res$auc_train, res.tmp$auc_train)
      res$auc_test       = c(res$auc_test, res.tmp$auc_test)
      if(compute.acc.sd)  res$accuracy_test_sd  = c(res$accuracy_test_sd, res.tmp$accuracy_test_sd)
      res$rho.i          = c(res$rho.i , res.tmp$rho.i)
      res$rho.ii         = c(res$rho.ii, res.tmp$rho.ii)
      res$time           = c(res$time, res.tmp$time)
      res$lambdas[j,]         = res.tmp$lambdas
      res$included_vars[[j]]  = res.tmp$included_vars
      res$importance[[j]]     = res.tmp$importance
      res$model[[j]]          = res.tmp$model
      res$rpart_object[[j]]   = res.tmp$rpart_object
    }
    return(res)
  }
  if(lambdas[1]==0 & lambdas[2]==0) {
    min.leaf.size = 2 
    eps.improvement = 1e-5
  }
  tictoc::tic()
  CTree_tree <- CTree(
    data          = tr,
    target        = "y",
    environment   = "z",
    features      = features,
    lambda1       = lambdas[1],
    lambda2       = lambdas[2],
    min.leaf.size = min.leaf.size,
    n.thresh      = n.thresh,
    prune         = prune,
    K.prune       = K.prune,
    eps.improvement=eps.improvement,
    relax.constraint=relax.constraint
  )
  
  robust_rpart <- convert_to_rpart(CTree_tree, tr, target_col = "y")

  res_train <- assess_accuracy_tree(robust_rpart, tr)
  res_test  <- assess_accuracy_tree(robust_rpart, te, compute.sd = compute.acc.sd)
  
  tt = tictoc::toc(quiet=T)
  res=list(
    accuracy_train = res_train$accuracy,
    accuracy_test  = res_test$accuracy,
    auc_train      = res_train$auc,
    auc_test       = res_test$auc,
    rho.i          = res_train$rho.i,
    rho.ii         = res_train$rho.ii,
    included_vars  = features[which(res_train$var.inclusion)],
    model          = CTree_tree,
    rpart_object   = robust_rpart, 
    importance     = CTree_tree$importance,
    time = tt$toc-tt$tic,
    lambdas = lambdas
  )
  if(compute.acc.sd) res$accuracy_test_sd = res_test$accuracy.sd 
  
  res
}

# 1) CRF ---------------------------------------------------------------
apply_CRF_core <- function(tr, te, features,
                                lambdas = c(10, 10),
                                n.trees = 200,
                                sample.fraction = 1,
                                replace = TRUE,
                                sample_nodes = TRUE,
                                mtry = "rf",
                                min.leaf.size = 30,
                                n.thresh = 40,
                                eps.improvement = 1e-2,
                                compute.oob = TRUE,
                                importance = TRUE,
                                importance.type = c("both", "permutation", "gini"),
                                n.perm = 1,
                                parallel = TRUE,
                                n.cores = NULL,
                                verbose = FALSE,
                                compute_rho_full = FALSE,
                                compute_rho_oob  = FALSE,
                                rho_min_oob      = 5,
                                rho_parallel     = parallel,
                                rho_n.cores      = n.cores,
                                relax.constraint=FALSE,
                                seed = 1, 
                                compute.acc.sd = F) {
  
  importance.type <- match.arg(importance.type)
  
  # Multiple lambda pairs
  if (length(lambdas) > 2) {
    if (!is.matrix(lambdas) || ncol(lambdas) != 2) {
      stop("If providing multiple lambdas, pass a matrix with 2 columns: cbind(lambda1, lambda2).")
    }
    
    res <- list(
      accuracy_train = NULL, accuracy_test = NULL,
      auc_train = NULL, auc_test = NULL,
      accuracy_test_sd=NULL,
      rho.i = NULL, rho.ii = NULL,
      time = NULL,
      lambdas = matrix(0, nrow(lambdas), 2),
      model = list(),
      importance = list()
    )
    
    for (j in seq_len(nrow(lambdas))) {
      res.tmp <- apply_CRF_core(
        tr, te, features,
        lambdas = lambdas[j, ],
        n.trees = n.trees,
        sample.fraction = sample.fraction,
        replace = replace,
        sample_nodes = sample_nodes,
        mtry = mtry,
        min.leaf.size = min.leaf.size,
        n.thresh = n.thresh,
        eps.improvement = eps.improvement,
        compute.oob = compute.oob,
        importance = importance,
        importance.type = importance.type,
        n.perm = n.perm,
        parallel = parallel,
        n.cores = n.cores,
        verbose = verbose,
        compute_rho_full = compute_rho_full,
        compute_rho_oob  = compute_rho_oob,
        rho_min_oob      = rho_min_oob,
        rho_parallel     = rho_parallel,
        rho_n.cores      = rho_n.cores,
        relax.constraint= relax.constraint,
        compute.acc.sd=compute.acc.sd,
        seed = seed
      )
      
      res$accuracy_train <- c(res$accuracy_train, res.tmp$accuracy_train)
      res$accuracy_test  <- c(res$accuracy_test,  res.tmp$accuracy_test)
      res$auc_train      <- c(res$auc_train,      res.tmp$auc_train)
      res$auc_test       <- c(res$auc_test,       res.tmp$auc_test)
      res$rho.i          <- c(res$rho.i,          res.tmp$rho.i)
      res$rho.ii         <- c(res$rho.ii,         res.tmp$rho.ii)
      res$time           <- c(res$time,           res.tmp$time)
      if(compute.acc.sd)  res$accuracy_test_sd  = c(res$accuracy_test_sd, res.tmp$accuracy_test_sd)
      res$lambdas[j, ]    <- res.tmp$lambdas
      res$model[[j]]      <- res.tmp$model
      res$importance[[j]] <- res.tmp$importance
    }
    
    return(res)
  }
  
  # Timing (tictoc if available)
  use_tictoc <- requireNamespace("tictoc", quietly = TRUE)
  if (use_tictoc) tictoc::tic()
  tic <- proc.time()[["elapsed"]]
  
  # Fit RobustForest 
  fit <- CRF(
    data = tr,
    target = "y",
    environment = "z",
    features = features,
    n.trees = n.trees,
    sample.fraction = sample.fraction,
    replace = replace,
    sample_nodes = sample_nodes,
    mtry = mtry,
    lambda1 = lambdas[1],
    lambda2 = lambdas[2],
    min.leaf.size = min.leaf.size,
    eps.improvement = eps.improvement,
    n.thresh = n.thresh,
    seed = seed,
    compute.oob = compute.oob,
    importance = importance,
    importance.type = importance.type,
    n.perm = n.perm,
    parallel.importance = TRUE,
    parallel = parallel,
    n.cores = n.cores,
    compute_rho_full = compute_rho_full,
    compute_rho_oob  = compute_rho_oob,
    rho_min_oob      = rho_min_oob,
    rho_parallel     = rho_parallel,
    rho_n.cores      = rho_n.cores,
    relax.constraint= relax.constraint,
    verbose = verbose
  )
  
  # Scores = probability for "positive class" (second level of factor(y))
  # We follow your assess_accuracy_generic convention: the positive class is lev[2].
  prob_tr <- predict(fit, tr, type = "prob")
  prob_te <- predict(fit, te, type = "prob")
  
  if (ncol(prob_tr) != 2L || ncol(prob_te) != 2L) {
    stop("apply_CRF_core assumes binary classification for this particular setup (probability matrix must have 2 columns).")
  }
  
  y_score_tr <- prob_tr[, 2]
  y_score_te <- prob_te[, 2]
  
  res_train <- assess_accuracy_generic(y_true = tr$y, y_score = y_score_tr, z = tr$z)
  res_test  <- assess_accuracy_generic(y_true = te$y, y_score = y_score_te, z = te$z, compute.sd = compute.acc.sd)
  
  toc <- proc.time()[["elapsed"]]
  if (use_tictoc) {
    tt <- tictoc::toc(quiet = TRUE)
    runtime <- tt$toc - tt$tic
  } else {
    runtime <- toc - tic
  }
  rho_full_i  <- if (!is.null(fit$rho_full)) fit$rho_full$rho.i else NA_real_
  rho_full_ii <- if (!is.null(fit$rho_full)) fit$rho_full$rho.ii else NA_real_
  rho_oob_i   <- if (!is.null(fit$rho_oob))  fit$rho_oob$rho.i  else NA_real_
  rho_oob_ii  <- if (!is.null(fit$rho_oob))  fit$rho_oob$rho.ii else NA_real_
  
  res = list(
    accuracy_train = res_train$accuracy,
    accuracy_test  = res_test$accuracy,
    auc_train      = res_train$auc,
    auc_test       = res_test$auc,
    rho.i  = rho_full_i,
    rho.ii = rho_full_ii,
    rho.i_oob  = rho_oob_i,
    rho.ii_oob = rho_oob_ii,
    model          = fit,
    importance     = fit$importance,
    time           = runtime,
    lambdas        = lambdas
  )
  if(compute.acc.sd)  res$accuracy_test_sd  = res_test$accuracy.sd
  
  res
}


# 2) CART via rpart
apply_rpart_core <- function(tr, te, features, tr.with.z=NULL, 
                             compute.acc.sd = F) {
  tictoc::tic()
  fit <- rpart::rpart(y ~ ., data = tr, method = "class")
  
  res_train <- assess_accuracy_tree(fit, tr, tr.with.z)
  res_test  <- assess_accuracy_tree(fit, te, compute.sd = compute.acc.sd)
  included <- unique(fit$frame$var[fit$frame$var != "<leaf>"])
  tt = tictoc::toc(quiet=T)
  if(any(is.na(fit$variable.importance[features]))) {
    fit$variable.importance[features[which(is.na(fit$variable.importance[features]))]] = 0
    fit$variable.importance = fit$variable.importance[features]
  }
  res = list(
    accuracy_train = res_train$accuracy,
    accuracy_test  = res_test$accuracy,
    auc_train      = res_train$auc,
    auc_test       = res_test$auc,
    rho.i          = res_train$rho.i,
    rho.ii         = res_train$rho.ii,
    importance     = fit$variable.importance[features],
    included_vars  = included,
    model          = fit,
    time = tt$toc-tt$tic
  )
  if(compute.acc.sd)  res$accuracy_test_sd  = res_test$accuracy.sd
  
  res
}

# 3) Random forest (ranger)
apply_rf_core <- function(tr, te, features, min.leaf.size = 30, num.trees=200, 
                          compute.acc.sd = F) {
  if (!requireNamespace("ranger", quietly = TRUE)) {
    stop("Package 'ranger' is required for random forest.")
  }
  tictoc::tic()
  y_fac_train <- factor(tr$y)
  y_fac_test  <- factor(te$y, levels = levels(y_fac_train))
  rf_fit <- ranger::ranger(
    formula     = y_fac_train ~ .,
    data        = data.frame(y_fac_train = y_fac_train,
                             tr[, features, drop = FALSE]),
    probability = TRUE,
    num.trees=num.trees,
    splitrule      = "gini",
    importance  = "impurity"
  )
  lev <- levels(y_fac_train)
  
  pred_tr <- predict(rf_fit, data.frame(tr[, features, drop = FALSE]))$predictions
  score_tr <- pred_tr[, lev[2]]
  
  pred_te <- predict(rf_fit, data.frame(te[, features, drop = FALSE]))$predictions
  score_te <- pred_te[, lev[2]]
  
  res_train <- assess_accuracy_generic(y_fac_train, score_tr, tr$z)
  res_test  <- assess_accuracy_generic(y_fac_test,  score_te, te$z, compute.sd=compute.acc.sd)

  vi <- ranger::importance(rf_fit)
  tt = tictoc::toc(quiet=T)
  
  res = list(
    accuracy_train = res_train$accuracy,
    accuracy_test  = res_test$accuracy,
    auc_train      = res_train$auc,
    auc_test       = res_test$auc,
    rho.i          = res_train$rho.i,
    rho.ii         = res_train$rho.ii,
    importance  = vi,
    model          = rf_fit,
    time = tt$toc-tt$tic
  )
  if(compute.acc.sd)  res$accuracy_test_sd  = res_test$accuracy.sd
  
  res
}



# 4) ICP + rpart
apply_ICP_rpart_core <- function(tr, te, features, alpha = 0.05, 
                                 compute.acc.sd = F) {
  if (!requireNamespace("InvariantCausalPrediction", quietly = TRUE)) {
    stop("Package 'InvariantCausalPrediction' is required for ICP.")
  }
  if (!requireNamespace("tictoc", quietly = TRUE)) {
    stop("Package 'tictoc' is required for timing in this function.")
  }
  
  tictoc::tic()
  
  # -----------------------------
  # Prepare inputs for ICP
  # -----------------------------
  X <- tr[, features, drop = FALSE]
  Y <- tr$y
  ExpInd <- droplevels(tr$z)
  
  # -----------------------------
  # Preprocessing to avoid:
  # "Error in t.test.default(x, y) : data are essentially constant" (After domain-separation occurs)
  #
  # 1) Drop environments with too few samples
  # 2) Drop environments with constant Y (no within-env variation)
  # 3) Drop predictors that are constant within any environment
  # -----------------------------
  min_env_n <- 5L
  tol_sd <- 1e-8  # "essentially constant" guard
  
  # (1) Drop small environments
  env_n <- table(ExpInd)
  keep_env <- names(env_n)[env_n >= min_env_n]
  keep <- ExpInd %in% keep_env
  if (!all(keep)) {
    X <- X[keep, , drop = FALSE]
    Y <- Y[keep]
    ExpInd <- droplevels(ExpInd[keep])
  }
  
  # (2) Drop environments with constant Y
  bad_env <- character(0)
  for (e in levels(ExpInd)) {
    idx <- which(ExpInd == e)
    y_e <- Y[idx]
    # handle factor/character/numeric robustly
    if (length(unique(y_e)) < 2) {
      bad_env <- c(bad_env, e)
    } else if (is.numeric(y_e)) {
      v <- stats::var(y_e)
      if (!is.finite(v) || sqrt(v) < tol_sd) bad_env <- c(bad_env, e)
    }
  }
  if (length(bad_env) > 0) {
    keep <- !(ExpInd %in% bad_env)
    X <- X[keep, , drop = FALSE]
    Y <- Y[keep]
    ExpInd <- droplevels(ExpInd[keep])
  }
  
  # If we have < 2 environments left, ICP invariance testing is not meaningful.
  # In that case, skip ICP and use all features.
  use_icp <- length(levels(ExpInd)) >= 2 && nrow(X) >= 2
  
  # (3) Drop predictors constant within ANY environment
  # (these are frequent culprits for constant residuals / constant test inputs)
  keep_cols <- rep(TRUE, ncol(X))
  if (use_icp && ncol(X) > 0) {
    for (j in seq_len(ncol(X))) {
      xj <- X[, j]
      # overall constant
      if (stats::sd(xj, na.rm = TRUE) < tol_sd) {
        keep_cols[j] <- FALSE
        next
      }
      # constant in any environment
      const_any <- FALSE
      for (e in levels(ExpInd)) {
        idx <- which(ExpInd == e)
        if (length(idx) < 2) next
        if (stats::sd(xj[idx], na.rm = TRUE) < tol_sd) {
          const_any <- TRUE
          break
        }
      }
      if (const_any) keep_cols[j] <- FALSE
    }
  }
  
  X_icp <- X[, keep_cols, drop = FALSE]
  features_icp_space <- colnames(X_icp)
  
  # If all columns got removed, ICP can't run -> use all original features.
  if (use_icp && ncol(X_icp) == 0) use_icp <- FALSE
  
  # -----------------------------
  # Run ICP (catch constant-data error)
  # -----------------------------
  icp_features <- features  # default fallback
  
  if (use_icp) {
    icp_fit <- tryCatch(
      suppressWarnings(
        InvariantCausalPrediction::ICP(
          X = as.matrix(X_icp),
          Y = Y,
          ExpInd = ExpInd,
          alpha = alpha,
          showCompletion = FALSE,
          showAcceptedSets = FALSE
        )
      ),
      error = function(e) {
        # If ICP fails (including "essentially constant"), just fall back to all features.
        NULL
      }
    )
    
    if (!is.null(icp_fit)) {
      accepted <- icp_fit$acceptedSets
      if (length(accepted) > 0) {
        idx_sel <- sort(unique(unlist(accepted)))
        # idx_sel refers to columns of X_icp
        idx_sel <- idx_sel[idx_sel >= 1 & idx_sel <= ncol(X_icp)]
        if (length(idx_sel) > 0) {
          icp_features <- features_icp_space[idx_sel]
        }
      } else {
        # no accepted sets: your prior behavior is to use all features
        icp_features <- features
      }
    } else {
      # ICP failed
      icp_features <- features
    }
  }
  
  # -----------------------------
  # Fit rpart on selected features
  # -----------------------------
  form_icp <- as.formula(paste("y ~", paste(icp_features, collapse = " + ")))
  fit_icp  <- rpart::rpart(
    formula = form_icp,
    data    = tr[, c("y", icp_features), drop = FALSE],
    method  = "class"
  )
  
  res_train <- assess_accuracy_tree(fit_icp, tr)
  res_test  <- assess_accuracy_tree(fit_icp, te, compute.sd=compute.acc.sd)
  
  included <- unique(fit_icp$frame$var[fit_icp$frame$var != "<leaf>"])
  tt <- tictoc::toc(quiet = TRUE)
  
  # -----------------------------
  # Variable importance vector aligned to 'features'
  # (same output as before: named numeric, length = length(features))
  # -----------------------------
  vi <- rep(0, length(features))
  names(vi) <- features
  if (!is.null(fit_icp$variable.importance)) {
    vi[names(fit_icp$variable.importance)] <- fit_icp$variable.importance
  }
  
  res = list(
    accuracy_train = res_train$accuracy,
    accuracy_test  = res_test$accuracy,
    auc_train      = res_train$auc,
    auc_test       = res_test$auc,
    rho.i          = res_train$rho.i,
    rho.ii         = res_train$rho.ii,
    included_vars  = included,
    importance     = vi,
    model          = fit_icp,
    time           = tt$toc - tt$tic
  )
  if(compute.acc.sd)  res$accuracy_test_sd  = res_test$accuracy.sd
  
  res
}


# 5) ICP + random forest
apply_ICP_rf_core <- function(tr, te, features, alpha = 0.05, min.leaf.size = 30, num.trees=200, 
                              compute.acc.sd = F) {
  if (!requireNamespace("InvariantCausalPrediction", quietly = TRUE)) {
    stop("Package 'InvariantCausalPrediction' is required for ICP.")
  }
  if (!requireNamespace("ranger", quietly = TRUE)) {
    stop("Package 'ranger' is required for random forest.")
  }
  if (!requireNamespace("tictoc", quietly = TRUE)) {
    stop("Package 'tictoc' is required for timing in this function.")
  }
  
  tictoc::tic()
  
  # -----------------------------
  # Prepare inputs for ICP
  # -----------------------------
  X <- tr[, features, drop = FALSE]
  Y <- tr$y
  ExpInd <- droplevels(tr$z)
  
  # -----------------------------
  # Defensive preprocessing for ICP:
  # avoid "data are essentially constant"
  # -----------------------------
  min_env_n <- 5L
  tol_sd <- 1e-8
  
  # (1) Drop small environments
  env_n <- table(ExpInd)
  keep_env <- names(env_n)[env_n >= min_env_n]
  keep <- ExpInd %in% keep_env
  if (!all(keep)) {
    X <- X[keep, , drop = FALSE]
    Y <- Y[keep]
    ExpInd <- droplevels(ExpInd[keep])
  }
  
  # (2) Drop environments with constant Y
  bad_env <- character(0)
  for (e in levels(ExpInd)) {
    idx <- which(ExpInd == e)
    y_e <- Y[idx]
    if (length(unique(y_e)) < 2) {
      bad_env <- c(bad_env, e)
    } else if (is.numeric(y_e)) {
      v <- stats::var(y_e)
      if (!is.finite(v) || sqrt(v) < tol_sd) bad_env <- c(bad_env, e)
    }
  }
  if (length(bad_env) > 0) {
    keep <- !(ExpInd %in% bad_env)
    X <- X[keep, , drop = FALSE]
    Y <- Y[keep]
    ExpInd <- droplevels(ExpInd[keep])
  }
  
  # If fewer than 2 environments remain, ICP is not meaningful -> skip ICP
  use_icp <- length(levels(ExpInd)) >= 2 && nrow(X) >= 2
  
  # (3) Drop predictors constant overall or constant within ANY environment (ICP only)
  keep_cols <- rep(TRUE, ncol(X))
  if (use_icp && ncol(X) > 0) {
    for (j in seq_len(ncol(X))) {
      xj <- X[, j]
      
      # overall constant
      if (stats::sd(xj, na.rm = TRUE) < tol_sd) {
        keep_cols[j] <- FALSE
        next
      }
      
      # constant within any environment
      const_any <- FALSE
      for (e in levels(ExpInd)) {
        idx <- which(ExpInd == e)
        if (length(idx) < 2) next
        if (stats::sd(xj[idx], na.rm = TRUE) < tol_sd) {
          const_any <- TRUE
          break
        }
      }
      if (const_any) keep_cols[j] <- FALSE
    }
  }
  
  X_icp <- X[, keep_cols, drop = FALSE]
  features_icp_space <- colnames(X_icp)
  
  if (use_icp && ncol(X_icp) == 0) use_icp <- FALSE
  
  # -----------------------------
  # Run ICP (catch constant-data errors)
  # -----------------------------
  icp_features <- features  # fallback
  
  if (use_icp) {
    icp_fit <- tryCatch(
      suppressWarnings(
        InvariantCausalPrediction::ICP(
          X = as.matrix(X_icp),
          Y = Y,
          ExpInd = ExpInd,
          alpha = alpha,
          showCompletion = FALSE,
          showAcceptedSets = FALSE
        )
      ),
      error = function(e) NULL
    )
    
    if (!is.null(icp_fit)) {
      accepted <- icp_fit$acceptedSets
      if (length(accepted) > 0) {
        idx_sel <- sort(unique(unlist(accepted)))
        idx_sel <- idx_sel[idx_sel >= 1 & idx_sel <= ncol(X_icp)]
        if (length(idx_sel) > 0) {
          icp_features <- features_icp_space[idx_sel]
        }
      } else {
        icp_features <- features
      }
    } else {
      icp_features <- features
    }
  }
  
  # -----------------------------
  # Fit ranger RF on selected features
  # -----------------------------
  y_fac_train <- factor(tr$y)
  y_fac_test  <- factor(te$y, levels = levels(y_fac_train))
  
  rf_fit <- ranger::ranger(
    formula        = y_fac_train ~ .,
    data           = data.frame(y_fac_train = y_fac_train,
                                tr[, icp_features, drop = FALSE]),
    probability    = TRUE,
    classification = T,
    min.node.size  = min.leaf.size,
    num.trees      = num.trees,
    importance     = "impurity"
  )
  
  lev <- levels(y_fac_train)
  
  pred_tr <- predict(rf_fit, data.frame(tr[, icp_features, drop = FALSE]))$predictions
  score_tr <- pred_tr[, lev[2]]
  
  pred_te <- predict(rf_fit, data.frame(te[, icp_features, drop = FALSE]))$predictions
  score_te <- pred_te[, lev[2]]
  
  # -----------------------------
  # Importance vector aligned to 'features'
  # -----------------------------
  vi <- rep(0, length(features))
  names(vi) <- features
  
  vi_sel <- ranger::importance(rf_fit)
  if (!is.null(vi_sel) && length(vi_sel) > 0) {
    vi[names(vi_sel)] <- vi_sel
  }
  
  # -----------------------------
  # Assess performance (generic: rhos NA)
  # -----------------------------
  res_train <- assess_accuracy_generic(y_fac_train, score_tr, tr$z)
  res_test  <- assess_accuracy_generic(y_fac_test,  score_te, te$z, compute.sd=compute.acc.sd)
  
  tt <- tictoc::toc(quiet = TRUE)
  
  res = list(
    accuracy_train = res_train$accuracy,
    accuracy_test  = res_test$accuracy,
    auc_train      = res_train$auc,
    auc_test       = res_test$auc,
    rho.i          = res_train$rho.i,
    rho.ii         = res_train$rho.ii,
    importance     = vi,
    model          = rf_fit,
    time           = tt$toc - tt$tic
  )
  if(compute.acc.sd)  res$accuracy_test_sd  = res_test$accuracy.sd
  
  res
}

# 6) Anchor regression
apply_anchor_core <- function(tr, te, features, gamma = 2, 
                              compute.acc.sd = F) {
  # Anchor regression: fit on train only (X + y_num, z as anchor, gamma via CV),
  # then for prediction I re-create the anchor-transformed features jointly for train+test,
  # drop y_num from the design, apply the stored coefficients manually, and pass the
  # linear predictor through a logistic link to get probabilities.
  
  if (!requireNamespace("AnchorRegression", quietly = TRUE)) {
    stop("Package 'AnchorRegression' is required for anchor regression.")
  }
  
  # Ensure binary factor y with same levels
  if (!is.factor(tr$y)) tr$y <- factor(tr$y)
  if (!is.factor(te$y)) te$y <- factor(te$y, levels = levels(tr$y))
  y_levels <- levels(tr$y)
  if (length(y_levels) != 2L) {
    stop("apply_anchor_core assumes binary classification (2 levels in y).")
  }
  tictoc::tic()
  # Numeric response for training (0/1)
  tr$y_num <- as.numeric(tr$y == y_levels[2])  # positive class = level[2]
  
  # Design for fitting: features + y_num
  X_tr <- tr[, features, drop = FALSE]
  non_list <- !vapply(X_tr, is.list, logical(1))
  X_tr <- X_tr[, non_list, drop = FALSE]
  
  x_train <- cbind(X_tr, y_num = tr$y_num)
  anchor_train <- data.frame(z = tr$z)
  
  tictoc::tic()
  # Fit anchor regression on TRAIN only
  anchor_fit <- AnchorRegression::anchor_regression(
    x               = x_train,
    anchor          = anchor_train,
    gamma           = gamma,
    target_variable = "y_num",
    lambda          = "CV"
  )
  tt = tictoc::toc(quiet=T)
  # Predict on TRAIN+TEST jointly (no test y leakage)
  preds <- predict_anchor_train_test(anchor_fit, tr, te,
                                     features = colnames(X_tr),
                                     gamma = gamma)
  p_train <- preds$prob_train
  p_test  <- preds$prob_test
  
  # Evaluate with your generic metric function
  res_train <- assess_accuracy_generic(tr$y, p_train, tr$z)
  res_test  <- assess_accuracy_generic(te$y, p_test, te$z, compute.sd = compute.acc.sd)
  
  # Extract non-zero coefficients as "included" variables (excluding intercept,y_num)
  nm  <- anchor_fit$names
  coe <- anchor_fit$coeff
  excluded <- "Intercept"
  if ("y_num" %in% nm) excluded <- c(excluded, "y_num")
  
  included <- nm[(!(nm %in% excluded)) & (coe != 0)]
  tt = tictoc::toc(quiet=T)
  
  res = list(
    accuracy_train = res_train$accuracy,
    accuracy_test  = res_test$accuracy,
    auc_train      = res_train$auc,
    auc_test       = res_test$auc,
    rho.i          = res_train$rho.i,
    rho.ii         = res_train$rho.ii,
    included_vars  = included,
    model          = anchor_fit,
    time = tt$toc-tt$tic
  )
  if(compute.acc.sd)  res$accuracy_test_sd  = res_test$accuracy.sd
  
  res
}

apply_anchor_boost_core <- function(tr, te, features, lam = 5, 
                                   python_module_dir = "functions/other_methods",
                                   python_module = "invariant_methods", 
                                   compute.acc.sd = F) {
  # If we are to run IRF for multiple lambdas
  if(length(lam)>1){
    res=list(accuracy_train = NULL, accuracy_test = NULL, auc_train = NULL, auc_test = NULL, accuracy_test_sd = NULL,
             rho.i = NULL, rho.ii = NULL, time=NULL, lambda=NULL, included_vars  = list(), model = list()
    )
    for(j in 1:length(lam)){
      res.tmp = apply_anchor_boost_core(tr, te, features, lam=lam[j], 
                               python_module_dir = python_module_dir, python_module = python_module,compute.acc.sd =compute.acc.sd)
      res$accuracy_train = c(res$accuracy_train, res.tmp$accuracy_train)
      res$accuracy_test  = c(res$accuracy_test, res.tmp$accuracy_test)
      if(compute.acc.sd)  res$accuracy_test_sd  = c(res$accuracy_test_sd, res.tmp$accuracy_test_sd)
      res$auc_train      = c(res$auc_train, res.tmp$auc_train)
      res$auc_test       = c(res$auc_test, res.tmp$auc_test)
      res$rho.i          = c(res$rho.i , res.tmp$rho.i)
      res$rho.ii         = c(res$rho.ii, res.tmp$rho.ii)
      res$time           = c(res$time, res.tmp$time)
      res$lambda         = c(res$lambda, res.tmp$lambda)
      res$importance[[j]]  = res.tmp$importance
      res$model[[j]]          = res.tmp$model
    }
    return(res)
  }
  
  tictoc::tic()
  scores <- run_python_anchorboost(tr, te, features,
                                   python_module_dir = python_module_dir,
                                   python_module = python_module,
                                   y_name = "y", z_name = "z", lam=lam)
  
  y_fac_train <- factor(tr$y)
  y_fac_test  <- factor(te$y, levels = levels(y_fac_train))
  
  res_train <- assess_accuracy_generic(y_fac_train, scores$train, tr$z)
  res_test  <- assess_accuracy_generic(y_fac_test,  scores$test,  te$z, compute.sd=compute.acc.sd)
  tt = tictoc::toc(quiet=T)
  res = list(
    accuracy_train = res_train$accuracy,
    accuracy_test  = res_test$accuracy,
    auc_train      = res_train$auc,
    auc_test       = res_test$auc,
    rho.i          = res_train$rho.i,   # stays NA for non-tree
    rho.ii         = res_train$rho.ii,
    lambda =lam, 
    time = tt$toc-tt$tic,
    importance  = scores$importance,              # you could later inspect feature importance
    model          = NULL               # Not saving the python function
  )
  if(compute.acc.sd)  res$accuracy_test_sd  = res_test$accuracy.sd
  
  res
}



# 8) IRF via Python
apply_IRF_core <- function(tr, te, features,
                           lambda = 5,
                           python_module_dir = "functions/other_methods",
                           python_module = "invariant_methods", 
                           compute.acc.sd = F) {
  # If we are to run IRF for multiple lambdas
  if(length(lambda)>1){
    res=list(accuracy_train = NULL, accuracy_test = NULL, auc_train = NULL, auc_test = NULL, accuracy_test_sd=NULL,
             rho.i = NULL, rho.ii = NULL, time=NULL, lambda=NULL, included_vars  = list(), model = list()
    )
    for(j in 1:length(lambda)){
      res.tmp = apply_IRF_core(tr, te, features, lambda=lambda[j], 
                           python_module_dir = python_module_dir, python_module = python_module,compute.acc.sd =compute.acc.sd)
      res$accuracy_train = c(res$accuracy_train, res.tmp$accuracy_train)
      res$accuracy_test  = c(res$accuracy_test, res.tmp$accuracy_test)
      if(compute.acc.sd)  res$accuracy_test_sd  = c(res$accuracy_test_sd,res.tmp$accuracy_test_sd)
      res$auc_train      = c(res$auc_train, res.tmp$auc_train)
      res$auc_test       = c(res$auc_test, res.tmp$auc_test)
      res$rho.i          = c(res$rho.i , res.tmp$rho.i)
      res$rho.ii         = c(res$rho.ii, res.tmp$rho.ii)
      res$time           = c(res$time, res.tmp$time)
      res$lambda         = c(res$lambda, res.tmp$lambda)
      res$importance[[j]]  = res.tmp$importance
      res$model[[j]]          = res.tmp$model
    }
    return(res)
  }
  tictoc::tic()
  scores <- run_python_irf(
    dat_train        = tr,
    dat_test         = te,
    features         = features,
    lambda           = lambda,
    python_module_dir = python_module_dir,
    python_module     = python_module,
    y_name           = "y",
    z_name           = "z"
  )
  
  y_fac_train <- factor(tr$y)
  y_fac_test  <- factor(te$y, levels = levels(y_fac_train))
  
  res_train <- assess_accuracy_generic(y_fac_train, scores$train, tr$z)
  res_test  <- assess_accuracy_generic(y_fac_test,  scores$test,  te$z, compute.sd=compute.acc.sd)
  tt = tictoc::toc(quiet=T)

  res = list(
    accuracy_train = res_train$accuracy,
    accuracy_test  = res_test$accuracy,
    auc_train      = res_train$auc,
    auc_test       = res_test$auc,
    rho.i          = res_train$rho.i,
    rho.ii         = res_train$rho.ii,
    time           = tt$toc-tt$tic,
    lambda.        = lambda,
    importance     = scores$importance, 
    model          = NULL # Not saving python model
  )
  if(compute.acc.sd)  res$accuracy_test_sd  = res_test$accuracy.sd
  
  res
}

# ==========================================================
# Python bridge helpers (Anchor boosting/IRF)
# ==========================================================

run_python_anchorboost <- function(dat_train,
                                   dat_test,
                                   features,
                                   python_module_dir,  # <- pass the dir, not just the name
                                   python_module,
                                   y_name = "y",
                                   z_name = "z", 
                                   lam=5.0) {
  if (!requireNamespace("reticulate", quietly = TRUE)) {
    stop("Package 'reticulate' is required for Python-based methods.")
  }
  
  py <- reticulate::import_from_path(python_module,
                                     path = python_module_dir)
  
  if (!reticulate::py_has_attr(py, "anchorboost_predict_both")) {
    stop("Function 'anchorboost_predict_both' not found in invariant_methods.")
  }
  fn <- py$anchorboost_predict_both
  
  X_train <- as.matrix(dat_train[, features, drop = FALSE])
  X_test  <- as.matrix(dat_test[,  features, drop = FALSE])
  
  y_fac   <- factor(dat_train[[y_name]])
  y_train <- as.integer(y_fac) - 1L
  z_train <- as.character(dat_train[[z_name]])
 
  out <- fn(X_train, y_train, z_train, X_test, lam=lam)
  importance = out$feature_importance
  names(importance) = features
  list(
    train = as.numeric(out$train),
    test  = as.numeric(out$test), 
    importance = importance
  )
}


run_python_irf <- function(dat_train,
                           dat_test,
                           features,
                           python_module_dir = "functions/other_methods",
                           python_module = "invariant_methods",
                           y_name = "y",
                           z_name = "z",
                           lambda = 5,
                           n_estimators = 100,
                           max_depth = 5,
                           min_sample_periods = 10,
                           min_impurity_decrease = 0,
                           random_state = 0) {
  
  if (!requireNamespace("reticulate", quietly = TRUE)) {
    stop("Package 'reticulate' is required for Python-based methods.")
  }
  
  py_mod <- reticulate::import_from_path(python_module, path = python_module_dir)
  
  if (!reticulate::py_has_attr(py_mod, "irf_predict_both")) {
    stop("Function 'irf_predict_both' not found in Python module '", python_module, "'.")
  }
  fn <- py_mod$irf_predict_both
  
  X_train <- as.matrix(dat_train[, features, drop = FALSE])
  X_test  <- as.matrix(dat_test[,  features, drop = FALSE])
  
  y_fac   <- factor(dat_train[[y_name]])
  y_train <- as.integer(y_fac) - 1L
  z_train <- as.character(dat_train[[z_name]])
  
  out <- fn(X_train, y_train, z_train, X_test,
            n_estimators        = as.integer(n_estimators),
            max_depth           = as.integer(max_depth),
            min_sample_periods  = as.integer(min_sample_periods),
            invariance_penalty  = lambda,
            min_impurity_decrease = min_impurity_decrease,
            random_state        = as.integer(random_state))
  importance = unlist(out$feature_importance)
  names(importance) = features

  list(
    train      = as.numeric(out$train),
    test       = as.numeric(out$test),
    importance = importance
  )
}



# Accuracy / AUC helpers 

# For rpart-like objects (CTree converted to rpart, rpart, ICP+rpart)
assess_accuracy_tree <- function(fit, dat, dat.z=NULL, compute.sd=F) {
  res <- list()
  if(!is.null(dat.z)) n_domains <- length(unique(dat.z$z))
  else n_domains <- length(unique(dat$z))
  
  pred_class <- as.numeric(paste(predict(fit, dat, type = "class")))
  res$accuracy <- mean(pred_class == dat$y)
  if(compute.sd) res$accuracy.sd <- sd(pred_class == dat$y)
  
  pred_prob <- predict(fit, dat, type = "prob")[, 2]
  pred_rocr <- ROCR::prediction(pred_prob, dat$y)
  auc_val   <- ROCR::performance(pred_rocr, measure = "auc")
  res$auc   <- auc_val@y.values[[1]]
  if(!is.null(dat.z)) dat = dat.z
  if (n_domains > 1) {
    res$rho.i  <- total_mutual_information(fit, dat, "y", "z")
    res$rho.ii <- total_entropy_change(fit, dat, "z")
  } else {
    res$rho.i  <- 0
    res$rho.ii <- 0
  }
  
  included.vars <- sort(unique(rownames(fit$splits)))
  if(!is.null(dat.z)) vars <- setdiff(names(dat.z), c("y", "z"))
  else vars <- setdiff(names(dat), c("y", "z"))
  res$var.inclusion <- vars %in% included.vars
  
  res
}

# For generic methods that output scores (probabilities or scores for positive class)
assess_accuracy_generic <- function(y_true, y_score, z, compute.sd=F) {
  res <- list()
  y_true <- factor(y_true)
  lev <- levels(y_true)
  if (length(lev) != 2L) {
    stop("assess_accuracy_generic assumes a binary response.")
  }
  
  y_pred <- ifelse(y_score >= 0.5, lev[2], lev[1])
  y_pred <- factor(y_pred, levels = lev)
  res$accuracy <- mean(y_pred == y_true)
  if(compute.sd) res$accuracy.sd <- sd(y_pred == y_true)
  
  pred_obj <- ROCR::prediction(y_score, y_true)
  auc_obj  <- ROCR::performance(pred_obj, measure = "auc")
  res$auc  <- auc_obj@y.values[[1]]
  
  # For non-tree methods, we don’t define leaf-based invariance here
  res$rho.i         <- NA_real_
  res$rho.ii        <- NA_real_
  res$var.inclusion <- NULL
  
  res
}

# -------------------------------------------------------------------
# Helper: predict for anchor_regression on TRAIN+TEST jointly
# using only features + anchor (no test y)
# -------------------------------------------------------------------
predict_anchor_train_test <- function(anchor_fit, tr, te, features, gamma) {
  # Features: drop list-columns to keep things sane
  X_tr <- tr[, features, drop = FALSE]
  non_list <- !vapply(X_tr, is.list, logical(1))
  X_tr <- X_tr[, non_list, drop = FALSE]
  
  X_te <- te[, colnames(X_tr), drop = FALSE]
  
  X_all_feat <- rbind(X_tr, X_te)
  n_all <- nrow(X_all_feat)
  
  # Anchors: treat z as categorical
  z_tr  <- tr$z
  z_te  <- te$z
  z_all <- factor(c(as.character(z_tr), as.character(z_te)))
  
  # Matrix of features
  x_mat <- as.matrix(X_all_feat)
  
  # Global mean μ = E[X]
  mu <- colMeans(x_mat)
  mu_mat <- matrix(rep(mu, each = n_all), nrow = n_all)
  colnames(mu_mat) <- colnames(x_mat)
  
  # Group means E[X | Z = z] for each level of z
  E_X_given_Z <- matrix(0, nrow = n_all, ncol = ncol(x_mat))
  colnames(E_X_given_Z) <- colnames(x_mat)
  
  for (lvl in levels(z_all)) {
    idx <- which(z_all == lvl)
    if (length(idx) > 0L) {
      mu_z <- colMeans(x_mat[idx, , drop = FALSE])
      E_X_given_Z[idx, ] <- matrix(rep(mu_z, each = length(idx)),
                                   nrow = length(idx))
    }
  }
  
  # Anchor transform on TRAIN+TEST together:
  # \tilde X = μ + (X - E[X|Z]) + sqrt(γ) (E[X|Z] - μ)
  anchor_data_all <- mu_mat +
    (x_mat - E_X_given_Z) +
    sqrt(gamma) * (E_X_given_Z - mu_mat)
  colnames(anchor_data_all) <- colnames(x_mat)
  
  # Linear predictor from anchor_fit$coeff / $names
  beta <- anchor_fit$coeff
  nm   <- anchor_fit$names
  if (length(beta) != length(nm)) {
    stop("anchor_fit$coeff and anchor_fit$names have different lengths.")
  }
  
  beta0 <- beta[1L]
  
  if ("y_num" %in% nm) {
    # Case 1: target variable appears in names (drop it)
    idx_target <- which(nm == "y_num")[1L]
    feat_idx   <- setdiff(seq_along(beta), c(1L, idx_target))
  } else {
    # Case 2: target variable was not included in names (usual CRAN behavior)
    feat_idx   <- seq_along(beta)[-1L]  # drop only the intercept
  }
  
  feat_names <- nm[feat_idx]
  beta_x     <- beta[feat_idx]
  
  
  if (!all(feat_names %in% colnames(anchor_data_all))) {
    missing <- setdiff(feat_names, colnames(anchor_data_all))
    stop("New data is missing feature(s) used in anchor_regression: ",
         paste(missing, collapse = ", "))
  }
  
  X_design_all <- anchor_data_all[, feat_names, drop = FALSE]
  
  eta_all <- as.numeric(beta0 + X_design_all %*% beta_x)
  
  # Split back into train/test and apply logistic link
  n_tr <- nrow(tr)
  n_te <- nrow(te)
  
  eta_train <- eta_all[1:n_tr]
  eta_test  <- eta_all[(n_tr + 1):(n_tr + n_te)]
  
  p_train <- 1 / (1 + exp(-eta_train))
  p_test  <- 1 / (1 + exp(-eta_test))
  
  list(
    prob_train = p_train,
    prob_test  = p_test
  )
}
