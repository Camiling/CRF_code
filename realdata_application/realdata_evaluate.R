source('functions/help_funcs/convert_to_rpart.R')
source("functions/other_methods/apply_othermethods.R")


# Apply methods to benchmark data sets and evaluate performance

benchmark_evaluate <- function(dat_train, dat_test, y, z,lambdas = c(10,10), min.leaf.size = 30,
                               num.trees=200, gamma.anchor=5, lambda.irf=5, CRF_sample_nodes=F,
                               eps.improvement=1e-2, n.thresh=20, n.cores=10,
                               include_CTree = FALSE, include_CRF = TRUE, include_rpart = FALSE, include_rf = TRUE,
                               include_icp_rpart = FALSE, include_icp_rf = TRUE, include_anchor = FALSE,
                               include_anchor_boost = TRUE, include_irf = TRUE, prune=FALSE, relax.constraint=TRUE,
                               python_module = "invariant_methods") {
  # get data to the correct format
  # ---- Basic preparation: rename to y/z, align factors, define features ----
  tr <- dat_train
  te <- dat_test
  
  # Rename y and z
  names(tr)[names(tr) == y] <- "y"
  names(tr)[names(tr) == z] <- "z"
  names(te)[names(te) == y] <- "y"
  names(te)[names(te) == z] <- "z"
  
  # Ensure y and z are factors with aligned levels
  tr$y <- as.factor(tr$y)
  te$y <- factor(te$y, levels = levels(tr$y))
  
  tr$z <- factor(tr$z, levels=c(unique(te$z), unique(tr$z)))
  te$z <- factor(te$z, levels = union(levels(tr$z), levels(as.factor(te$z))))
  
  # Features = all non-(y,z) columns
  features <- setdiff(colnames(tr), c("y", "z"))
  
  res <- list()
  cat('Starting analysis... \n')
  if (include_CTree) {
    r <- apply_CTree_core(tr, te, features,
                               lambdas = lambdas, n.thresh=n.thresh, eps.improvement =eps.improvement, 
                               min.leaf.size = min.leaf.size, prune=prune,relax.constraint=relax.constraint, 
                               compute.acc.sd = T)
    cat('CTree: done in ',  r$time , ' seconds \n')
    res$CTree_accuracy_train <- r$accuracy_train
    res$CTree_accuracy_test  <- r$accuracy_test
    res$CTree_accuracy_test_sd  <- r$accuracy_test_sd
    res$CTree_auc_train      <- r$auc_train
    res$CTree_auc_test       <- r$auc_test
    res$CTree_rho.i          <- r$rho.i
    res$CTree_rho.ii         <- r$rho.ii
    res$CTree_included_vars  <- r$included_vars
    res$CTree_importance     <- r$importance
    res$CTree_tree           <- r$model
    res$CTree_lambdas        <- r$lambdas
  }
  if (include_CRF) {
    r <- apply_CRF_core(tr, te, features,
                               lambdas = lambdas,
                               min.leaf.size = min.leaf.size, 
                               n.trees = num.trees, sample_nodes = CRF_sample_nodes,
                               mtry = "rf", n.thresh = n.thresh, eps.improvement = eps.improvement,
                               compute.oob = F, importance = TRUE, importance.type = 'gini',
                               n.perm = 1, parallel = TRUE, n.cores = n.cores,
                               compute_rho_full = TRUE, compute_rho_oob  = F,relax.constraint=relax.constraint,
                               verbose = FALSE,seed = 1, 
                               compute.acc.sd = T)
    cat('CRF: done in ',  r$time , ' seconds \n')
    res$CRF_accuracy_train <- r$accuracy_train
    res$CRF_accuracy_test  <- r$accuracy_test
    res$CRF_accuracy_test_sd  <- r$accuracy_test_sd
    res$CRF_auc_train      <- r$auc_train
    res$CRF_auc_test       <- r$auc_test
    res$CRF_rho.i          <- r$rho.i
    res$CRF_rho.ii         <- r$rho.ii
    res$CRF_var_importance <- r$importance
    res$CRF_model          <- r$model
    res$CRF_lambdas        <- r$lambdas
  }
  if (include_rpart) {
    r <- apply_rpart_core(tr, te, features, 
                          compute.acc.sd = T)
    cat('Rpart: done in ',  r$time, ' seconds \n')
    res$rpart_accuracy_train <- r$accuracy_train
    res$rpart_accuracy_test  <- r$accuracy_test
    res$rpart_accuracy_test_sd  <- r$accuracy_test_sd
    res$rpart_auc_train      <- r$auc_train
    res$rpart_auc_test       <- r$auc_test
    res$rpart_rho.i          <- r$rho.i
    res$rpart_rho.ii         <- r$rho.ii
    res$rpart_included_vars  <- r$included_vars
    res$rpart_tree           <- r$model
  }
  
  if (include_rf) {
    r <- apply_rf_core(tr, te, features, min.leaf.size = min.leaf.size, num.trees=num.trees, 
                       compute.acc.sd = T)
    cat('Random Forest: done in ',  r$time , ' seconds \n')
    res$rf_accuracy_train <- r$accuracy_train
    res$rf_accuracy_test  <- r$accuracy_test
    res$rf_accuracy_test_sd  <- r$accuracy_test_sd
    res$rf_auc_train      <- r$auc_train
    res$rf_auc_test       <- r$auc_test
    res$rf_rho.i          <- r$rho.i
    res$rf_rho.ii         <- r$rho.ii
    res$rf_var_importance  <- r$importance
    res$rf_model          <- r$model
  }
  if (include_icp_rpart) {
    r <- apply_ICP_rpart_core(tr, te, features, 
                              compute.acc.sd = T)
    cat('ICP + rpart: done in ',  r$time , ' seconds \n')
    res$icp_rpart_accuracy_train <- r$accuracy_train
    res$icp_rpart_accuracy_test  <- r$accuracy_test
    res$icp_rpart_accuracy_test_sd  <- r$accuracy_test_sd
    res$icp_rpart_auc_train      <- r$auc_train
    res$icp_rpart_auc_test       <- r$auc_test
    res$icp_rpart_rho.i          <- r$rho.i
    res$icp_rpart_rho.ii         <- r$rho.ii
    res$icp_rpart_included_vars  <- r$included_vars
    res$icp_rpart_tree           <- r$model
  }
  
  if (include_icp_rf) {
    r <- apply_ICP_rf_core(tr, te, features, min.leaf.size = min.leaf.size, num.trees=num.trees, 
                           compute.acc.sd = T)
    cat('ICP + Random Forest: done in ',  r$time , ' seconds \n')
    res$icp_rf_accuracy_train <- r$accuracy_train
    res$icp_rf_accuracy_test  <- r$accuracy_test
    res$icp_rf_accuracy_test_sd  <- r$accuracy_test_sd
    res$icp_rf_auc_train      <- r$auc_train
    res$icp_rf_auc_test       <- r$auc_test
    res$icp_rf_rho.i          <- r$rho.i
    res$icp_rf_rho.ii         <- r$rho.ii
    res$icp_rf_var_importance  <- r$importance
    res$icp_rf_model          <- r$model
  }
  
  if (include_anchor) {
    r <- apply_anchor_core(tr, te, features, 
                           compute.acc.sd = T)
    cat('Anchor regression: done in ',  r$time , ' seconds \n')
    res$anchor_accuracy_train <- r$accuracy_train
    res$anchor_accuracy_test  <- r$accuracy_test
    res$anchor_accuracy_test_sd  <- r$accuracy_test_sd
    res$anchor_auc_train      <- r$auc_train
    res$anchor_auc_test       <- r$auc_test
    res$anchor_rho.i          <- r$rho.i
    res$anchor_rho.ii         <- r$rho.ii
    res$anchor_included_vars  <- r$included_vars
    res$anchor_model          <- r$model
  }
  if (include_anchor_boost) {
    r <- apply_anchor_boost_core(tr, te, features, 
                                 python_module = python_module, lam=gamma.anchor, 
                                 compute.acc.sd = T)
    cat('Anchor boosting: done in ',  r$time , ' seconds \n')
    res$anchor_boost_accuracy_train <- r$accuracy_train
    res$anchor_boost_accuracy_test  <- r$accuracy_test
    res$anchor_boost_accuracy_test_sd  <- r$accuracy_test_sd
    res$anchor_boost_auc_train      <- r$auc_train
    res$anchor_boost_auc_test       <- r$auc_test
    res$anchor_boost_rho.i          <- r$rho.i
    res$anchor_boost_rho.ii         <- r$rho.ii
    res$anchor_boost_var_importance <- r$importance
    res$anchor_boost_model          <- r$model
    res$anchor_boost_gamma          <- r$lambda   
  }

  
  if (include_irf) {
    r <- apply_IRF_core(tr, te, features, lambda=lambda.irf,
                        python_module = python_module, 
                        compute.acc.sd = T)
    cat('IRF: done in ',  r$time , ' seconds \n')
    res$irf_accuracy_train <- r$accuracy_train
    res$irf_accuracy_test  <- r$accuracy_test
    res$irf_accuracy_test_sd  <- r$accuracy_test_sd
    res$irf_auc_train      <- r$auc_train
    res$irf_auc_test       <- r$auc_test
    res$irf_rho.i          <- r$rho.i
    res$irf_rho.ii         <- r$rho.ii
    res$irf_lambda         <- r$lambda
    res$irf_var_importance <- r$importance
  }
  
  res
}

print_method_results <- function(res,te=NULL,
                                 include_CTree    = FALSE,
                                 include_CRF  = TRUE,
                                 include_rpart     = FALSE,
                                 include_rf        = TRUE,
                                 include_icp       = TRUE,
                                 include_icp_rpart = FALSE,
                                 include_icp_rf    = TRUE,
                                 include_anchor    = TRUE,
                                 include_anchor_boost = TRUE,
                                 include_irf       = TRUE, 
                                 percentage=T, 
                                 print.acc.sd = T) {
  
  # ----------------------------
  # stats helpers (accept scalar OR replicate vectors)
  # ----------------------------
  safe_mean <- function(x) {
    if (is.null(x) || length(x) == 0 || all(is.na(x))) return(NA_real_)
    mean(x, na.rm = TRUE)
  }
  safe_sd <- function(x) {
    if (is.null(x) || length(x) == 0) return(NA_real_)
    s = sd(x, na.rm = TRUE)
    if (is.na(s)) 0 else s
  }
  safe_n <- function(x) {
    if (is.null(x) || length(x) == 0) return(0L)
    sum(!is.na(x))
  }
  safe_se <- function(x) {
    n = safe_n(x)
    if (n <= 1) return(0)
    safe_sd(x) / sqrt(n)
  }
  get_at <- function(x, j) {
    if (is.null(x)) return(NA_real_)
    if (length(x) == 1) return(x)
    x[j]
  }
  if(!is.null(te)){
    n_scale = sqrt(nrow(te))
  }
  else {
    n_scale=1
  }
  # ----------------------------
  # normalize lambdas so you ALWAYS have an n x 2 matrix when possible
  # - handles the common bug: matrix(c(10,10), nrow=2) -> 2x1, but really means 1x2
  # ----------------------------
  normalize_lambdas <- function(L) {
    if (is.null(L)) return(NULL)
    if (is.data.frame(L)) L = as.matrix(L)
    
    if (is.vector(L)) {
      if (length(L) == 2) return(matrix(L, nrow = 1, byrow = TRUE))
      stop("lambdas vector must have length 2.")
    }
    
    if (!is.matrix(L)) stop("lambdas must be a matrix/data.frame or length-2 vector.")
    
    # If already has >=2 cols, just take first two
    if (ncol(L) >= 2) return(L[, 1:2, drop = FALSE])
    
    # If it's 2x1, interpret as ONE lambda-pair (lambda1, lambda2)
    if (ncol(L) == 1 && nrow(L) == 2) return(matrix(c(L[1,1], L[2,1]), nrow = 1, byrow = TRUE))
    
    # If it's kx1 with even k, interpret as k/2 pairs stacked
    if (ncol(L) == 1 && (nrow(L) %% 2 == 0)) {
      v = as.vector(L)
      return(matrix(v, ncol = 2, byrow = TRUE))
    }
    
    # Otherwise: give up gracefully but keep 1 column (lambda2 will print as "-")
    L
  }
  
  get_lambda <- function(L, i, j) {
    if (is.null(L)) return(NA_real_)
    if (nrow(L) < i) return(NA_real_)
    if (ncol(L) < j) return(NA_real_)
    L[i, j]
  }
  
  # ----------------------------
  # formatting helpers
  # ----------------------------
  fmt2_plain <- function(x) {
    m = safe_mean(x)
    if (is.na(m)) return("-")
    sprintf("%.2f", m)
  }
  fmt4_plain <- function(x) {
    m = safe_mean(x)
    if (is.na(m)) return("-")
    sprintf("%.4f", m)
  }
  fmt_lambda <- function(x) {
    if (is.null(x) || length(x) == 0 || is.na(x)) return("-")
    as.character(x)
  }
  
  bold_cmd <- "\\textBF"
  eps <- 1e-12
  
  # ----------------------------
  # 1) Pre-pass: compute bold cutoffs on TEST metrics
  # bold if within 2*SE(best row) of best mean
  # ----------------------------
  test_auc_mean <- c(); test_auc_se <- c()
  test_acc_mean <- c(); test_acc_se <- c()
  
  add_candidate <- function(acc_te, auc_te) {
    test_auc_mean <<- c(test_auc_mean, safe_mean(auc_te))
    test_auc_se   <<- c(test_auc_se,   safe_se(auc_te))
    test_acc_mean <<- c(test_acc_mean, safe_mean(acc_te))
    test_acc_se   <<- c(test_acc_se,   safe_se(acc_te))
  }
  
  # helper: add candidates for a method either as 1 row (scalar) or multiple rows
  add_candidates_method <- function(acc_te, auc_te) {
    m = max(length(acc_te), length(auc_te), 1L)
    if (m <= 1) {
      add_candidate(acc_te, auc_te)
    } else {
      for (j in 1:m) add_candidate(get_at(acc_te, j), get_at(auc_te, j))
    }
  }
  
  if (include_CTree && !is.null(res$CTree_accuracy_test) && !is.null(res$CTree_auc_test))
    add_candidates_method(res$CTree_accuracy_test, res$CTree_auc_test)
  
  if (include_CRF && !is.null(res$CRF_accuracy_test) && !is.null(res$CRF_auc_test))
    add_candidates_method(res$CRF_accuracy_test, res$CRF_auc_test)
  
  if (include_rpart && !is.null(res$rpart_accuracy_test) && !is.null(res$rpart_auc_test))
    add_candidate(res$rpart_accuracy_test, res$rpart_auc_test)
  
  if (include_rf && !is.null(res$rf_accuracy_test) && !is.null(res$rf_auc_test))
    add_candidate(res$rf_accuracy_test, res$rf_auc_test)
  
  if (include_icp_rpart && !is.null(res$icp_rpart_accuracy_test) && !is.null(res$icp_rpart_auc_test))
    add_candidate(res$icp_rpart_accuracy_test, res$icp_rpart_auc_test)
  
  if (include_icp_rf && !is.null(res$icp_rf_accuracy_test) && !is.null(res$icp_rf_auc_test))
    add_candidate(res$icp_rf_accuracy_test, res$icp_rf_auc_test)
  
  if (include_anchor && !is.null(res$anchor_accuracy_test) && !is.null(res$anchor_auc_test))
    add_candidate(res$anchor_accuracy_test, res$anchor_auc_test)
  
  if (include_anchor_boost && !is.null(res$anchor_boost_gamma) &&
      !is.null(res$anchor_boost_accuracy_test) && !is.null(res$anchor_boost_auc_test)) {
    for (j in 1:length(res$anchor_boost_gamma))
      add_candidate(get_at(res$anchor_boost_accuracy_test, j), get_at(res$anchor_boost_auc_test, j))
  }
  
  if (include_irf && !is.null(res$irf_lambda) &&
      !is.null(res$irf_accuracy_test) && !is.null(res$irf_auc_test)) {
    for (j in 1:length(res$irf_lambda))
      add_candidate(get_at(res$irf_accuracy_test, j), get_at(res$irf_auc_test, j))
  }
  
  if (length(test_auc_mean) > 0) {
    i_best_auc = which.max(test_auc_mean)
    auc_cut = test_auc_mean[i_best_auc] - 2 * test_auc_se[i_best_auc]
  } else auc_cut = -Inf
  
  if (length(test_acc_mean) > 0) {
    i_best_acc = which.max(test_acc_mean)
    acc_cut = test_acc_mean[i_best_acc] - 2 * test_acc_se[i_best_acc]
  } else acc_cut = -Inf
  
  should_bold_auc <- function(auc_te) safe_mean(auc_te) >= (auc_cut - eps)
  should_bold_acc <- function(acc_te) safe_mean(acc_te) >= (acc_cut - eps)
  if(percentage){
    should_bold_auc <- function(auc_te) safe_mean(auc_te) >= (100*auc_cut - eps)
    should_bold_acc <- function(acc_te) safe_mean(acc_te) >= (100*acc_cut - eps)
  }
  
  
  fmt2 <- function(x, bold = FALSE) {
    s = fmt2_plain(x)
    if (bold && s != "-") s = paste0(bold_cmd, "{", s, "}")
    s
  }
  
  # ----------------------------
  # printer: Method & lambda1 & lambda2 & ACC_tr & AUC_tr & ACC_te & AUC_te & rho_i & rho_ii
  # ----------------------------
  print_line <- function(method, lambda1, lambda2,
                         acc_tr, auc_tr, acc_te, auc_te, acc_sd, rho_i, rho_ii) {
    if(percentage) {
      auc_tr = 100*auc_tr
      acc_tr = 100*acc_tr
      auc_te = 100*auc_te
      acc_te = 100*acc_te
      acc_sd = 100*acc_sd
    }
    sd.opt = ' '
    if(print.acc.sd){
      sd.opt = paste0('(', fmt2(acc_sd/n_scale,), ')')
    }
    cat(
      method, "&",
      fmt_lambda(lambda1), "&",
      fmt_lambda(lambda2), "&",
      fmt2(auc_tr, bold = FALSE), "&",
      fmt2(acc_tr, bold = FALSE), "&",
      fmt2(auc_te, bold = should_bold_auc(auc_te)), "&",
      fmt2(acc_te, bold = should_bold_acc(acc_te)), sd.opt, "&",
      fmt4_plain(rho_i), "&",
      fmt4_plain(rho_ii),
      "\\\\\n"
    )
  }
  
  # helper: print CTree/CRF correctly
  # - if metrics are scalar (one run), print ONE line
  # - lambdas may be 1x2 OR 2x1 OR a grid; in scalar case we print the LAST pair
  print_CTree_block <- function(method,
                                 L_raw,
                                 acc_tr, auc_tr, acc_te, auc_te, acc_sd, rho_i, rho_ii) {
    
    L = normalize_lambdas(L_raw)
    m = max(length(acc_te), length(auc_te), length(acc_tr), length(auc_tr),
            length(rho_i), length(rho_ii), 1L)
    
    if (m <= 1) {
      # one run -> one row
      idx = if (is.null(L)) 1L else nrow(L)
      print_line(method,
                 get_lambda(L, idx, 1), get_lambda(L, idx, 2),
                 acc_tr, auc_tr, acc_te, auc_te, acc_sd, rho_i, rho_ii)
    } else {
      # multiple configs -> align rows
      if (is.null(L)) stop(method, ": lambdas missing but multiple results provided.")
      if (nrow(L) != m) stop(method, ": nrow(lambdas) != number of results.")
      for (j in 1:m) {
        print_line(method,
                   get_lambda(L, j, 1), get_lambda(L, j, 2),
                   get_at(acc_tr, j), get_at(auc_tr, j),
                   get_at(acc_te, j), get_at(auc_te, j), get_at(acc_sd, j),
                   get_at(rho_i, j), get_at(rho_ii, j))
      }
    }
  }
  
  # ----------------------------
  # 2) Print rows
  # ----------------------------
  if (include_CTree && !is.null(res$CTree_accuracy_train)) {
    print_CTree_block("CTree",
                       res$CTree_lambdas,
                       res$CTree_accuracy_train, res$CTree_auc_train,
                       res$CTree_accuracy_test,  res$CTree_auc_test,
                       res$CTree_accuracy_test_sd,
                       res$CTree_rho.i,          res$CTree_rho.ii)
  }
  
  if (include_CRF && !is.null(res$CRF_accuracy_train)) {
    print_CTree_block("CRF",
                       res$CRF_lambdas,
                       res$CRF_accuracy_train, res$CRF_auc_train,
                       res$CRF_accuracy_test,  res$CRF_auc_test,
                       res$CRF_accuracy_test_sd,
                       res$CRF_rho.i,          res$CRF_rho.ii)
  }
  
  if (include_rpart && !is.null(res$rpart_accuracy_train)) {
    print_line("rpart", "-", "-",
               res$rpart_accuracy_train, res$rpart_auc_train,
               res$rpart_accuracy_test,  res$rpart_auc_test,
               res$rpart_accuracy_test_sd,
               res$rpart_rho.i,          res$rpart_rho.ii)
  }
  
  if (include_rf && !is.null(res$rf_accuracy_train)) {
    print_line("RF", "-", "-",
               res$rf_accuracy_train, res$rf_auc_train,
               res$rf_accuracy_test,  res$rf_auc_test,
               res$rf_accuracy_test_sd,
               res$rf_rho.i,          res$rf_rho.ii)
  }
  
  if (include_icp_rpart && !is.null(res$icp_rpart_accuracy_train)) {
    print_line("ICP+rpart", "-", "-",
               res$icp_rpart_accuracy_train, res$icp_rpart_auc_train,
               res$icp_rpart_accuracy_test,  res$icp_rpart_auc_test,
               res$icp_rpart_accuracy_test_sd,
               res$icp_rpart_rho.i,          res$icp_rpart_rho.ii)
  }
  
  if (include_icp_rf && !is.null(res$icp_rf_accuracy_train)) {
    print_line("ICP+RF", "-", "-",
               res$icp_rf_accuracy_train, res$icp_rf_auc_train,
               res$icp_rf_accuracy_test,  res$icp_rf_auc_test,
               res$icp_rf_accuracy_test_sd,
               res$icp_rf_rho.i,          res$icp_rf_rho.ii)
  }
  
  if (include_anchor && !is.null(res$anchor_accuracy_train)) {
    print_line("Anchor regression", "-", "-",
               res$anchor_accuracy_train, res$anchor_auc_train,
               res$anchor_accuracy_test,  res$anchor_auc_test,
               res$anchor_accuracy_test_sd,
               res$anchor_rho.i,          res$anchor_rho.ii)
  }
  
  # Anchor boosting: lambda1 = gamma, lambda2 = "-"
  if (include_anchor_boost && !is.null(res$anchor_boost_gamma) && !is.null(res$anchor_boost_accuracy_train)) {
    for (j in 1:length(res$anchor_boost_gamma)) {
      print_line("Anchor boosting",
                 res$anchor_boost_gamma[j], "-",
                 get_at(res$anchor_boost_accuracy_train, j),
                 get_at(res$anchor_boost_auc_train, j),
                 get_at(res$anchor_boost_accuracy_test, j),
                 get_at(res$anchor_boost_auc_test, j),
                 get_at(res$anchor_boost_accuracy_test_sd, j),
                 get_at(res$anchor_boost_rho.i, j),
                 get_at(res$anchor_boost_rho.ii, j))
    }
  }
  
  # IRF: lambda1 = lambda, lambda2 = "-"
  if (include_irf && !is.null(res$irf_lambda) && !is.null(res$irf_accuracy_train)) {
    for (j in 1:length(res$irf_lambda)) {
      print_line("IRF",
                 res$irf_lambda[j], "-",
                 get_at(res$irf_accuracy_train, j),
                 get_at(res$irf_auc_train, j),
                 get_at(res$irf_accuracy_test, j),
                 get_at(res$irf_auc_test, j),
                 get_at(res$irf_accuracy_test_sd, j),
                 get_at(res$irf_rho.i, j),
                 get_at(res$irf_rho.ii, j))
    }
  }
}


clean_colnames = function(df) {
  n = names(df)
  n = gsub("`", "", n, fixed = TRUE)   # remove literal backticks
  n = trimws(n)                       # trim leading/trailing whitespace
  n = gsub("\\s+", "_", n)            # replace any whitespace runs with underscores
  n = gsub("/", "_", n)  
  #n = gsub("(", "_", n) 
  #n = gsub(")", "_", n) 
  n = gsub("\\(|\\)", "_", n)
  n = make.unique(n, sep = "_")       # ensure uniqueness after cleaning
  names(df) = n
  df
}


