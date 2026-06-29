
# Print results of high-dim sim
print_tree_sim_res_large = function(obj, lambdas_comb=c(0,0), gamma.anchor=0, lambda.irf=0,
                                    include_CTree = TRUE, include_CRF = TRUE, include_rpart = TRUE, include_rf = TRUE,
                                    include_icp_rpart = TRUE, include_icp_rf = TRUE, include_anchor = FALSE,
                                    include_anchor_boost = TRUE, include_irf = TRUE, include_training=TRUE,
                                    sd=TRUE, multiple_test_domains=FALSE, inclusion=F, setting=NULL,
                                    d=c(5,5,5,5,5)){
  
  n.lambdacombs = max(1, nrow(lambdas_comb))
  n.gamma.anchor = length(gamma.anchor)
  n.lambda.irf = length(lambda.irf)
  
  if(setting=='trees'){ # only print rpart, CTree and CRF for various params
    include_rf = FALSE
    include_icp_rpart = FALSE
    include_icp_rf = FALSE
    include_anchor = FALSE
    include_anchor_boost = FALSE
    include_irf = FALSE
    inclusion=TRUE
  }else if(setting=='invariant'){ # Compare invariance-based methods
    include_rpart = FALSE
    include_CTree=FALSE
    inclusion=FALSE
  }
  
  if(!include_anchor){ # Not including this in paper
    obj$accuracy.vals.anchor = obj$accuracy.vals.anchor*0
    obj$auc.vals.anchor = obj$auc.vals.anchor*0
  }
  
  # Safe helpers (handle vectors vs matrices, and NA)
  get_i = function(x, i) if (is.null(dim(x))) x else x[, i]
  n_eff = function(x) sum(!is.na(x))
  safe_mean = function(x) mean(x, na.rm=TRUE)
  safe_sd   = function(x) { s = sd(x, na.rm=TRUE); if (is.na(s)) 0 else s }
  safe_se   = function(x) {
    n = n_eff(x)
    if (n <= 1) return(0)
    safe_sd(x) / sqrt(n)
  }
  
  # Collapse inclusion/importance to 5 block means (x1..x5)
  # d can be scalar or length-5 equal dims; we always use d[1] if length-5
  collapse_inclusion = function(v){
    if(is.null(v)) return(NULL)
    block_mean_5_equal(v, d)
  }
  
  # Pre-pass: compute bolding cutoffs from TEST performance across printed rows
  # Rule: bold if mean is max OR within 2 SE of the best row (best row's SE).
  bold_cmd = "\\textBF"
  eps = 1e-12
  
  test_mean_auc = c(); test_se_auc = c()
  test_mean_acc = c(); test_se_acc = c()
  
  add_stats = function(acc, auc) {
    test_mean_auc <<- c(test_mean_auc, safe_mean(auc))
    test_se_auc   <<- c(test_se_auc,   safe_se(auc))
    test_mean_acc <<- c(test_mean_acc, safe_mean(acc))
    test_se_acc   <<- c(test_se_acc,   safe_se(acc))
  }
  
  if (include_CTree) for (i in 1:n.lambdacombs) {
    if(setting=='invariant'){
      if(lambdas_comb[i,1]==lambdas_comb[i,2] & lambdas_comb[i,1]!=0){
        add_stats(get_i(obj$accuracy.vals.CTree, i), get_i(obj$auc.vals.CTree, i))
      }
    } else {
      add_stats(get_i(obj$accuracy.vals.CTree, i), get_i(obj$auc.vals.CTree, i))
    }
  }
  if (include_CRF) for (i in 1:n.lambdacombs) {
    if(setting=='invariant'){
      if(lambdas_comb[i,1]==lambdas_comb[i,2] & lambdas_comb[i,1]!=0){
        add_stats(get_i(obj$accuracy.vals.CRF, i), get_i(obj$auc.vals.CRF, i))
      }
    } else {
      add_stats(get_i(obj$accuracy.vals.CRF, i), get_i(obj$auc.vals.CRF, i))
    }
  }
  
  if (include_rpart)     add_stats(obj$accuracy.vals.rpart,     obj$auc.vals.rpart)
  if (include_rf)        add_stats(obj$accuracy.vals.rf,        obj$auc.vals.rf)
  if (include_icp_rpart) add_stats(obj$accuracy.vals.icp_rpart, obj$auc.vals.icp_rpart)
  if (include_icp_rf)    add_stats(obj$accuracy.vals.icp_rf,    obj$auc.vals.icp_rf)
  if (include_anchor)    add_stats(obj$accuracy.vals.anchor,    obj$auc.vals.anchor)
  
  if (include_anchor_boost) for (i in 1:n.gamma.anchor) add_stats(get_i(obj$accuracy.vals.anchor_boost, i), get_i(obj$auc.vals.anchor_boost, i))
  if (include_irf)          for (i in 1:n.lambda.irf)   add_stats(get_i(obj$accuracy.vals.irf,         i), get_i(obj$auc.vals.irf,         i))
  
  if (length(test_mean_auc) > 0) {
    i_best_auc = which.max(test_mean_auc)
    auc_cut = test_mean_auc[i_best_auc] - 2*test_se_auc[i_best_auc]
  } else {
    auc_cut = -Inf
  }
  
  if (length(test_mean_acc) > 0) {
    i_best_acc = which.max(test_mean_acc)
    acc_cut = test_mean_acc[i_best_acc] - 2*test_se_acc[i_best_acc]
  } else {
    acc_cut = -Inf
  }
  
  should_bold_auc = function(auc_vec) safe_mean(auc_vec) >= (auc_cut - eps)
  should_bold_acc = function(acc_vec) safe_mean(acc_vec) >= (acc_cut - eps)
  
  # Wrapper used ONLY for test printing
  print_performance_test = function(acc, auc, ...) {
    print_performance(
      acc, auc, ...,
      bold_auc = should_bold_auc(auc),
      bold_acc = should_bold_acc(acc),
      bold_cmd = bold_cmd
    )
  }
  
  # --------------------------------------------------------------------------
  # Printing (TRAIN unbolded; TEST bolded by wrapper above)
  # --------------------------------------------------------------------------
  if(include_CTree){
    for(i in 1:n.lambdacombs){
      if(n.lambdacombs==1){
        lambda1.tmp = lambdas_comb[1]
        lambda2.tmp = lambdas_comb[2]
      } else {
        lambda1.tmp = lambdas_comb[i,1]
        lambda2.tmp = lambdas_comb[i,2]
      }
      if(! (setting=='invariant' & (lambda1.tmp!=lambda2.tmp | (lambda1.tmp ==0 & lambda2.tmp == 0 )))){
        cat('CTree ', lambda1.tmp, lambda2.tmp, sep=' & ')
        cat(' & ')
        if(include_training){
          print_performance(get_i(obj$accuracy.vals.train.CTree, i),
                            get_i(obj$auc.vals.train.CTree, i),
                            obj$mean.rho.i.vals.train.CTree[i],
                            obj$mean.rho.ii.vals.train.CTree[i],
                            sd=sd)
        }
        if(inclusion) obj$var_importance.CTree = obj$var_inclusion.CTree
        print_performance_test(get_i(obj$accuracy.vals.CTree, i),
                               get_i(obj$auc.vals.CTree, i),
                               sd=sd,
                               inclusion=collapse_inclusion(get_i(obj$var_importance.CTree, i)),
                               show_rho=multiple_test_domains,
                               suppress_rho=!multiple_test_domains)
      }
    }
  }
  
  if(include_CRF){
    for(i in 1:n.lambdacombs){
      if(n.lambdacombs==1){
        lambda1.tmp = lambdas_comb[1]
        lambda2.tmp = lambdas_comb[2]
      } else {
        lambda1.tmp = lambdas_comb[i,1]
        lambda2.tmp = lambdas_comb[i,2]
      }
      if(!(setting=='invariant' & (lambda1.tmp!=lambda2.tmp | (lambda1.tmp ==0 & lambda2.tmp == 0 )))){
        cat('CRF ', lambda1.tmp, lambda2.tmp, sep=' & ')
        cat(' & ')
        if(include_training){
          print_performance(get_i(obj$accuracy.vals.train.CRF, i),
                            get_i(obj$auc.vals.train.CRF, i),
                            obj$mean.rho.i.vals.train.CRF[i],
                            obj$mean.rho.ii.vals.train.CRF[i],
                            sd=sd)
        }
        if(inclusion) obj$var_importance.CRF = obj$var_inclusion.CRF
        print_performance_test(get_i(obj$accuracy.vals.CRF, i),
                               get_i(obj$auc.vals.CRF, i),
                               sd=sd,
                               inclusion=collapse_inclusion(get_i(obj$var_importance.CRF, i)),
                               show_rho=multiple_test_domains,
                               suppress_rho=!multiple_test_domains)
      }
    }
  }
  
  if(include_rpart){
    cat('rpart & - & - & ')
    if(include_training){
      print_performance(obj$accuracy.vals.train.rpart,
                        obj$auc.vals.train.rpart,
                        obj$mean.rho.i.vals.train.rpart,
                        obj$mean.rho.ii.vals.train.rpart,
                        sd=sd)
    }
    if(inclusion) obj$var_importance.rpart = obj$var_inclusion.rpart
    print_performance_test(obj$accuracy.vals.rpart,
                           obj$auc.vals.rpart,
                           sd=sd,
                           inclusion=collapse_inclusion(obj$var_importance.rpart),
                           show_rho=multiple_test_domains,
                           suppress_rho=!multiple_test_domains)
  }
  
  if(include_rf){
    cat('RF & - & - & ')
    if(include_training){
      print_performance(obj$accuracy.vals.train.rf,
                        obj$auc.vals.train.rf,
                        obj$mean.rho.i.vals.train.rf,
                        obj$mean.rho.ii.vals.train.rf,
                        sd=sd,
                        show_rho=FALSE)
    }
    print_performance_test(obj$accuracy.vals.rf,
                           obj$auc.vals.rf,
                           sd=sd,
                           inclusion=collapse_inclusion(obj$var_importance.rf),
                           show_rho=multiple_test_domains,
                           suppress_rho=!multiple_test_domains)
  }
  
  if(include_icp_rpart){
    cat('ICP + rpart & - & - & ')
    if(include_training){
      print_performance(obj$accuracy.vals.train.icp_rpart,
                        obj$auc.vals.train.icp_rpart,
                        obj$mean.rho.i.vals.train.icp_rpart,
                        obj$mean.rho.ii.vals.train.icp_rpart,
                        sd=sd)
    }
    if(inclusion) obj$var_importance.icp_rpart = obj$var_inclusion.icp_rpart
    print_performance_test(obj$accuracy.vals.icp_rpart,
                           obj$auc.vals.icp_rpart,
                           sd=sd,
                           inclusion=collapse_inclusion(obj$var_importance.icp_rpart),
                           show_rho=multiple_test_domains,
                           suppress_rho=!multiple_test_domains)
  }
  
  if(include_icp_rf){
    cat('ICP + RF & - & - & ')
    if(include_training){
      print_performance(obj$accuracy.vals.train.icp_rf,
                        obj$auc.vals.train.icp_rf,
                        obj$mean.rho.i.vals.train.icp_rf,
                        obj$mean.rho.ii.vals.train.icp_rf,
                        sd=sd,
                        show_rho=FALSE)
    }
    print_performance_test(obj$accuracy.vals.icp_rf,
                           obj$auc.vals.icp_rf,
                           sd=sd,
                           inclusion=collapse_inclusion(obj$var_importance.icp_rf),
                           show_rho=multiple_test_domains,
                           suppress_rho=!multiple_test_domains)
  }
  
  if(include_anchor){
    cat('Anchor & - & - & ')
    if(include_training){
      print_performance(obj$accuracy.vals.train.anchor,
                        obj$auc.vals.train.anchor,
                        obj$mean.rho.i.vals.train.anchor,
                        obj$mean.rho.ii.vals.train.anchor,
                        sd=sd,
                        show_rho=FALSE)
    }
    print_performance_test(obj$accuracy.vals.anchor,
                           obj$auc.vals.anchor,
                           sd=sd,
                           inclusion=collapse_inclusion(obj$var_importance.anchor),
                           show_rho=multiple_test_domains,
                           suppress_rho=!multiple_test_domains)
  }
  
  if(include_anchor_boost){
    for(i in 1:n.gamma.anchor){
      if(n.gamma.anchor==1){
        gamma.tmp = gamma.anchor
      } else {
        gamma.tmp = gamma.anchor[i]
      }
      cat('AnchorBoosting', gamma.tmp, sep=' & ')
      cat(' & - & ')
      if(include_training){
        print_performance(get_i(obj$accuracy.vals.train.anchor_boost, i),
                          get_i(obj$auc.vals.train.anchor_boost, i),
                          obj$mean.rho.i.vals.train.anchor_boost[i],
                          obj$mean.rho.ii.vals.train.anchor_boost[i],
                          sd=sd,
                          show_rho=FALSE)
      }
      print_performance_test(get_i(obj$accuracy.vals.anchor_boost, i),
                             get_i(obj$auc.vals.anchor_boost, i),
                             sd=sd,
                             inclusion=collapse_inclusion(get_i(obj$var_importance.anchor_boost, i)),
                             show_rho=multiple_test_domains,
                             suppress_rho=!multiple_test_domains)
    }
  }
  
  if(include_irf){
    for(i in 1:n.lambda.irf){
      if(n.lambda.irf==1){
        lambda.irf.tmp = lambda.irf
      } else {
        lambda.irf.tmp = lambda.irf[i]
      }
      cat('IRF', lambda.irf.tmp, sep=' & ')
      cat(' & - & ')
      if(include_training){
        print_performance(get_i(obj$accuracy.vals.train.irf, i),
                          get_i(obj$auc.vals.train.irf, i),
                          obj$mean.rho.i.vals.train.irf[i],
                          obj$mean.rho.ii.vals.train.irf[i],
                          sd=sd,
                          show_rho=FALSE)
      }
      print_performance_test(get_i(obj$accuracy.vals.irf, i),
                             get_i(obj$auc.vals.irf, i),
                             sd=sd,
                             inclusion=collapse_inclusion(get_i(obj$var_importance.irf, i)),
                             show_rho=multiple_test_domains,
                             suppress_rho=!multiple_test_domains)
    }
  }
}


print_performance = function(acc, auc, rho1=NA, rho2=NA, sd=FALSE, inclusion=NULL,
                             show_rho=TRUE, suppress_rho=FALSE,
                             bold_auc=FALSE, bold_acc=FALSE,
                             bold_cmd="\\textBF") {
  
  safe_mean = function(x) mean(x, na.rm=TRUE)
  safe_sd   = function(x) { s = sd(x, na.rm=TRUE); if (is.na(s)) 0 else s }
  
  auc_str = sprintf("%.3f", round(safe_mean(auc), 3))
  if (sd) auc_str = paste0(auc_str, " (", sprintf("%.3f", round(safe_sd(auc), 3)), ")")
  if (bold_auc) auc_str = paste0(bold_cmd, "{", auc_str, "}")
  
  acc_str = sprintf("%.3f", round(safe_mean(acc), 3))
  if (sd) acc_str = paste0(acc_str, " (", sprintf("%.3f", round(safe_sd(acc), 3)), ")")
  if (bold_acc) acc_str = paste0(bold_cmd, "{", acc_str, "}")
  
  cat(auc_str); cat(" & "); cat(acc_str); cat(" &")
  
  if (suppress_rho) {
    cat("& ")
    if (!is.null(inclusion)) cat(sprintf("%.2f", round(inclusion, 2)), sep=" & ")
    cat("\\\\ \n")
    return(invisible(NULL))
  }
  
  cat(" ")
  if (!show_rho) {
    rho1 = rho2 = "-"
  } else {
    if (is.na(rho1) || is.na(rho2)) {
      rho1 = rho2 = "-"
    } else {
      rho1 = sprintf("%.3f", round(rho1, 3))
      rho2 = sprintf("%.3f", round(rho2, 3))
    }
  }
  
  if (is.null(inclusion)) {
    cat(rho1, rho2, sep=" & ")
    cat(" && ")
  } else {
    cat(rho1, " & ", rho2, " && ")
    # inclusion is a 5-vector -> print x1..x5 block means
    cat(sprintf("%.2f", round(inclusion, 2)), sep=" & ")
    cat("\\\\ \n")
  }
}


block_mean_5_equal = function(v, d){
  v = as.numeric(v)
  
  # accept scalar or length-5 equal dims; use scalar internally
  if(length(d) == 5) d = d[1]
  stopifnot(length(d) == 1)
  
  if(length(v) != 5*d){
    stop("block_mean_5_equal: length(v)=", length(v),
         " but expected 5*d=", 5*d,
         ". Pass correct d (scalar) or equal-length d vector.")
  }
  
  out = sapply(1:5, function(k){
    i0 = (k-1)*d + 1
    i1 = k*d
    mean(v[i0:i1], na.rm = TRUE)
  })
  names(out) = paste0("x", 1:5)
  out
}

print_res_aggregated = function(res_a, res_b, res_c, res_d, res_e,
                                lambdas_comb, gamma.anchor, lambda_irf,
                                include_CRF = TRUE,
                                include_rf = TRUE,
                                include_icp_rf = TRUE,
                                include_anchor_boost = TRUE,
                                include_irf = TRUE,
                                sd = TRUE,
                                bold_cmd = "\\textBF") {
  
  # ----------------------------
  # helpers
  # ----------------------------
  safe_mean = function(x) {
    if (is.null(x) || length(x) == 0 || all(is.na(x))) return(NA_real_)
    mean(x, na.rm = TRUE)
  }
  safe_sd = function(x) {
    if (is.null(x) || length(x) == 0 || all(is.na(x))) return(NA_real_)
    s = sd(x, na.rm = TRUE)
    if (is.na(s)) 0 else s
  }
  safe_n = function(x) {
    if (is.null(x) || length(x) == 0) return(0L)
    sum(!is.na(x))
  }
  safe_se = function(x) {
    n = safe_n(x)
    if (n <= 1) return(0)
    s = safe_sd(x)
    if (is.na(s)) return(0)
    s / sqrt(n)
  }
  
  fmt_cell = function(x, bold = FALSE) {
    m = safe_mean(x)
    if (is.na(m)) return("-")
    if (sd) {
      s = safe_sd(x)
      if (is.na(s)) s = 0
      out = sprintf("%.3f (%.3f)", round(m, 3), round(s, 3))
    } else {
      out = sprintf("%.3f", round(m, 3))
    }
    if (bold) out = paste0(bold_cmd, "{", out, "}")
    out
  }
  
  fmt_lam = function(x) {
    if (is.null(x) || length(x) == 0 || is.na(x)) return("-")
    as.character(x)
  }
  
  get_col = function(x, j) {
    if (is.null(x)) return(NULL)
    if (is.null(dim(x))) return(x)
    x[, j]
  }
  
  # lambdas_comb must be n x 2
  if (is.vector(lambdas_comb) && length(lambdas_comb) == 2) {
    lambdas_comb = matrix(lambdas_comb, nrow = 1, byrow = TRUE)
  }
  if (!is.matrix(lambdas_comb) || ncol(lambdas_comb) != 2) {
    stop("lambdas_comb must be a matrix with 2 columns (or a length-2 vector).")
  }
  
  # invariant-only: lambda1=lambda2 != 0
  inv_idx = which(lambdas_comb[,1] == lambdas_comb[,2] & lambdas_comb[,1] != 0)
  
  models = list(a = res_a, b = res_b, c = res_c, d = res_d, e = res_e)
  model_names = names(models)
  
  # ----------------------------
  # row specs (one row per method/config)
  # ----------------------------
  rows = list()
  
  if (include_CRF && length(inv_idx) > 0) {
    for (j in inv_idx) {
      rows[[length(rows) + 1]] = list(
        method  = "CRF",
        lambda1 = lambdas_comb[j, 1],
        lambda2 = lambdas_comb[j, 2],
        key     = "CRF",
        idx     = j
      )
    }
  }
  
  if (include_rf) {
    rows[[length(rows) + 1]] = list(method="RF", lambda1=NA, lambda2=NA, key="rf", idx=NA)
  }
  if (include_icp_rf) {
    rows[[length(rows) + 1]] = list(method="ICP + RF", lambda1=NA, lambda2=NA, key="icp_rf", idx=NA)
  }
  if (include_anchor_boost && length(gamma.anchor) > 0) {
    for (j in seq_along(gamma.anchor)) {
      rows[[length(rows) + 1]] = list(method="AnchorBoosting", lambda1=gamma.anchor[j], lambda2=NA, key="anchor_boost", idx=j)
    }
  }
  if (include_irf && length(lambda_irf) > 0) {
    for (j in seq_along(lambda_irf)) {
      rows[[length(rows) + 1]] = list(method="IRF", lambda1=lambda_irf[j], lambda2=NA, key="irf", idx=j)
    }
  }
  
  if (length(rows) == 0) stop("No rows to print (check include_* flags and lambdas_comb).")
  
  # ----------------------------
  # extract test metrics vector for a given (obj,row)
  # ----------------------------
  get_metrics = function(obj, row) {
    if (is.null(obj)) return(list(auc=NULL, acc=NULL))
    
    if (row$key == "CRF") {
      return(list(
        auc = get_col(obj$auc.vals.CRF, row$idx),
        acc = get_col(obj$accuracy.vals.CRF, row$idx)
      ))
    }
    if (row$key == "rf") {
      return(list(auc = obj$auc.vals.rf, acc = obj$accuracy.vals.rf))
    }
    if (row$key == "icp_rf") {
      return(list(auc = obj$auc.vals.icp_rf, acc = obj$accuracy.vals.icp_rf))
    }
    if (row$key == "anchor_boost") {
      return(list(
        auc = get_col(obj$auc.vals.anchor_boost, row$idx),
        acc = get_col(obj$accuracy.vals.anchor_boost, row$idx)
      ))
    }
    if (row$key == "irf") {
      return(list(
        auc = get_col(obj$auc.vals.irf, row$idx),
        acc = get_col(obj$accuracy.vals.irf, row$idx)
      ))
    }
    list(auc=NULL, acc=NULL)
  }
  
  # ----------------------------
  # compute bold cutoffs PER MODEL and METRIC:
  # bold if mean >= best_mean - 2*SE(best)
  # ----------------------------
  R = length(rows)
  auc_mean = matrix(NA_real_, nrow=R, ncol=length(model_names), dimnames=list(NULL, model_names))
  auc_se   = matrix(0,        nrow=R, ncol=length(model_names), dimnames=list(NULL, model_names))
  acc_mean = matrix(NA_real_, nrow=R, ncol=length(model_names), dimnames=list(NULL, model_names))
  acc_se   = matrix(0,        nrow=R, ncol=length(model_names), dimnames=list(NULL, model_names))
  
  for (r in seq_len(R)) {
    for (m in model_names) {
      met = get_metrics(models[[m]], rows[[r]])
      auc_mean[r, m] = safe_mean(met$auc)
      auc_se[r, m]   = safe_se(met$auc)
      acc_mean[r, m] = safe_mean(met$acc)
      acc_se[r, m]   = safe_se(met$acc)
    }
  }
  
  auc_cut = setNames(rep(-Inf, length(model_names)), model_names)
  acc_cut = setNames(rep(-Inf, length(model_names)), model_names)
  
  for (m in model_names) {
    if (!all(is.na(auc_mean[, m]))) {
      i_best = which.max(auc_mean[, m])
      auc_cut[m] = auc_mean[i_best, m] - 2 * auc_se[i_best, m]
    }
    if (!all(is.na(acc_mean[, m]))) {
      i_best = which.max(acc_mean[, m])
      acc_cut[m] = acc_mean[i_best, m] - 2 * acc_se[i_best, m]
    }
  }
  
  should_bold_auc = function(m, auc_vec) safe_mean(auc_vec) >= auc_cut[m]
  should_bold_acc = function(m, acc_vec) safe_mean(acc_vec) >= acc_cut[m]
  
  # ----------------------------
  # HEADER (no extra '&')
  # ----------------------------
  cat("Method & $\\lambda_1$ & $\\lambda_2$")
  for (k in seq_along(model_names)) {
    m = model_names[k]
    if (k == 1) {
      cat(" & AUC (", m, ") & Acc (", m, ")", sep = "")
    } else {
      cat(" && AUC (", m, ") & Acc (", m, ")", sep = "")
    }
  }
  cat(" \\\\\n")
  
  # ----------------------------
  # ROWS (first block uses '&', later blocks use '&&')
  # ----------------------------
  for (r in seq_len(R)) {
    row = rows[[r]]
    cat(row$method, " & ", fmt_lam(row$lambda1), " & ", fmt_lam(row$lambda2), sep = "")
    
    for (k in seq_along(model_names)) {
      m = model_names[k]
      met = get_metrics(models[[m]], row)
      
      if (k == 1) {
        cat(" & ", fmt_cell(met$auc, bold = should_bold_auc(m, met$auc)),
            " & ", fmt_cell(met$acc, bold = should_bold_acc(m, met$acc)), sep = "")
      } else {
        cat(" && ", fmt_cell(met$auc, bold = should_bold_auc(m, met$auc)),
            " & ", fmt_cell(met$acc, bold = should_bold_acc(m, met$acc)), sep = "")
      }
    }
    cat(" \\\\\n")
  }
}
