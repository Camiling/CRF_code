# Print diversity simulation results as a LaTeX-style table.
# Columns: Method, lambda1, lambda2, Acc, AUC, Disagree, DblFault, Jaccard, X1..X5
# One row per (lambda combination x sampling variant): sample_nodes=TRUE vs FALSE.
# inclusion=TRUE prints InclusionRate; inclusion=FALSE prints scaled MeanDecreaseCustomLoss.
print_diversity_sim_res <- function(obj, lambdas_comb, features,
                                    show_sd = TRUE, inclusion = T,
                                    show_auc = F) {
  n.lambdacombs <- nrow(lambdas_comb)
  bold_cmd      <- "\\textBF"
  eps           <- 1e-12

  safe_mean <- function(x) mean(x, na.rm = TRUE)
  safe_sd   <- function(x) { s <- stats::sd(x, na.rm = TRUE); if (is.na(s)) 0 else s }
  safe_se   <- function(x) { n <- sum(!is.na(x)); if (n <= 1) 0 else safe_sd(x) / sqrt(n) }

  fmt <- function(vals, digits = 3, bold = FALSE) {
    m   <- safe_mean(vals)
    out <- sprintf(paste0("%.", digits, "f"), round(m, digits))
    if (show_sd) {
      s   <- safe_sd(vals)
      out <- paste0(out, " (", sprintf(paste0("%.", digits, "f"), round(s, digits)), ")")
    }
    if (bold) out <- paste0(bold_cmd, "{", out, "}")
    out
  }

  fmt2 <- function(v) sprintf("%.2f", round(v, 2))

  # Collect all acc/auc vectors across both variants and all lambdas for bolding cutoffs
  all_acc <- c(lapply(seq_len(n.lambdacombs), function(j) obj$acc.all[, j]),
               lapply(seq_len(n.lambdacombs), function(j) obj$acc.nodes[, j]))
  all_auc <- c(lapply(seq_len(n.lambdacombs), function(j) obj$auc.all[, j]),
               lapply(seq_len(n.lambdacombs), function(j) obj$auc.nodes[, j]))

  acc_means <- sapply(all_acc, safe_mean)
  auc_means <- sapply(all_auc, safe_mean)
  best_acc <- which.max(acc_means)
  best_auc <- which.max(auc_means)
  acc_cut  <- acc_means[best_acc] - 2 * safe_se(all_acc[[best_acc]])
  auc_cut  <- auc_means[best_auc] - 2 * safe_se(all_auc[[best_auc]])

  should_bold_acc <- function(vals) safe_mean(vals) >= acc_cut - eps
  should_bold_auc <- function(vals) safe_mean(vals) >= auc_cut - eps

  feat_mat <- if (inclusion) list(all = obj$inc.all, nodes = obj$inc.nodes)
              else            list(all = obj$imp.all, nodes = obj$imp.nodes)

  print_row <- function(label, acc, auc, dis, df, jac, feat_vals, j) {
    cat(label, lambdas_comb[j, 1], lambdas_comb[j, 2], sep = " & ")
    cat(" & ", fmt(acc[, j], bold = should_bold_acc(acc[, j])), sep = "")
    if (show_auc)
      cat(" & ", fmt(auc[, j], bold = should_bold_auc(auc[, j])), sep = "")
    cat(" & ", fmt(dis[, j]),
        " & ", fmt(df[, j]),
        " & ", fmt(jac[, j]),
        " & ", sep = "")
    cat(fmt2(feat_vals[, j]), sep = " & ")
    cat(" \\\\\n")
  }

  for (j in seq_len(n.lambdacombs))
    print_row("CRF (full)",   obj$acc.all,   obj$auc.all,   obj$dis.all,   obj$df.all,
              obj$jac.all,   feat_mat$all,   j)
  for (j in seq_len(n.lambdacombs))
    print_row("CRF (mtry)", obj$acc.nodes, obj$auc.nodes, obj$dis.nodes, obj$df.nodes,
              obj$jac.nodes, feat_mat$nodes, j)
}


# Print header line matching the table columns above.
print_diversity_header <- function(features, show_auc = TRUE) {
  feat_cols <- paste(features, collapse = " & ")
  auc_col   <- if (show_auc) " & AUC" else ""
  cat("Method & $\\lambda_1$ & $\\lambda_2$ & Acc", auc_col,
      "& Disagree & DblFault & Jaccard &",
      feat_cols, "\\\\\n")
}
