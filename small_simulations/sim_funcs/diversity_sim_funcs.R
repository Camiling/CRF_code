library(ROCR)

# Run N replicate simulations comparing CRF with node sampling (sample_nodes=TRUE)
# vs. considering all features at every node (sample_nodes=FALSE).
# Returns arrays of test accuracy, AUC, three pairwise diversity measures,
# feature inclusion rates, and scaled custom-loss importance per replicate.
perform_diversity_sim <- function(N, n, n.test = 200, model = 'a',
                                  lambdas = matrix(c(0,0, 5,5, 10,10), ncol=2, byrow=T),
                                  min.leaf.size = 5,
                                  n.thresh = 30,
                                  n.trees = 200,
                                  eps.improvement = 1e-2,
                                  n.cores = NULL,
                                  relax.constraint = TRUE) {
  p              <- 5
  n.lambdacombs  <- nrow(lambdas)

  mk_mat <- function() matrix(0, N, n.lambdacombs)
  mk_arr <- function() array(0, c(N, p, n.lambdacombs))

  acc.nodes <- mk_mat(); auc.nodes <- mk_mat()
  dis.nodes <- mk_mat(); df.nodes  <- mk_mat(); jac.nodes <- mk_mat()
  inc.nodes <- mk_arr(); imp.nodes <- mk_arr()

  acc.all  <- mk_mat(); auc.all  <- mk_mat()
  dis.all  <- mk_mat(); df.all   <- mk_mat(); jac.all  <- mk_mat()
  inc.all  <- mk_arr(); imp.all  <- mk_arr()

  seeds <- sample(1:10000, size = N)
  for (i in seq_len(N)) {
    cat(sprintf('[Model %s] Replicate %d / %d (%.0f%%) ...\n',
                model, i, N, 100 * (i - 1) / N))
    flush.console()
    res.tmp <- perform_diversity_sim_oneiteration(
      n = n, n.test = n.test, model = model,
      lambdas = lambdas,
      min.leaf.size = min.leaf.size, n.thresh = n.thresh,
      n.trees = n.trees,
      eps.improvement = eps.improvement, n.cores = n.cores,
      relax.constraint = relax.constraint, seed = seeds[i]
    )
    for (j in seq_len(n.lambdacombs)) {
      acc.nodes[i, j]   <- res.tmp$acc.nodes[j]
      auc.nodes[i, j]   <- res.tmp$auc.nodes[j]
      dis.nodes[i, j]   <- res.tmp$dis.nodes[j]
      df.nodes[i, j]    <- res.tmp$df.nodes[j]
      jac.nodes[i, j]   <- res.tmp$jac.nodes[j]
      inc.nodes[i, , j] <- res.tmp$inc.nodes[, j]
      imp.nodes[i, , j] <- res.tmp$imp.nodes[, j]

      acc.all[i, j]     <- res.tmp$acc.all[j]
      auc.all[i, j]     <- res.tmp$auc.all[j]
      dis.all[i, j]     <- res.tmp$dis.all[j]
      df.all[i, j]      <- res.tmp$df.all[j]
      jac.all[i, j]     <- res.tmp$jac.all[j]
      inc.all[i, , j]   <- res.tmp$inc.all[, j]
      imp.all[i, , j]   <- res.tmp$imp.all[, j]
    }
    cat(sprintf('[Model %s] Replicate %d / %d (%.0f%%) done.\n',
                model, i, N, 100 * i / N))
    flush.console()
  }

  list(
    acc.nodes = acc.nodes, auc.nodes = auc.nodes,
    dis.nodes = dis.nodes, df.nodes  = df.nodes,  jac.nodes = jac.nodes,
    inc.nodes = apply(inc.nodes, c(2, 3), mean),
    imp.nodes = apply(imp.nodes, c(2, 3), mean),

    acc.all  = acc.all,  auc.all  = auc.all,
    dis.all  = dis.all,  df.all   = df.all,   jac.all  = jac.all,
    inc.all  = apply(inc.all,  c(2, 3), mean),
    imp.all  = apply(imp.all,  c(2, 3), mean),

    lambdas = lambdas, N = N, n = n, model = model
  )
}


# Single replicate: fits CRF with sample_nodes=TRUE and sample_nodes=FALSE for each lambda pair.
perform_diversity_sim_oneiteration <- function(n, n.test = 200, model = 'a',
                                               lambdas = matrix(c(0,0, 5,5), ncol=2, byrow=T),
                                               min.leaf.size = 5,
                                               n.thresh = 30,
                                               n.trees = 200,
                                               eps.improvement = 1e-2,
                                               n.cores = NULL,
                                               relax.constraint = TRUE,
                                               seed = 1) {
  set.seed(seed)
  p             <- 5
  n.lambdacombs <- nrow(lambdas)
  features      <- c('X1', 'X2', 'X3', 'X4', 'X5')

  # Training data
  dat    <- generate_data_tree(n, model = model, more.envirs = FALSE)
  df.tr  <- as.data.frame(cbind(dat$y, dat$X, dat$z))
  names(df.tr) <- c('y', features, 'z')

  # Test data (new environment)
  dat.te <- generate_data_tree(n.test, model = model, new = TRUE)
  df.te  <- as.data.frame(cbind(dat.te$y, dat.te$X, dat.te$z))
  names(df.te) <- c('y', features, 'z')

  df.tr$y <- as.factor(df.tr$y)
  df.te$y <- factor(df.te$y, levels = levels(df.tr$y))
  df.tr$z <- factor(df.tr$z, levels = union(unique(df.te$z), unique(df.tr$z)))
  df.te$z <- factor(df.te$z, levels = levels(df.tr$z))

  acc.nodes <- numeric(n.lambdacombs); auc.nodes <- numeric(n.lambdacombs)
  dis.nodes <- numeric(n.lambdacombs); df.nodes  <- numeric(n.lambdacombs)
  jac.nodes <- numeric(n.lambdacombs)
  inc.nodes <- matrix(0, p, n.lambdacombs)
  imp.nodes <- matrix(0, p, n.lambdacombs)

  acc.all  <- numeric(n.lambdacombs); auc.all  <- numeric(n.lambdacombs)
  dis.all  <- numeric(n.lambdacombs); df.all   <- numeric(n.lambdacombs)
  jac.all  <- numeric(n.lambdacombs)
  inc.all  <- matrix(0, p, n.lambdacombs)
  imp.all  <- matrix(0, p, n.lambdacombs)

  for (j in seq_len(n.lambdacombs)) {
    lam <- lambdas[j, ]

    # CRF with node-level feature sampling (sample_nodes=TRUE, mtry=sqrt(p))
    fit.nodes <- CRF(
      data = df.tr, target = 'y', environment = 'z', features = features,
      n.trees = n.trees, sample_nodes = TRUE, mtry = 'rf',
      lambda1 = lam[1], lambda2 = lam[2],
      min.leaf.size = min.leaf.size, n.thresh = n.thresh,
      eps.improvement = eps.improvement,
      compute.oob = FALSE, importance = FALSE,
      parallel = TRUE, n.cores = n.cores,
      relax.constraint = relax.constraint,
      compute_diversity = TRUE, diversity_data = df.te,
      seed = seed
    )
    prob.nodes <- predict(fit.nodes, df.te, type = 'prob')[, 2]
    r.nodes    <- .assess_acc_auc(df.te$y, prob.nodes)
    acc.nodes[j]   <- r.nodes$accuracy
    auc.nodes[j]   <- r.nodes$auc
    dis.nodes[j]   <- fit.nodes$diversity$mean_disagreement
    df.nodes[j]    <- fit.nodes$diversity$mean_double_fault
    jac.nodes[j]   <- fit.nodes$diversity$mean_jaccard
    inc.nodes[, j] <- fit.nodes$importance_split$InclusionRate
    imp.nodes[, j] <- .scale_to_zero_one(fit.nodes$importance_split$MeanDecreaseCustomLoss)

    # CRF considering all features at every node (sample_nodes=FALSE)
    fit.all <- CRF(
      data = df.tr, target = 'y', environment = 'z', features = features,
      n.trees = n.trees, sample_nodes = FALSE,
      lambda1 = lam[1], lambda2 = lam[2],
      min.leaf.size = min.leaf.size, n.thresh = n.thresh,
      eps.improvement = eps.improvement,
      compute.oob = FALSE, importance = FALSE,
      parallel = TRUE, n.cores = n.cores,
      relax.constraint = relax.constraint,
      compute_diversity = TRUE, diversity_data = df.te,
      seed = seed
    )
    prob.all <- predict(fit.all, df.te, type = 'prob')[, 2]
    r.all    <- .assess_acc_auc(df.te$y, prob.all)
    acc.all[j]   <- r.all$accuracy
    auc.all[j]   <- r.all$auc
    dis.all[j]   <- fit.all$diversity$mean_disagreement
    df.all[j]    <- fit.all$diversity$mean_double_fault
    jac.all[j]   <- fit.all$diversity$mean_jaccard
    inc.all[, j] <- fit.all$importance_split$InclusionRate
    imp.all[, j] <- .scale_to_zero_one(fit.all$importance_split$MeanDecreaseCustomLoss)
  }

  list(
    acc.nodes = acc.nodes, auc.nodes = auc.nodes,
    dis.nodes = dis.nodes, df.nodes  = df.nodes,  jac.nodes = jac.nodes,
    inc.nodes = inc.nodes, imp.nodes = imp.nodes,
    acc.all  = acc.all,  auc.all  = auc.all,
    dis.all  = dis.all,  df.all   = df.all,   jac.all  = jac.all,
    inc.all  = inc.all,  imp.all  = imp.all
  )
}


# Local helper: binary accuracy and AUC from probability scores.
.assess_acc_auc <- function(y_true, y_prob) {
  y_true   <- factor(y_true)
  lev      <- levels(y_true)
  y_pred   <- factor(ifelse(y_prob >= 0.5, lev[2], lev[1]), levels = lev)
  acc      <- mean(y_pred == y_true)
  pred_obj <- ROCR::prediction(y_prob, y_true)
  auc      <- ROCR::performance(pred_obj, measure = "auc")@y.values[[1]]
  list(accuracy = acc, auc = auc)
}

.scale_to_zero_one <- function(x) {
  r <- max(x) - min(x)
  if (r == 0) return(x)
  (x - min(x)) / r
}
