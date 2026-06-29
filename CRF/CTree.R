library(rpart)
library(rpart.plot)
library(dplyr)
library(igraph)
library(treeClust)
source('CRF/CTree_pruning.R')

CTree <- function(data, target, environment, features, lambda1=5, lambda2=5, min.leaf.size=3, 
                       prune=F, eps.improvement=1e-2, cv_selection = F, n.thresh=NULL, K.prune=10,
                       rule.prune = "1se", verbose=FALSE, relax.constraint =FALSE) {
  # Create binary decision tree that enforces invariance constraints
  # Arguments:
  # - data: data frame containing the dataset
  # - target: name of the target column (Y) as a string
  # - environment: name of the environmental variable (Z) as a string
  # - features: vector of feature column names to consider for splitting
  # - lambda1: weighting parameter for the mutual information, related to condition (i) 
  # - lambda2: weighting parameter for the environmental entropy change, related to condition (ii) 
  # - min.leaf.size: minimum size of final leaves
  # - eps.improvement: the loss improvement threshold for performing a split
  # - cv_selection: should lambda1/lambda2 be selected with cross-validation?
  # - n.thresh: number of thresholds to consider for the splitting. If NULL, all observed values are considered.
  
  if(cv_selection){
    if(length(unique(data[[environment]]))<3) stop('Not enough environments to do cross-validation!')
    if(verbose) cat('Starting LOEO-CV for selecting causal constraint penalty parameters.\n')
    if(any(lambda1!=0) & length(lambda1)<2) lambda1 = seq(0,5, by=1)
    if(any(lambda2!=0) & length(lambda2)<2) lambda2 = seq(0,5, by=1)
    if(is.null(n.thresh)) n.thresh=20
    cv.res = cv_select_tree(data, target, environment, features, lambda1, lambda2, min.leaf.size,
                            eps.improvement=eps.improvement, n.thresh=n.thresh,relax.constraint=relax.constraint)
    lambda1 = cv.res$lambda1.opt
    lambda2 = cv.res$lambda2.opt
    if(verbose) cat('CV Finished. \n')
  }
  
  if(verbose) cat('Starting to grow tree. \n')
  tree = grow_tree(data, target, environment, features, lambda1, lambda2, min.leaf.size,
                   loss_val=Inf, eps.improvement=eps.improvement, n.thresh=n.thresh, verbose=verbose, 
                   relax.constraint=relax.constraint)
  
  if(!is.list(tree)){
    if(verbose) cat('Warning: no splits found to improve loss function. Try reducing lambda1 and lambda2. \n')
    return(list(lambda1=as.numeric(lambda1), lambda2=as.numeric(lambda2)))
  }
  
  if(verbose) {
    robust_rpart <- convert_to_rpart(tree, data, target_col = target)
    cat('Finished growing tree. Currently at ', nrow(robust_rpart$splits), ' splits and ',
        length(unique(robust_rpart$where)) , ' leaf nodes. \n' )
    cat('Features: ', sort(unique(robust_rpart$frame$var[robust_rpart$frame$var!='<leaf>'])), '\n')
  }
  
  if(prune){
    if(verbose) cat('Starting tree pruning. \n')
    tree= RT_prune_with_cv(tree, data, target, environment, features, lambda1, lambda2,
                           K = K.prune, seed = 1, rule = rule.prune, min.leaf.size = min.leaf.size,
                           eps.improvement = eps.improvement, n.thresh = n.thresh, verbose=verbose, 
                           relax.constraint=relax.constraint)
    if(verbose) {
      robust_rpart <- convert_to_rpart(tree, data, target_col = target)
      cat('Finished tree pruning. Final tree has ', nrow(robust_rpart$splits), ' splits and ',
          length(unique(robust_rpart$where)) , ' leaf nodes. \n' )
    }
  }
  
  # store penalties (as before)
  tree$lambda1 = as.numeric(lambda1)
  tree$lambda2 = as.numeric(lambda2)
  
  # NEW: split-based importance (computed on the final tree structure)
  tree$importance <- CTree_importance_split(
    tree = tree,
    data = data,
    target = target,
    environment = environment,
    features = features,
    lambda1 = as.numeric(lambda1),
    lambda2 = as.numeric(lambda2),
    min.leaf.size = min.leaf.size
  )
  
  return(tree)
}


cv_select_tree = function(data, target, environment, features, lambda1, lambda2, min.leaf.size, eps.improvement, n.thresh,relax.constraint=F){
  envirs = sort(unique(data[[environment]]))
  n.envir = length(envirs)
  n.lambda1 = length(lambda1)
  n.lambda2 = length(lambda2)
  errors = matrix(0, n.lambda1, n.lambda2)
  for(z in 1:n.envir){
    # Data in fold
    data.fold = filter(data, .data[[environment]] == envirs[z])
    data.validation = filter(data, .data[[environment]] %in% envirs[-z])
    for(i in 1:n.lambda1){
      for(j in 1:n.lambda2){
        tree.fold = grow_tree(data.fold, target, environment, features, lambda1[i], lambda2[j], min.leaf.size, 
                              loss_val=Inf, eps.improvement=eps.improvement,n.thresh=n.thresh,relax.constraint=relax.constraint)
        preds.fold = predict_tree(tree.fold, data.validation)
        fold.error = mean(as.numeric(preds.fold)!=data.validation[[target]])
        errors[i,j] = errors[i,j] + fold.error
        cat("lambda1 = ", lambda1[i], ", lambda2 = ", lambda2[j], ', fold = ', envirs[z], ', error = ', fold.error, '\n')
      }
    }
  } 
  errors = errors/n.envir
  ind.opt = which(errors==min(errors),arr.ind = T)
  if(length(ind.opt)>2) ind.opt = ind.opt[1,]
  res=list()
  if(n.lambda1==1){
    res$lambda1.opt = lambda1
  }
  else{
    res$lambda1.opt = lambda1[ind.opt[1]]
  }
  if(n.lambda2==1){
    res$lambda2.opt = lambda2
  }
  else{
    res$lambda2.opt = lambda2[ind.opt[2]]
  }
  return(res)
}

grow_tree <- function(data, target, environment, features, lambda1, lambda2, min.leaf.size, 
                      loss_val, eps.improvement, n.thresh, verbose, relax.constraint=F) {
  # If only unique Y or no more features to split by, or we have already done the last allowed split with lamabda2=0, stop splitting
  if (length(unique(data[[target]])) == 1 || length(features) == 0 || lambda2<0) {
    # Return name of predicted target variable in final leaf
    if(verbose) {
      if(length(unique(data[[target]])) == 1) cat('Pure leaf: no more splits needed. \n')
      if( length(features) == 0) cat('No more features to split by. Leaf reached. \n')
      if(lambda2<0) cat('Cannot relax second constraint again. Leaf reached. \n')
    }
    return(as.character(names(table(data[[target]]))[which.max(table(data[[target]]))]))
  }
  # Otherwise, do another split
  split <- best_split_overall(data, target, environment, features, lambda1, lambda2, min.leaf.size, n.thresh)
  new_loss = split$loss
  # If improvement in loss is not big or an actual improvement, do not do the split
  if (abs(new_loss-loss_val)< eps.improvement | new_loss>loss_val) {
    # Allow relaxation of constraint #2 when no more splits give improvements in loss function
    if(lambda2>=0 & relax.constraint){ # Ensuring we only do this once, to not get domain-specific subtrees.
      if(verbose) cat('No more splits that improve current loss function. Allowing relaxation in second penalty term once. \n')
      split <- best_split_overall(data, target, environment, features, lambda1, lambda2=lambda2/10, min.leaf.size, n.thresh)
      lambda2=-1 # Making sure we stop after this split
      new_loss = split$loss
    }
    else{
      # Return name of predicted target variable in final leaf
      if(verbose & relax.constraint) cat('Cannot relax second constraint again: leaf reached. \n')
      else if(verbose) cat('Cannot improve loss function further. Leaf reached. \n')
      return(as.character(names(table(data[[target]]))[which.max(table(data[[target]]))]))
    }
  }
  # If no further features to split by were found, no split is done (is never true? as we now allow repeated splits on one variable)
  if (is.null(split$feature)) {
    # Return name of predicted target variable in final leaf
    if(verbose) cat('No more features to split by: leaf reached. \n')
    return(as.character(names(table(data[[target]]))[which.max(table(data[[target]]))]))
  }
  # Save tree as a list
  tree <- list()
  # The feature the split is done by, and the threshold value used
  tree$feature <- split$feature
  tree$threshold <- split$threshold
  tree$nodes <- list()
  # Identify left and right leaves according to feature threshold
  left <- filter(data, .data[[split$feature]] < split$threshold)
  right <- filter(data, .data[[split$feature]] >= split$threshold)
  if(verbose) cat('Split performed. Continuing to grow tree. \n')
  # Recursively grow tree
  tree$nodes$left <- grow_tree(left, target, environment, features, lambda1, lambda2, min.leaf.size, 
                               new_loss, eps.improvement,n.thresh, verbose = verbose, relax.constraint=relax.constraint)
  tree$nodes$right <- grow_tree(right, target, environment, features, lambda1, lambda2, min.leaf.size, 
                                new_loss, eps.improvement,n.thresh, verbose = verbose, relax.constraint=relax.constraint)
 
  return(tree)
}


entropy <- function(labels) {
  # Compute general entropy
  # Estimate probability of each class in leaf by relative frequency
  proportions <- table(labels) / length(labels)
  return(-sum(proportions * log2(proportions), na.rm = TRUE))
}

conditional_entropy <- function(data, target, split_variable, threshold, min.leaf.size) {
  # Compute the conditional entropy after a given split (weighted)
  left <- filter(data, .data[[split_variable]] < threshold)
  right <- filter(data, .data[[split_variable]] >= threshold)
  n_total <- nrow(data)
  n_left <- nrow(left)
  n_right <- nrow(right)
  # Do not allow split that leaves either leaf too small
  if (n_left < min.leaf.size || n_right < min.leaf.size) return(Inf)
  h_left <- entropy(left[[target]])
  h_right <- entropy(right[[target]])
  # Weighted entropy for the two leaves
  return((n_left / n_total) * h_left + (n_right / n_total) * h_right)
}

conditional_mutual_information <- function(data, target, environment, split_variable, threshold, min.leaf.size) {
  # Compute the mutual information I(Y; Z | X) after a given split (weighted)
  left <- filter(data, .data[[split_variable]] < threshold)
  right <- filter(data, .data[[split_variable]] >= threshold)
  n_total <- nrow(data)
  n_left <- nrow(left)
  n_right <- nrow(right)
  # Do not allow split that leaves either leaf too small
  if (n_left < min.leaf.size || n_right < min.leaf.size) return(Inf)
  mi_left <- mutual_information(left, target, environment)
  mi_right <- mutual_information(right, target, environment)
  #Return weighted MI for the two leaves
  return((n_left / n_total) * mi_left + (n_right / n_total) * mi_right)
}

mutual_information <- function(data, target, environment) {
  # Compute mutual information I(Y; Z)
  # Make matrix table of joint distribution
  joint_distribution <- table(data[[target]], data[[environment]]) / nrow(data)
  # Compute marginal distributions
  marginal_target <- rowSums(joint_distribution)
  marginal_env <- colSums(joint_distribution)
  mi <- 0
  # Loop over y and z 
  for (i in seq_along(marginal_target)) {
    for (j in seq_along(marginal_env)) {
      if (joint_distribution[i, j] > 0) {
        mi <- mi + joint_distribution[i, j] * log2(joint_distribution[i, j] / 
                                                     (marginal_target[i] * marginal_env[j]))
      }
    }
  }
  return(mi)
}


loss_function <- function(data, target, environment, feature, threshold, lambda1, lambda2, min.leaf.size) {
  # Compute the loss for a given split
  h_y_given_x <- conditional_entropy(data, target, feature, threshold, min.leaf.size)
  # Compute the loss for the combination of constraints (i) and (ii)
  h_z_given_x <- conditional_entropy(data, target=environment, feature, threshold, min.leaf.size)
  h_z <- entropy(data[[environment]])
  #h_z_diff <- abs(h_z_given_x-h_z)
  h_z_diff <- h_z - h_z_given_x
  #i_y_z_given_x <- abs(mutual_information(data, target, environment)-
  #                       conditional_mutual_information(data, target, environment, feature, threshold, min.leaf.size))
  i_y_z_given_x <- conditional_mutual_information(data, target, environment, feature, threshold, min.leaf.size)
  if(h_y_given_x == Inf &  i_y_z_given_x==Inf & abs(h_z_diff)==Inf) return(Inf)
  # Avoid multiplication of 0 with Inf (gives NaN)
  if(lambda1 == 0 & i_y_z_given_x ==Inf) i_y_z_given_x  = 0
  if(lambda2 == 0 &  abs(h_z_diff) == Inf)  h_z_diff  = 0
  return(h_y_given_x + lambda1 * i_y_z_given_x + lambda2 * h_z_diff)
}


best_split <- function(data, target, environment, feature, lambda1, lambda2, min.leaf.size, n.thresh=NULL) {
  # Find the best split threshold for a given feature to split by
  # Consider all observed feature values as a threshold
  if(!is.null(n.thresh)){
    unique_values = quantile(data[[feature]],seq(0,1,length.out=n.thresh))
    if(length(unique(data[[feature]])) < length(unique_values)){
      unique_values = sort(unique(data[[feature]]))
    }
  }
  else {
    unique_values <- unique(data[[feature]])
  }
  best_threshold <- NULL
  lowest_loss <- Inf
  
  for (threshold in unique_values) {
    loss <- loss_function(data, target, environment, feature, threshold, lambda1, lambda2, min.leaf.size)
    if (loss < lowest_loss) {
      lowest_loss <- loss
      best_threshold <- threshold
    }
  }
  return(list(threshold = best_threshold, loss = lowest_loss))
}


best_split_overall <- function(data, target, environment, features, lambda1, lambda2, min.leaf.size, n.thresh=NULL) {
  # Find the best feature to split by, and best threshold
  best_feature <- NULL
  best_threshold <- NULL
  lowest_loss <- Inf
  # Consider all features
  for (feature in features) {
    split <- best_split(data, target, environment, feature, lambda1, lambda2, min.leaf.size, n.thresh)
    if (split$loss < lowest_loss) {
      lowest_loss <- split$loss
      best_feature <- feature
      best_threshold <- split$threshold
    }
  }
  if(lowest_loss==Inf){
    best_conditional_entropy <- Inf
    best_conditional_mi <- Inf
    best_entropy_diff <- Inf
  }
  else {
    best_conditional_entropy <- conditional_entropy(data, target, best_feature, best_threshold, min.leaf.size)
    best_conditional_mi <- conditional_mutual_information(data, target, environment, best_feature, best_threshold, min.leaf.size)
    best_entropy_diff <- entropy(data[[environment]]) - conditional_entropy(data, target=environment, best_feature, best_threshold, min.leaf.size)
                               
  }
  return(list(feature = best_feature, threshold = best_threshold, loss = lowest_loss, 
              conditional_entropy = best_conditional_entropy, conditional_mi=best_conditional_mi, 
              entropy_diff = best_entropy_diff))
}

predict_tree <- function(tree, new_data) {
  # Make a new prediction
  # Arguments:
  # - tree: a decision tree generated by the `invariant_tree` function
  # - new_data: a data frame containing the new data to predict. Must have same names for target, features and environment.
  
  # Apply the traversal function to each row in the new data
  predictions <- apply(new_data, 1, function(row) traverse_tree(tree, as.list(row)))
  
  return(predictions)
}

traverse_tree <- function(tree, row) {
  # Traverse the tree for a single row
  # If the current node is a leaf, return the prediction
  if (!is.list(tree)) {
    return(tree)
  }
  # Extract the splitting feature and threshold
  feature <- tree$feature
  threshold <- tree$threshold
  # Check the value of the feature for the current row and traverse accordingly
  if (row[[feature]] < threshold) {
    return(traverse_tree(tree$nodes$left, row))
  } else {
    return(traverse_tree(tree$nodes$right, row))
  }
}

total_mutual_information = function(fit, data, target_col, envir){
  # Compute the mutual information of a whole tree
  # Arguments:
  # - fit: The tree in the format of an rpart object
  # - data: The full data set
  # - target_col: Name of the target variable
  # - envir: Name of the environment variable
  
  # Identify which leaf each observation belongs to
  where = treeClust::rpart.predict.leaves(fit, data, type='where')
  
  leaves = unique(where)
  n.leaves = length(leaves)
  N = nrow(data)
  mi_total = 0
  for(i in 1:n.leaves){
    dat.leaf = data[which(where==leaves[i]),]
    mi_leaf = mutual_information(dat.leaf, target_col, envir)
    if(mi_leaf==Inf) mi_leaf = 0
    mi_total = mi_total + nrow(dat.leaf)/N * mi_leaf
  }
  return(mi_total)
}

total_entropy_change = function(fit, data, envir){
  # Compute the mutual information of a whole tree
  # Arguments:
  # - fit: The tree in the format of an rpart object
  # - data: The full data set
  # - envir: Name of the environment variable
  
  # Identify which leaf each observation belongs to
  where = treeClust::rpart.predict.leaves(fit, data, type='where')
  
  leaves = unique(where)
  n.leaves = length(leaves)
  N = nrow(data)
  h_z_root = entropy(data[[envir]]) 
  h_z_leaves = 0
  for(i in 1:n.leaves){
    dat.leaf = data[which(where==leaves[i]),]
    h_z_leaf = entropy(dat.leaf[[envir]])
    h_z_leaves = h_z_leaves + nrow(dat.leaf)/N * h_z_leaf
  }
  return(h_z_root-h_z_leaves)
}


# Split-based importance for CTree (post-hoc on final tree)
.gini_impurity_RT <- function(y, classes) {
  tab <- table(factor(as.character(y), levels = classes))
  p <- as.numeric(tab) / sum(tab)
  1 - sum(p^2)
}

CTree_importance_split <- function(tree, data, target, environment, features,
                                        lambda1, lambda2, min.leaf.size) {
  # Computes split-based importance by walking the FINAL tree
  # and accumulating node-size-weighted gains per feature.
  #
  # Returns a data.frame with rownames = features and columns:
  # - TimesUsed
  # - MeanDecreaseGini
  # - MeanDecreaseCustomLoss
  
  classes <- sort(unique(as.character(data[[target]])))
  
  imp_gini  <- setNames(numeric(length(features)), features)
  imp_loss  <- setNames(numeric(length(features)), features)
  times_used <- setNames(integer(length(features)), features)
  
  # Recursive walk over nodes; idx are the row indices reaching this node
  walk <- function(node, idx) {
    # leaf in your representation is NOT a list
    if (!is.list(node)) return(invisible(NULL))
    
    feat <- node$feature
    thr  <- node$threshold
    
    x <- data[[feat]][idx]
    left_idx  <- idx[which(x < thr)]
    right_idx <- idx[which(x >= thr)]
    
    # If something degenerate happens (e.g. after pruning), stop safely
    if (length(left_idx) == 0 || length(right_idx) == 0) return(invisible(NULL))
    
    n_node <- length(idx)
    
    # ---- MeanDecreaseGini (RF-style) ----
    g_parent <- .gini_impurity_RT(data[[target]][idx], classes)
    g_left   <- .gini_impurity_RT(data[[target]][left_idx], classes)
    g_right  <- .gini_impurity_RT(data[[target]][right_idx], classes)
    
    w_left  <- length(left_idx) / n_node
    w_right <- length(right_idx) / n_node
    g_gain  <- g_parent - (w_left * g_left + w_right * g_right)
    
    imp_gini[feat] <<- imp_gini[feat] + n_node * g_gain
    times_used[feat] <<- times_used[feat] + 1L
    
    # ---- MeanDecreaseCustomLoss (diagnostic, matches RobustRF code) ----
    # Baseline unsplit loss at node: H(Y) + lambda1 * I(Y;Z)
    # (lambda2 term is 0 without an X split)
    dat_node <- data[idx, , drop = FALSE]
    mi_node <- as.numeric(mutual_information(dat_node, target, environment))
    if (!is.finite(mi_node) || is.na(mi_node)) mi_node <- 0
    
    loss_unsplit <- entropy(dat_node[[target]]) + lambda1 * mi_node
    
    # Split loss using your existing loss function at this node's split
    loss_split <- as.numeric(loss_function(dat_node, target, environment, feat, thr,
                                           lambda1, lambda2, min.leaf.size))
    if (!is.finite(loss_split) || is.na(loss_split)) loss_split <- Inf
    
    loss_gain <- max(0, loss_unsplit - loss_split)
    if (!is.finite(loss_gain) || is.na(loss_gain)) loss_gain <- 0
    
    imp_loss[feat] <<- imp_loss[feat] + n_node * loss_gain
    
    # recurse
    walk(node$nodes$left,  left_idx)
    walk(node$nodes$right, right_idx)
    invisible(NULL)
  }
  
  walk(tree, seq_len(nrow(data)))
  
  data.frame(
    Feature = features,
    TimesUsed = as.numeric(times_used[features]),
    MeanDecreaseGini = as.numeric(imp_gini[features]),
    MeanDecreaseCustomLoss = as.numeric(imp_loss[features]),
    row.names = features,
    check.names = FALSE
  )
}


