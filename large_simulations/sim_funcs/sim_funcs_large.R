library(ROCR)



perform_sim_tree_large = function(N, n, d=c(5,5,5,5,5),
                                  n.test=300, model='a',
                                  lambdas= matrix(c(5,5,
                                                    10,10), nrow=2, byrow=T),
                                  gamma.anchor=c(1, 5),lambda.irf = c(1, 5),
                                  min.leaf.size=5, CRF_sample_nodes=F,
                                  eps.improvement=1e-2, n.thresh=50, n.cores=30,num.trees=200,
                                  include_CTree = TRUE, include_CRF = TRUE, include_rpart = TRUE, include_rf = TRUE,
                                  include_icp_rpart = TRUE, include_icp_rf = TRUE, include_anchor = TRUE,
                                  include_anchor_boost = TRUE, include_irf = TRUE, prune=FALSE, normalise=F,
                                  n_test_domains=1, more.envirs=F,relax.constraint=TRUE){
  p=5 # The number of blocks, not total number of features.
  n.lambdacombs = max(1,nrow(lambdas))
  n.gamma = length(gamma.anchor)
  n.lambda.irf = length(lambda.irf)
  
  accuracy.vals.CTree = matrix(0,N,n.lambdacombs)
  accuracy.vals.CRF = matrix(0,N,n.lambdacombs)
  accuracy.vals.rpart = rep(0,N)
  accuracy.vals.rf = rep(0,N)
  accuracy.vals.icp_rpart = rep(0,N)
  accuracy.vals.icp_rf = rep(0,N)
  accuracy.vals.anchor= rep(0,N)
  accuracy.vals.anchor_boost = matrix(0,N,n.gamma)
  accuracy.vals.irf = matrix(0,N,n.lambda.irf)
  accuracy.vals.train.CTree = matrix(0,N,n.lambdacombs)
  accuracy.vals.train.CRF = matrix(0,N,n.lambdacombs)
  accuracy.vals.train.rpart = rep(0,N)
  accuracy.vals.train.rf = rep(0,N)
  accuracy.vals.train.icp_rpart = rep(0,N)
  accuracy.vals.train.icp_rf = rep(0,N)
  accuracy.vals.train.anchor= rep(0,N)
  accuracy.vals.train.anchor_boost = matrix(0,N,n.gamma)
  accuracy.vals.train.irf = matrix(0,N,n.lambda.irf)
  
  auc.vals.CTree = matrix(0,N,n.lambdacombs)
  auc.vals.CRF = matrix(0,N,n.lambdacombs)
  auc.vals.rpart = rep(0,N)
  auc.vals.rf = rep(0,N)
  auc.vals.icp_rpart = rep(0,N)
  auc.vals.icp_rf = rep(0,N)
  auc.vals.anchor= rep(0,N)
  auc.vals.anchor_boost = matrix(0,N,n.gamma)
  auc.vals.irf = matrix(0,N,n.lambda.irf)
  auc.vals.train.CTree = matrix(0,N,n.lambdacombs)
  auc.vals.train.CRF = matrix(0,N,n.lambdacombs)
  auc.vals.train.rpart = rep(0,N)
  auc.vals.train.rf = rep(0,N)
  auc.vals.train.icp_rpart = rep(0,N)
  auc.vals.train.icp_rf = rep(0,N)
  auc.vals.train.anchor= rep(0,N)
  auc.vals.train.anchor_boost = matrix(0,N,n.gamma)
  auc.vals.train.irf = matrix(0,N,n.lambda.irf)
  
  rho.i.vals.train.CTree = matrix(0,N,n.lambdacombs)
  rho.i.vals.train.CRF = matrix(0,N,n.lambdacombs)
  rho.i.vals.train.rpart = rep(0,N)
  rho.i.vals.train.rf = rep(0,N)
  rho.i.vals.train.icp_rpart = rep(0,N)
  rho.i.vals.train.icp_rf = rep(0,N)
  rho.i.vals.train.anchor= rep(0,N)
  rho.i.vals.train.anchor_boost = matrix(0,N,n.gamma)
  rho.i.vals.train.irf = matrix(0,N,n.lambda.irf)
  
  rho.ii.vals.train.CTree = matrix(0,N,n.lambdacombs)
  rho.ii.vals.train.CRF = matrix(0,N,n.lambdacombs)
  rho.ii.vals.train.rpart = rep(0,N)
  rho.ii.vals.train.rf = rep(0,N)
  rho.ii.vals.train.icp_rpart = rep(0,N)
  rho.ii.vals.train.icp_rf = rep(0,N)
  rho.ii.vals.train.anchor= rep(0,N)
  rho.ii.vals.train.anchor_boost = matrix(0,N,n.gamma)
  rho.ii.vals.train.irf = matrix(0,N,n.lambda.irf)
  
  # IMPORTANT: p is now sum(d)
  p_eff = sum(d)
  
  var_importance.CTree = array(0,c(N,p_eff,n.lambdacombs))
  var_inclusion.CTree  = array(0,c(N,p_eff,n.lambdacombs))
  var_importance.CRF   = array(0,c(N,p_eff,n.lambdacombs))
  var_inclusion.CRF    = array(0,c(N,p_eff,n.lambdacombs))
  var_importance.rpart      = matrix(0,N,p_eff)
  var_inclusion.rpart       = matrix(0,N,p_eff)
  var_importance.rf         = matrix(0,N,p_eff)
  var_inclusion.icp_rpart   = matrix(0,N,p_eff)
  var_importance.icp_rpart  = matrix(0,N,p_eff)
  var_importance.icp_rf     = matrix(0,N,p_eff)
  var_importance.anchor     = matrix(0,N,p_eff)
  var_importance.anchor_boost = array(0,c(N,p_eff,n.gamma))
  var_importance.irf        = array(0,c(N,p_eff,n.lambda.irf))
  
  seeds = sample(1:10000,size=N)
  for(i in 1:N){
    res.tmp = perform_sim_tree_oneiteration(n,
                                            n.test=n.test,
                                            model=model,
                                            lambdas_comb = lambdas,
                                            gamma.anchor=gamma.anchor,
                                            lambda.irf = lambda.irf,
                                            min.leaf.size=min.leaf.size,
                                            CRF_sample_nodes=CRF_sample_nodes,
                                            eps.improvement=eps.improvement,
                                            n.thresh=n.thresh,
                                            n.cores=n.cores,
                                            num.trees=num.trees,
                                            include_CTree = include_CTree,
                                            include_CRF = include_CRF,
                                            include_rpart = include_rpart,
                                            include_rf = include_rf,
                                            include_icp_rpart = include_icp_rpart,
                                            include_icp_rf = include_icp_rf,
                                            include_anchor = include_anchor,
                                            include_anchor_boost = include_anchor_boost,
                                            include_irf = include_irf,
                                            prune=prune,
                                            normalise=normalise,
                                            n_test_domains=n_test_domains,
                                            more.envirs=more.envirs,
                                            relax.constraint=relax.constraint,
                                            seed=seeds[i],
                                            d=d)
    
    if(include_CTree){
      accuracy.vals.CTree[i,] = res.tmp$accuracy.val.CTree
      accuracy.vals.train.CTree[i,] = res.tmp$accuracy.val.train.CTree
      auc.vals.CTree[i,] = res.tmp$auc.val.CTree
      auc.vals.train.CTree[i,] = res.tmp$auc.val.train.CTree
      rho.i.vals.train.CTree[i,] = res.tmp$rho.i.val.train.CTree
      rho.ii.vals.train.CTree[i,] = res.tmp$rho.ii.val.train.CTree
      var_importance.CTree[i,,] = res.tmp$var_importance.CTree
      var_inclusion.CTree[i,,]  = res.tmp$var_inclusion.CTree
    }
    if(include_CRF){
      accuracy.vals.CRF[i,] = res.tmp$accuracy.val.CRF
      accuracy.vals.train.CRF[i,] = res.tmp$accuracy.val.train.CRF
      auc.vals.CRF[i,] = res.tmp$auc.val.CRF
      auc.vals.train.CRF[i,] = res.tmp$auc.val.train.CRF
      rho.i.vals.train.CRF[i,] = res.tmp$rho.i.val.train.CRF
      rho.ii.vals.train.CRF[i,] = res.tmp$rho.ii.val.train.CRF
      var_importance.CRF[i,,] = res.tmp$var_importance.CRF
      var_inclusion.CRF[i,,]  = res.tmp$var_inclusion.CRF
    }
    if(include_rpart){
      accuracy.vals.rpart[i] = res.tmp$accuracy.val.rpart
      accuracy.vals.train.rpart[i] = res.tmp$accuracy.val.train.rpart
      auc.vals.rpart[i] = res.tmp$auc.val.rpart
      auc.vals.train.rpart[i] = res.tmp$auc.val.train.rpart
      rho.i.vals.train.rpart[i] = res.tmp$rho.i.val.train.rpart
      rho.ii.vals.train.rpart[i] = res.tmp$rho.ii.val.train.rpart
      var_importance.rpart[i,] = res.tmp$var_importance.rpart
      var_inclusion.rpart[i,]  = res.tmp$var_inclusion.rpart
    }
    if(include_rf){
      accuracy.vals.rf[i] = res.tmp$accuracy.val.rf
      accuracy.vals.train.rf[i] = res.tmp$accuracy.val.train.rf
      auc.vals.rf[i] = res.tmp$auc.val.rf
      auc.vals.train.rf[i] = res.tmp$auc.val.train.rf
      rho.i.vals.train.rf[i] = res.tmp$rho.i.val.train.rf
      rho.ii.vals.train.rf[i] = res.tmp$rho.ii.val.train.rf
      var_importance.rf[i,] = res.tmp$var_importance.rf
    }
    if(include_icp_rpart){
      accuracy.vals.icp_rpart[i] = res.tmp$accuracy.val.icp_rpart
      accuracy.vals.train.icp_rpart[i] = res.tmp$accuracy.val.train.icp_rpart
      auc.vals.icp_rpart[i] = res.tmp$auc.val.icp_rpart
      auc.vals.train.icp_rpart[i] = res.tmp$auc.val.train.icp_rpart
      rho.i.vals.train.icp_rpart[i] = res.tmp$rho.i.val.train.icp_rpart
      rho.ii.vals.train.icp_rpart[i] = res.tmp$rho.ii.val.train.icp_rpart
      var_importance.icp_rpart[i,] = res.tmp$var_importance.icp_rpart
      var_inclusion.icp_rpart[i,]  = res.tmp$var_inclusion.icp_rpart
    }
    if(include_icp_rf){
      accuracy.vals.icp_rf[i] = res.tmp$accuracy.val.icp_rf
      accuracy.vals.train.icp_rf[i] = res.tmp$accuracy.val.train.icp_rf
      auc.vals.icp_rf[i] = res.tmp$auc.val.icp_rf
      auc.vals.train.icp_rf[i] = res.tmp$auc.val.train.icp_rf
      rho.i.vals.train.icp_rf[i] = res.tmp$rho.i.val.train.icp_rf
      rho.ii.vals.train.icp_rf[i] = res.tmp$rho.ii.val.train.icp_rf
      var_importance.icp_rf[i,] = res.tmp$var_importance.icp_rf
    }
    if(include_anchor){
      accuracy.vals.anchor[i] = res.tmp$accuracy.val.anchor
      accuracy.vals.train.anchor[i] = res.tmp$accuracy.val.train.anchor
      auc.vals.anchor[i] = res.tmp$auc.val.anchor
      auc.vals.train.anchor[i] = res.tmp$auc.val.train.anchor
      rho.i.vals.train.anchor[i] = res.tmp$rho.i.val.train.anchor
      rho.ii.vals.train.anchor[i] = res.tmp$rho.ii.val.train.anchor
      var_importance.anchor[i,] = res.tmp$var_importance.anchor
    }
    if(include_anchor_boost){
      accuracy.vals.anchor_boost[i,] = res.tmp$accuracy.val.anchor_boost
      accuracy.vals.train.anchor_boost[i,] = res.tmp$accuracy.val.train.anchor_boost
      auc.vals.anchor_boost[i,] = res.tmp$auc.val.anchor_boost
      auc.vals.train.anchor_boost[i,] = res.tmp$auc.val.train.anchor_boost
      rho.i.vals.train.anchor_boost[i,] = res.tmp$rho.i.val.train.anchor_boost
      rho.ii.vals.train.anchor_boost[i,] = res.tmp$rho.ii.val.train.anchor_boost
      var_importance.anchor_boost[i,,] = res.tmp$var_importance.anchor_boost
    }
    if(include_irf){
      accuracy.vals.irf[i,] = res.tmp$accuracy.val.irf
      accuracy.vals.train.irf[i,] = res.tmp$accuracy.val.train.irf
      auc.vals.irf[i,] = res.tmp$auc.val.irf
      auc.vals.train.irf[i,] = res.tmp$auc.val.train.irf
      rho.i.vals.train.irf[i,] = res.tmp$rho.i.val.train.irf
      rho.ii.vals.train.irf[i,] = res.tmp$rho.ii.val.train.irf
      var_importance.irf[i,,] = res.tmp$var_importance.irf
    }
  }
  var_importance.CTree.full = var_importance.CTree
  var_importance.CRF.full = var_importance.CRF
  var_inclusion.CTree.full = var_inclusion.CTree
  var_inclusion.CRF.full = var_inclusion.CRF
  var_importance.CTree = apply(var_importance.CTree,c(2,3),mean)
  var_inclusion.CTree  = apply(var_inclusion.CTree,c(2,3),mean)
  var_importance.CRF   = apply(var_importance.CRF,c(2,3),mean)
  var_inclusion.CRF    = apply(var_inclusion.CRF,c(2,3),mean)
  var_importance.rpart      = colMeans(var_importance.rpart)
  var_inclusion.rpart       = colMeans(var_inclusion.rpart)
  var_importance.rf         = colMeans(var_importance.rf)
  var_importance.icp_rpart  = colMeans(var_importance.icp_rpart)
  var_inclusion.icp_rpart   = colMeans(var_inclusion.icp_rpart)
  var_importance.icp_rf     = colMeans(var_importance.icp_rf)
  var_importance.anchor     = colMeans(var_importance.anchor)
  var_importance.anchor_boost= apply(var_importance.anchor_boost,c(2,3),mean)
  var_importance.irf        = apply(var_importance.irf,c(2,3),mean)
  
  res = list(
    accuracy.vals.CTree = accuracy.vals.CTree,
    accuracy.vals.CRF = accuracy.vals.CRF,
    accuracy.vals.rpart = accuracy.vals.rpart,
    accuracy.vals.rf = accuracy.vals.rf,
    accuracy.vals.icp_rpart = accuracy.vals.icp_rpart,
    accuracy.vals.icp_rf = accuracy.vals.icp_rf,
    accuracy.vals.anchor=accuracy.vals.anchor,
    accuracy.vals.anchor_boost = accuracy.vals.anchor_boost,
    accuracy.vals.irf = accuracy.vals.irf,
    accuracy.vals.train.CTree = accuracy.vals.train.CTree,
    accuracy.vals.train.CRF = accuracy.vals.train.CRF,
    accuracy.vals.train.rpart = accuracy.vals.train.rpart,
    accuracy.vals.train.rf = accuracy.vals.train.rf,
    accuracy.vals.train.icp_rpart = accuracy.vals.train.icp_rpart,
    accuracy.vals.train.icp_rf = accuracy.vals.train.icp_rf,
    accuracy.vals.train.anchor= accuracy.vals.train.anchor,
    accuracy.vals.train.anchor_boost = accuracy.vals.train.anchor_boost,
    accuracy.vals.train.irf = accuracy.vals.train.irf,
    
    mean.accuracy.vals.CTree = colMeans(accuracy.vals.CTree),
    mean.accuracy.vals.CRF = colMeans(accuracy.vals.CRF),
    mean.accuracy.vals.rpart = mean(accuracy.vals.rpart),
    mean.accuracy.vals.rf = mean(accuracy.vals.rf),
    mean.accuracy.vals.icp_rpart = mean(accuracy.vals.icp_rpart),
    mean.accuracy.vals.icp_rf = mean(accuracy.vals.icp_rf),
    mean.accuracy.vals.anchor=mean(accuracy.vals.anchor),
    mean.accuracy.vals.anchor_boost = colMeans(accuracy.vals.anchor_boost),
    mean.accuracy.vals.irf = colMeans(accuracy.vals.irf),
    mean.accuracy.vals.train.CTree = colMeans(accuracy.vals.train.CTree),
    mean.accuracy.vals.train.CRF = colMeans(accuracy.vals.train.CRF),
    mean.accuracy.vals.train.rpart = mean(accuracy.vals.train.rpart),
    mean.accuracy.vals.train.rf = mean(accuracy.vals.train.rf),
    mean.accuracy.vals.train.icp_rpart = mean(accuracy.vals.train.icp_rpart),
    mean.accuracy.vals.train.icp_rf = mean(accuracy.vals.train.icp_rf),
    mean.accuracy.vals.train.anchor= mean(accuracy.vals.train.anchor),
    mean.accuracy.vals.train.anchor_boost = colMeans(accuracy.vals.train.anchor_boost),
    mean.accuracy.vals.train.irf = colMeans(accuracy.vals.train.irf),
    
    auc.vals.CTree = auc.vals.CTree,
    auc.vals.CRF = auc.vals.CRF,
    auc.vals.rpart = auc.vals.rpart,
    auc.vals.rf = auc.vals.rf,
    auc.vals.icp_rpart = auc.vals.icp_rpart,
    auc.vals.icp_rf = auc.vals.icp_rf,
    auc.vals.anchor=auc.vals.anchor,
    auc.vals.anchor_boost = auc.vals.anchor_boost,
    auc.vals.irf = auc.vals.irf,
    auc.vals.train.CTree = auc.vals.train.CTree,
    auc.vals.train.CRF = auc.vals.train.CRF,
    auc.vals.train.rpart = auc.vals.train.rpart,
    auc.vals.train.rf = auc.vals.train.rf,
    auc.vals.train.icp_rpart = auc.vals.train.icp_rpart,
    auc.vals.train.icp_rf = auc.vals.train.icp_rf,
    auc.vals.train.anchor= auc.vals.train.anchor,
    auc.vals.train.anchor_boost = auc.vals.train.anchor_boost,
    auc.vals.train.irf = auc.vals.train.irf,
    
    mean.auc.vals.CTree = colMeans(auc.vals.CTree),
    mean.auc.vals.CRF = colMeans(auc.vals.CRF),
    mean.auc.vals.rpart = mean(auc.vals.rpart),
    mean.auc.vals.rf = mean(auc.vals.rf),
    mean.auc.vals.icp_rpart = mean(auc.vals.icp_rpart),
    mean.auc.vals.icp_rf = mean(auc.vals.icp_rf),
    mean.auc.vals.anchor=mean(auc.vals.anchor),
    mean.auc.vals.anchor_boost = colMeans(auc.vals.anchor_boost),
    mean.auc.vals.irf = colMeans(auc.vals.irf),
    mean.auc.vals.train.CTree = colMeans(auc.vals.train.CTree),
    mean.auc.vals.train.CRF = colMeans(auc.vals.train.CRF),
    mean.auc.vals.train.rpart = mean(auc.vals.train.rpart),
    mean.auc.vals.train.rf = mean(auc.vals.train.rf),
    mean.auc.vals.train.icp_rpart = mean(auc.vals.train.icp_rpart),
    mean.auc.vals.train.icp_rf = mean(auc.vals.train.icp_rf),
    mean.auc.vals.train.anchor= mean(auc.vals.train.anchor),
    mean.auc.vals.train.anchor_boost = colMeans(auc.vals.train.anchor_boost),
    mean.auc.vals.train.irf = colMeans(auc.vals.train.irf),
    
    rho.i.vals.train.CTree = rho.i.vals.train.CTree,
    rho.i.vals.train.CRF = rho.i.vals.train.CRF,
    rho.i.vals.train.rpart = rho.i.vals.train.rpart,
    rho.i.vals.train.rf = rho.i.vals.train.rf,
    rho.i.vals.train.icp_rpart = rho.i.vals.train.icp_rpart,
    rho.i.vals.train.icp_rf = rho.i.vals.train.icp_rf,
    rho.i.vals.train.anchor= rho.i.vals.train.anchor,
    rho.i.vals.train.anchor_boost = rho.i.vals.train.anchor_boost,
    rho.i.vals.train.irf = rho.i.vals.train.irf,
    
    mean.rho.i.vals.train.CTree = colMeans(rho.i.vals.train.CTree),
    mean.rho.i.vals.train.CRF = colMeans(rho.i.vals.train.CRF),
    mean.rho.i.vals.train.rpart = mean(rho.i.vals.train.rpart),
    mean.rho.i.vals.train.rf = mean(rho.i.vals.train.rf),
    mean.rho.i.vals.train.icp_rpart = mean(rho.i.vals.train.icp_rpart),
    mean.rho.i.vals.train.icp_rf = mean(rho.i.vals.train.icp_rf),
    mean.rho.i.vals.train.anchor= mean(rho.i.vals.train.anchor),
    mean.rho.i.vals.train.anchor_boost = colMeans(rho.i.vals.train.anchor_boost),
    mean.rho.i.vals.train.irf = colMeans(rho.i.vals.train.irf),
    
    rho.ii.vals.train.CTree = rho.ii.vals.train.CTree,
    rho.ii.vals.train.CRF = rho.ii.vals.train.CRF,
    rho.ii.vals.train.rpart = rho.ii.vals.train.rpart,
    rho.ii.vals.train.rf = rho.ii.vals.train.rf,
    rho.ii.vals.train.icp_rpart = rho.ii.vals.train.icp_rpart,
    rho.ii.vals.train.icp_rf = rho.ii.vals.train.icp_rf,
    rho.ii.vals.train.anchor= rho.ii.vals.train.anchor,
    rho.ii.vals.train.anchor_boost = rho.ii.vals.train.anchor_boost,
    rho.ii.vals.train.irf = rho.ii.vals.train.irf,
    
    mean.rho.ii.vals.train.CTree = colMeans(rho.ii.vals.train.CTree),
    mean.rho.ii.vals.train.CRF = colMeans(rho.ii.vals.train.CRF),
    mean.rho.ii.vals.train.rpart = mean(rho.ii.vals.train.rpart),
    mean.rho.ii.vals.train.rf = mean(rho.ii.vals.train.rf),
    mean.rho.ii.vals.train.icp_rpart = mean(rho.ii.vals.train.icp_rpart),
    mean.rho.ii.vals.train.icp_rf = mean(rho.ii.vals.train.icp_rf),
    mean.rho.ii.vals.train.anchor= mean(rho.ii.vals.train.anchor),
    mean.rho.ii.vals.train.anchor_boost = colMeans(rho.ii.vals.train.anchor_boost),
    mean.rho.ii.vals.train.irf = colMeans(rho.ii.vals.train.irf),
    
    var_importance.CRF.full = var_importance.CRF.full,
    var_importance.CTree.full = var_importance.CTree.full,
    var_inclusion.CTree.full = var_inclusion.CTree.full ,
    var_inclusion.CRF.full = var_inclusion.CRF.full,
    
    var_importance.CTree = var_importance.CTree,
    var_inclusion.CTree = var_inclusion.CTree,
    var_importance.CRF = var_importance.CRF,
    var_inclusion.CRF = var_inclusion.CRF,
    var_importance.rpart = var_importance.rpart,
    var_inclusion.rpart = var_inclusion.rpart,
    var_importance.rf = var_importance.rf,
    var_inclusion.icp_rpart = var_inclusion.icp_rpart,
    var_importance.icp_rpart = var_importance.icp_rpart,
    var_importance.icp_rf = var_importance.icp_rf,
    var_importance.anchor=var_importance.anchor,
    var_importance.anchor_boost = var_importance.anchor_boost,
    var_importance.irf = var_importance.irf,
    
    lambdas_comb=lambdas, gamma.anchor=gamma.anchor, lambda.irf=lambda.irf,
    d = d
  )
  return(res)
}

perform_sim_tree_oneiteration = function(n, n.test=200, model = 'a', lambdas_comb=NA,
                                         gamma.anchor=c(1, 5),lambda.irf = c(1, 5),
                                         min.leaf.size=5, CRF_sample_nodes=F,
                                         eps.improvement=1e-2, n.thresh=50, n.cores=30,num.trees=200,
                                         include_CTree = TRUE, include_CRF = TRUE, include_rpart = TRUE, include_rf = TRUE,
                                         include_icp_rpart = TRUE, include_icp_rf = TRUE, include_anchor = FALSE,
                                         include_anchor_boost = TRUE, include_irf = TRUE, prune=FALSE, normalise=F,
                                         n_test_domains=1, more.envirs=F,
                                         relax.constraint=F,seed=1,
                                         d = c(5,5,5,5,5)){
  
  set.seed(seed)
  
  # Generate training data (NEW)
  dat = generate_data_tree_large(n, model=model, more.envirs=more.envirs, new=FALSE, d=d, seed=seed)
  df.use = as.data.frame(cbind(dat$y, dat$X))
  df.use.tree = as.data.frame(cbind(dat$y, dat$X, dat$z))
  names(df.use)[1] = "y"
  names(df.use.tree)[1] = "y"
  names(df.use.tree)[ncol(df.use.tree)] = "z"
  
  features = colnames(dat$X)
  
  # Generate test data
  dat.test = generate_data_tree_large(n.test, model=model, new=TRUE, d=d, seed=seed)
  df.test.use = as.data.frame(cbind(dat.test$y, dat.test$X))
  df.test.use.tree = as.data.frame(cbind(dat.test$y, dat.test$X, dat.test$z))
  names(df.test.use)[1] = "y"
  names(df.test.use.tree)[1] = "y"
  names(df.test.use.tree)[ncol(df.test.use.tree)] = "z"
  
  # Ensure y and z are factors with aligned levels
  df.use.tree$y <- as.factor(df.use.tree$y)
  df.test.use.tree$y <- factor(df.test.use.tree$y, levels = levels(df.use.tree$y))
  
  df.use.tree$z <- factor(df.use.tree$z, levels=c(unique(df.test.use.tree$z), unique(df.use.tree$z)))
  df.test.use.tree$z <- factor(df.test.use.tree$z, levels = union(levels(df.use.tree$z), levels(as.factor(df.test.use.tree$z))))
  
  df.use.tree.no.z = df.use.tree[,-which(colnames(df.use.tree)=='z')]
  
  res=list()
  
  if (include_CTree) {
    r <- apply_CTree_core(df.use.tree, df.test.use.tree, features,
                               lambdas = lambdas_comb, n.thresh=n.thresh, eps.improvement =eps.improvement,
                               min.leaf.size = min.leaf.size, prune=prune, relax.constraint=relax.constraint)
    
    res$accuracy.val.train.CTree     <- r$accuracy_train
    res$accuracy.val.CTree           <- r$accuracy_test
    res$auc.val.CTree                <- r$auc_test
    res$auc.val.train.CTree          <- r$auc_train
    res$rho.i.val.train.CTree        <- r$rho.i
    res$rho.ii.val.train.CTree       <- r$rho.ii
    var_importance.tmp                    <- lapply(r$importance, FUN= function(s) scale_to_zero_one(s$MeanDecreaseCustomLoss))
    res$var_importance.CTree         <- matrix(unlist(var_importance.tmp), ncol= nrow(lambdas_comb))
    var_inclusion.tmp                     <- lapply(r$included_vars, FUN= function(s) features %in% s)
    res$var_inclusion.CTree          <- matrix(unlist(var_inclusion.tmp), ncol= nrow(lambdas_comb))
  }
  
  if (include_CRF) {
    r <- apply_CRF_core(df.use.tree, df.test.use.tree, features,
                             lambdas = lambdas_comb, n.thresh=n.thresh, eps.improvement =eps.improvement,
                             min.leaf.size = min.leaf.size,n.trees=num.trees, replace=T,
                             sample_nodes = CRF_sample_nodes, mtry = "rf",
                             compute.oob = F, importance = TRUE, importance.type = 'gini',
                             n.perm = 1, parallel = TRUE, n.cores = n.cores,
                             compute_rho_full = TRUE, compute_rho_oob  = F,
                             verbose = FALSE, relax.constraint= relax.constraint, seed = sample(1:1000,1))
    
    res$accuracy.val.train.CRF       <- r$accuracy_train
    res$accuracy.val.CRF             <- r$accuracy_test
    res$auc.val.CRF                  <- r$auc_test
    res$auc.val.train.CRF            <- r$auc_train
    res$rho.i.val.train.CRF          <- r$rho.i
    res$rho.ii.val.train.CRF         <- r$rho.ii
    importance_tmp                        <- lapply(r$importance, FUN=function(s) scale_to_zero_one(s$MeanDecreaseCustomLoss))
    res$var_importance.CRF           <- matrix(unlist(importance_tmp), ncol= nrow(lambdas_comb))
    inclusion_tmp                         <- lapply(r$importance, FUN=function(s) s$InclusionRate)
    res$var_inclusion.CRF            <- matrix(unlist(inclusion_tmp), ncol= nrow(lambdas_comb))
  }
  
  if (include_rpart) {
    r <- apply_rpart_core(df.use.tree.no.z, df.test.use.tree, features, df.use.tree)
    res$accuracy.val.train.rpart      <- r$accuracy_train
    res$accuracy.val.rpart            <- r$accuracy_test
    res$auc.val.train.rpart           <- r$auc_train
    res$auc.val.rpart                 <- r$auc_test
    res$rho.i.val.train.rpart         <- r$rho.i
    res$rho.ii.val.train.rpart        <- r$rho.ii
    
    res$var_importance.rpart          <- scale_to_zero_one(r$importance)
    res$var_inclusion.rpart           <- (features %in% r$included_vars)
  }
  
  if (include_rf) {
    r <- apply_rf_core(df.use.tree.no.z, df.test.use.tree, features,
                       min.leaf.size = min.leaf.size, num.trees=num.trees)
    res$accuracy.val.train.rf      <- r$accuracy_train
    res$accuracy.val.rf            <- r$accuracy_test
    res$auc.val.train.rf           <- r$auc_train
    res$auc.val.rf                 <- r$auc_test
    res$rho.i.val.train.rf         <- r$rho.i
    res$rho.ii.val.train.rf        <- r$rho.ii
    res$var_importance.rf          <- scale_to_zero_one(r$importance[features])
  }
  
  if (include_icp_rpart) {
    r <- apply_ICP_rpart_core(df.use.tree, df.test.use.tree, features)
    res$accuracy.val.train.icp_rpart      <- r$accuracy_train
    res$accuracy.val.icp_rpart            <- r$accuracy_test
    res$auc.val.train.icp_rpart           <- r$auc_train
    res$auc.val.icp_rpart                 <- r$auc_test
    res$rho.i.val.train.icp_rpart         <- r$rho.i
    res$rho.ii.val.train.icp_rpart        <- r$rho.ii
    res$var_inclusion.icp_rpart           <- (features %in% r$included_vars)
    res$var_importance.icp_rpart          <- scale_to_zero_one(r$importance)
  }
  
  if (include_icp_rf) {
    r <- apply_ICP_rf_core(df.use.tree, df.test.use.tree, features,
                           min.leaf.size = min.leaf.size, num.trees=num.trees)
    res$accuracy.val.train.icp_rf      <- r$accuracy_train
    res$accuracy.val.icp_rf            <- r$accuracy_test
    res$auc.val.train.icp_rf           <- r$auc_train
    res$auc.val.icp_rf                 <- r$auc_test
    res$rho.i.val.train.icp_rf         <- r$rho.i
    res$rho.ii.val.train.icp_rf        <- r$rho.ii
    res$var_importance.icp_rf          <- scale_to_zero_one(r$importance)
  }
  
  if (include_anchor) {
    r <- apply_anchor_core(df.use.tree, df.test.use.tree, features)
    res$accuracy.val.train.anchor      <- r$accuracy_train
    res$accuracy.val.anchor            <- r$accuracy_test
    res$auc.val.train.anchor           <- r$auc_train
    res$auc.val.anchor                 <- r$auc_test
    res$rho.i.val.train.anchor         <- r$rho.i
    res$rho.ii.val.train.anchor        <- r$rho.ii
    res$var_importance.anchor          <- (features %in% r$included_vars)
  }
  
  if (include_anchor_boost) {
    r <- apply_anchor_boost_core(df.use.tree, df.test.use.tree, features,
                                 python_module = "invariant_methods", lam=gamma.anchor)
    res$accuracy.val.train.anchor_boost      <- r$accuracy_train
    res$accuracy.val.anchor_boost            <- r$accuracy_test
    res$auc.val.train.anchor_boost           <- r$auc_train
    res$auc.val.anchor_boost                 <- r$auc_test
    res$rho.i.val.train.anchor_boost         <- r$rho.i
    res$rho.ii.val.train.anchor_boost        <- r$rho.ii
    importance_tmp                           <- lapply(r$importance, scale_to_zero_one)
    res$var_importance.anchor_boost          <- matrix(unlist(importance_tmp), ncol=length(gamma.anchor))
  }
  
  if (include_irf) {
    r <- apply_IRF_core(df.use.tree, df.test.use.tree, features, lambda=lambda.irf,
                        python_module = "invariant_methods")
    res$accuracy.val.train.irf      <- r$accuracy_train
    res$accuracy.val.irf            <- r$accuracy_test
    res$auc.val.train.irf           <- r$auc_train
    res$auc.val.irf                 <- r$auc_test
    res$rho.i.val.train.irf         <- r$rho.i
    res$rho.ii.val.train.irf        <- r$rho.ii
    importance_tmp                  <- lapply(r$importance, scale_to_zero_one)
    res$var_importance.irf          <- matrix(unlist(importance_tmp), ncol=length(lambda.irf))
  }
  
  return(res)
}

scale_to_zero_one = function(x){
  if((max(x)-min(x)) == 0) return(x)
  res = (x-min(x))/(max(x)-min(x))
  return(res)
}
