rm(list=ls())
source('CRF/CTree.R')
source('CRF/CRF.R')
source('functions/help_funcs/convert_to_rpart.R')
source("functions/other_methods/apply_othermethods.R")
source('large_simulations/sim_funcs/generate_data_large.R')
source('large_simulations/sim_funcs/print_funcs_large.R')
source('large_simulations/sim_funcs/sim_funcs_large.R')


# Do simulations for larger graphs. 

run_model_a = T
run_model_b = T
run_model_c = T
run_model_d = T
run_model_e = T

print_all=T

# Do simulations for larger graphs. 

N=10 # Number of replicates
n=800 # Sample size

lambdas_comb = matrix(c(0, 0,
                        1,0,
                        0,1,
                        1,1,
                        5,0,
                        0,5,
                        5, 5,
                        10, 10), ncol=2, byrow=T)
gamma.anchor=c(1, 5, 10)
lambda.irf = c(1, 5,10)


lambdas_comb = matrix(c(5, 5,
                        10, 10), ncol=2, byrow=T)
gamma.anchor=c(1, 10)
lambda.irf = c(1,10)
n=100
N=2

# Model (a)
if(run_model_a){
  set.seed(123)
  res.sim.a = perform_sim_tree_large(N, n, model='a',lambdas = lambdas_comb,gamma.anchor= gamma.anchor,
                               lambda.irf=lambda.irf,min.leaf.size = 5, n.thresh = 30)
  print_tree_sim_res_large(res.sim.a, lambdas_comb, gamma.anchor, lambda.irf, setting='trees')
  print_tree_sim_res_large(res.sim.a, lambdas_comb, gamma.anchor, lambda.irf, setting='invariant')
}

# Model (b)
if(run_model_b){
  set.seed(123)
  res.sim.b = perform_sim_tree_large(N, n, model='b',lambdas = lambdas_comb,gamma.anchor= gamma.anchor,
                               lambda.irf=lambda.irf, min.leaf.size = 5, n.thresh = 30)
  print_tree_sim_res_large(res.sim.b, lambdas_comb, gamma.anchor, lambda.irf, setting='trees')
  print_tree_sim_res_large(res.sim.b, lambdas_comb, gamma.anchor, lambda.irf, setting='invariant')
}

# Model (c)
if(run_model_c){
  set.seed(123)
  res.sim.c = perform_sim_tree_large(N, n, model='c',lambdas = lambdas_comb,gamma.anchor= gamma.anchor,
                               lambda.irf=lambda.irf,min.leaf.size = 5, n.thresh = 30)
  print_tree_sim_res_large(res.sim.c, lambdas_comb, gamma.anchor, lambda.irf, setting='trees')
  print_tree_sim_res_large(res.sim.c, lambdas_comb, gamma.anchor, lambda.irf, setting='invariant')
}

# Model (d)
if(run_model_d){
  set.seed(123)
  res.sim.d = perform_sim_tree_large(N, n, model='d',lambdas = lambdas_comb,gamma.anchor= gamma.anchor,
                               lambda.irf=lambda.irf, min.leaf.size = 5, n.thresh = 30)
  print_tree_sim_res_large(res.sim.d, lambdas_comb, gamma.anchor, lambda.irf, setting='trees')
  print_tree_sim_res_large(res.sim.d, lambdas_comb, gamma.anchor, lambda.irf, setting='invariant')
}

# Model (e)
if(run_model_e){
  set.seed(123)
  res.sim.e = perform_sim_tree_large(N, n, model='e',lambdas = lambdas_comb,gamma.anchor= gamma.anchor,
                               lambda.irf=lambda.irf, min.leaf.size = 5, n.thresh = 30)
  print_tree_sim_res_large(res.sim.e, lambdas_comb, gamma.anchor, lambda.irf, setting='trees')
  print_tree_sim_res_large(res.sim.e, lambdas_comb, gamma.anchor, lambda.irf, setting='invariant')
}


if(print_all){
  print_res_aggregated(res.sim.a, res.sim.b, res.sim.c, res.sim.d, res.sim.e, lambdas_comb, gamma.anchor, lambda.irf, sd=F)
}

