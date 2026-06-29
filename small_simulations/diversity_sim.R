rm(list = ls())
source('CRF/CTree.R')
source('CRF/CRF.R')
source('small_simulations/sim_funcs/generate_data_small.R')
source('small_simulations/sim_funcs/diversity_sim_funcs.R')
source('small_simulations/sim_funcs/print_funcs_diversity.R')

# Same simulation settings as small_sim.R
N      <- 20   # number of replicates
n      <- 300   # training sample size
n.test <- 300   # test sample size

lambdas_comb <- matrix(c( 5,  5,
                          10, 10), ncol = 2, byrow = TRUE)

features <- c('X1', 'X2', 'X3', 'X4', 'X5')


# Model (a) ----
set.seed(123)
res.div.a <- perform_diversity_sim(N, n, n.test, model = 'a',
                                   lambdas = lambdas_comb,
                                   min.leaf.size = 5, n.thresh = 30)
save(res.div.a, file = 'small_simulations/data/div_res_a.RData')
load('small_simulations/data/div_res_a.RData')
cat('\n--- Model (a) ---\n')
print_diversity_header(features)
print_diversity_sim_res(res.div.a, lambdas_comb, features)

# Model (b) ----
set.seed(123)
res.div.b <- perform_diversity_sim(N, n, n.test, model = 'b',
                                   lambdas = lambdas_comb,
                                   min.leaf.size = 5, n.thresh = 30)
save(res.div.b, file = 'small_simulations/data/div_res_b.RData')
load('small_simulations/data/div_res_b.RData')
cat('\n--- Model (b) ---\n')
print_diversity_header(features)
print_diversity_sim_res(res.div.b, lambdas_comb, features)

# Model (c) ----
set.seed(123)
res.div.c <- perform_diversity_sim(N, n, n.test, model = 'c',
                                   lambdas = lambdas_comb,
                                   min.leaf.size = 5, n.thresh = 30)
save(res.div.c, file = 'small_simulations/data/div_res_c.RData')
load('small_simulations/data/div_res_c.RData')
cat('\n--- Model (c) ---\n')
print_diversity_header(features)
print_diversity_sim_res(res.div.c, lambdas_comb, features)

# Model (d) ----
set.seed(123)
res.div.d <- perform_diversity_sim(N, n, n.test, model = 'd',
                                   lambdas = lambdas_comb,
                                   min.leaf.size = 5, n.thresh = 30)
save(res.div.d, file = 'small_simulations/data/div_res_d.RData')
load('small_simulations/data/div_res_d.RData')
cat('\n--- Model (d) ---\n')
print_diversity_header(features)
print_diversity_sim_res(res.div.d, lambdas_comb, features)

# Model (e) ----
set.seed(123)
res.div.e <- perform_diversity_sim(N, n, n.test, model = 'e',
                                   lambdas = lambdas_comb,
                                   min.leaf.size = 5, n.thresh = 30)
save(res.div.e, file = 'small_simulations/data/div_res_e.RData')
load('small_simulations/data/div_res_e.RData')
cat('\n--- Model (e) ---\n')
print_diversity_header(features)
print_diversity_sim_res(res.div.e, lambdas_comb, features)
