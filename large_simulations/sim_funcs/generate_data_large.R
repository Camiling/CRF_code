

# -------------------------------
# Generate larger SCMs with in total sum(d) params
# -------------------------------
generate_data_tree_large = function(n,
                                    model = 'a',
                                    new = FALSE,
                                    more.envirs = FALSE,
                                    d = c(5,5,5,5,5),
                                    sigma = 0.1,
                                    domains_train = c(1,2),
                                    domain_test = 3,
                                    seed = 1){
  
  # "large" object holds dimension + mechanisms (maps/weights)
  large = make_large_params(d = d, sigma = sigma, seed = seed)
  
  # training domains like before
  if(more.envirs) domains_train = unique(c(domains_train, 4))
  
  z.vals = rep(0, n)
  y = rep(0, n)
  
  p_total = sum(d)
  X = matrix(NA_real_, n, p_total)
  
  colnames(X) = c(
    paste0("x1_", seq_len(d[1])),
    paste0("x2_", seq_len(d[2])),
    paste0("x3_", seq_len(d[3])),
    paste0("x4_", seq_len(d[4])),
    paste0("x5_", seq_len(d[5]))
  )
  
  if(model == 'b'){ # swapped order of a and b in paper.
    for(i in 1:n){
      if(!new) {
        z = sample(domains_train, 1)
        if(z==1) {
          beta_yz =  1
          beta_4z =  1
          beta_5z =  1
        } else if(z==2){
          beta_yz = -1
          beta_4z = -1
          beta_5z = -1
        } else { # z=4 (if used)
          beta_yz = 0.5
          beta_4z = 0.5
          beta_5z = -1
        }
        dat = generate_data_model_a_large(beta_yz = beta_yz, beta_4z = beta_4z, beta_5z = beta_5z, z = z, large = large)
      } else {
        z = domain_test
        dat = generate_data_model_a_large(beta_yz = 1, beta_4z = 2, beta_5z = -10, z = z, large = large)
      }
      X[i,] = dat$x
      y[i] = dat$y
      z.vals[i] = dat$z
    }
    
  } else if(model == 'a'){ # swapped order of a and b in paper.
    for(i in 1:n){
      if(!new) {
        z = sample(domains_train, 1)
        if(z==1) {
          beta_2z =  0.3
          beta_3z =  0.3
          beta_4z = -0.5
        } else if(z==2){
          beta_2z = -0.3
          beta_3z = -0.3
          beta_4z =  0.5
        } else { # z=4
          beta_2z =  0
          beta_3z = -1
          beta_4z =  0.5
        }
        dat = generate_data_model_b_large(beta_2z = beta_2z, beta_3z = beta_3z, beta_4z = beta_4z, beta_4y = 1, z = z, large = large)
      } else {
        z = domain_test
        dat = generate_data_model_b_large(beta_2z = 0.5, beta_3z = -0.5, beta_4z = -10, beta_4y = 1, z = z, large = large)
      }
      X[i,] = dat$x
      y[i] = dat$y
      z.vals[i] = dat$z
    }
    
  } else if(model == 'c'){
    for(i in 1:n){
      if(!new) {
        z = sample(domains_train, 1)
        if(z==1) {
          beta_4z =  1
          beta_5z =  1
        } else if(z==2){
          beta_4z = -1
          beta_5z = -1
        } else { # z=4
          beta_4z = -0.5
          beta_5z = -0.5
        }
        dat = generate_data_model_c_large(beta_4z = beta_4z, beta_5z = beta_5z, z = z, large = large)
      } else {
        z = domain_test
        dat = generate_data_model_c_large(beta_4z = 10, beta_5z = -10, z = z, large = large)
      }
      X[i,] = dat$x
      y[i] = dat$y
      z.vals[i] = dat$z
    }
    
  } else if(model == 'd'){
    for(i in 1:n){
      if(!new) {
        z = sample(domains_train, 1)
        if(z==1) {
          beta_yz =  0.2
          beta_2z =  0.3
          beta_3z =  0.3
          beta_4z = -0.5
        } else if(z==2){
          beta_yz = -0.2
          beta_2z = -0.3
          beta_3z = -0.3
          beta_4z =  0.5
        } else { # z=4
          beta_yz = -1
          beta_2z = -0.2
          beta_3z =  0.5
          beta_4z =  0.5
        }
        dat = generate_data_model_d_large(beta_yz = beta_yz, beta_2z = beta_2z, beta_3z = beta_3z, beta_4z = beta_4z, beta_4y = 1, z = z, large = large)
      } else {
        z = domain_test
        dat = generate_data_model_d_large(beta_yz = 0.5, beta_2z = 0.5, beta_3z = -0.5, beta_4z = -10, beta_4y = 1, z = z, large = large)
      }
      X[i,] = dat$x
      y[i] = dat$y
      z.vals[i] = dat$z
    }
    
  } else if(model == 'e'){
    for(i in 1:n){
      if(!new) {
        z = sample(domains_train, 1)
        if(z==1) {
          beta_2z =  0.3
          beta_3z =  0.3
          beta_4z = -0.5
        } else if(z==2){
          beta_2z = -0.3
          beta_3z = -0.3
          beta_4z =  0.5
        } else { # z=4
          beta_2z =  0
          beta_3z = -1
          beta_4z =  0.5
        }
        dat = generate_data_model_b_large(beta_2z = beta_2z, beta_3z = beta_3z, beta_4z = beta_4z, beta_4y = 1, z = z, large = large)
      } else {
        z = domain_test
        dat = generate_data_model_b_large(beta_2z = 0.7, beta_3z = -0.7, beta_4z = -15, beta_4y = 1, z = z, large = large)
      }
      X[i,] = dat$x
      y[i] = dat$y
      z.vals[i] = dat$z
    }
    
  } else stop("Unknown model: ", model)
  
  res = list(X = X, y = y, z = z.vals, d = d)
  return(res)
}

# -------------------------------
# MODEL-SPECIFIC GENERATORS
# -------------------------------

generate_data_model_a_large = function(sigma = NULL,
                                       beta_yz = 0,
                                       beta_4z = 1,
                                       beta_4y = 1,
                                       beta_5z = 0,
                                       z = 1,
                                       large){
  
  d = large$d
  d1=d[1]; d2=d[2]; d3=d[3]; d4=d[4]; d5=d[5]
  if(is.null(sigma)) sigma = large$sigma
  
  x1 = rnorm(d1, 0, sigma)
  x2 = rnorm(d2, 0, sigma)
  
  x3 = as.vector(large$A31 %*% x1 + rnorm(d3, 0, sigma))
  
  lin = sum(large$w3 * x3) + sum(large$w2 * x2) + beta_yz
  y = rbinom(1, 1, logistic_prob(lin))
  
  x4 = const_vec(beta_4z, d4) + beta_4y * y + rnorm(d4, 0, sigma)
  
  x5 = const_vec(beta_5z, d5) + as.vector(large$A53 %*% x3) + rnorm(d5, 0, sigma)
  
  x = c(x1, x2, x3, x4, x5)
  res = list(x = x, y = y, z = z)
  return(res)
}

generate_data_model_b_large = function(sigma = NULL,
                                       beta_3z = 0,
                                       beta_2z = 0,
                                       beta_4z = 0,
                                       beta_4y = 2,
                                       z = 1,
                                       large){
  
  d = large$d
  d1=d[1]; d2=d[2]; d3=d[3]; d4=d[4]; d5=d[5]
  if(is.null(sigma)) sigma = large$sigma
  
  x1 = rnorm(d1, 0, sigma)
  
  x3 = as.vector(large$A31 %*% x1 + const_vec(beta_3z, d3) + rnorm(d3, 0, sigma))
  x2 = const_vec(beta_2z, d2) + rnorm(d2, 0, sigma)
  
  lin = sum(large$w3 * x3) + sum(large$w2 * x2)
  y = rbinom(1, 1, logistic_prob(lin))
  
  x4 = const_vec(beta_4z, d4) + beta_4y * y + rnorm(d4, 0, sigma)
  
  x5 = as.vector(large$A53 %*% x3) + rnorm(d5, 0, sigma)
  
  x = c(x1, x2, x3, x4, x5)
  res = list(x = x, y = y, z = z)
  return(res)
}

generate_data_model_c_large = function(sigma = NULL,
                                       beta_4z = 0.5,
                                       beta_4y = 1,
                                       beta_5z = 0.5,
                                       z = 1,
                                       large){
  # same as a but with no z->y (beta_yz = 0)
  generate_data_model_a_large(sigma = sigma, beta_yz = 0, beta_4z = beta_4z, beta_4y = beta_4y, beta_5z = beta_5z, z = z, large = large)
}

generate_data_model_d_large = function(sigma = NULL,
                                       beta_yz = 0,
                                       beta_3z = 0,
                                       beta_2z = 0,
                                       beta_4z = 0,
                                       beta_4y = 2,
                                       z = 1,
                                       large){
  # same as b but add beta_yz into logit
  d = large$d
  d1=d[1]; d2=d[2]; d3=d[3]; d4=d[4]; d5=d[5]
  if(is.null(sigma)) sigma = large$sigma
  
  x1 = rnorm(d1, 0, sigma)
  
  x3 = as.vector(large$A31 %*% x1 + const_vec(beta_3z, d3) + rnorm(d3, 0, sigma))
  x2 = const_vec(beta_2z, d2) + rnorm(d2, 0, sigma)
  
  lin = sum(large$w3 * x3) + sum(large$w2 * x2) + beta_yz
  y = rbinom(1, 1, logistic_prob(lin))
  
  x4 = const_vec(beta_4z, d4) + beta_4y * y + rnorm(d4, 0, sigma)
  
  x5 = as.vector(large$A53 %*% x3) + rnorm(d5, 0, sigma)
  
  x = c(x1, x2, x3, x4, x5)
  res = list(x = x, y = y, z = z)
  return(res)
}

logistic_prob = function(expr){
  return(1/(1+exp(-expr)))
}

# Helper: constant vector intercept
const_vec = function(a, d) rep(a, d)

# Helper: build "large" (high-dim) parameters once
# Scales maps/weights by sqrt(dim) so logits don't blow up as dimension increases
make_large_params = function(d = c(5,5,5,5,5),
                             sigma = 0.1,
                             beta_31 = 5,
                             beta_53 = 5,
                             beta_y3 = 100,
                             beta_y2 = 100,
                             seed = 1){
  set.seed(seed)
  
  d1 = d[1]; d2 = d[2]; d3 = d[3]; d4 = d[4]; d5 = d[5]
  
  A31 = (beta_31 / sqrt(d1)) * matrix(1, nrow = d3, ncol = d1)
  A53 = (beta_53 / sqrt(d3)) * matrix(1, nrow = d5, ncol = d3)
  
  w3 = const_vec(beta_y3 / sqrt(d3), d3)
  w2 = const_vec(beta_y2 / sqrt(d2), d2)
  
  list(d = d, sigma = sigma, A31 = A31, A53 = A53, w3 = w3, w2 = w2)
}
