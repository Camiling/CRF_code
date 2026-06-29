generate_data_tree = function(n, model='a', new=F, more.envirs=F){
  p=5
  X = matrix(NA, n, p)
  z.vals=rep(0,n)
  y=rep(0,n)
  # Generate data from model
  if(model == 'b'){
    # Model corresponding to Figure 1 (b) (we swapped order of a and b)
    for(i in 1:n){
      if(!new) {
        if(more.envirs) z=sample(c(1,2,4), 1) # Either from domain 1 or 2
        else z=sample(1:2, 1) # Either from domain 1 or 2
        if(z==1) {
          beta_yz = 1 
          beta_4z = 1
          beta_5z = 1
        }
        else if (z==2){
          beta_yz = -1 
          beta_4z = -1 
          beta_5z = -1        
        }
        else{
          beta_yz = 0.5
          beta_4z = 0.5
          beta_5z = -1
        }
        dat = generate_data_model_a(beta_yz = beta_yz, beta_4z = beta_4z, beta_5z=beta_5z, z=z)
      }
      else dat = generate_data_model_a(beta_yz = 1, beta_4z = 2, beta_5z = -10, z=3)
      X[i,] = dat$x
      y[i] = dat$y
      z.vals[i] = dat$z
    }
  } else if (model == 'a'){
    # Model corresponding to Figure 1 (a) (we swapped order of a and b)
    for(i in 1:n){
      if(!new) {
        if(more.envirs) z=sample(c(1,2,4), 1) # Either from domain 1 or 2
        else z=sample(1:2, 1) # Either from domain 1 or 2
        if(z==1) {
          beta_2z = 0.3 
          beta_3z = 0.3 
          beta_4z = -0.5 
        }
        else if (z==2){
          beta_2z = -0.3 
          beta_3z = -0.3 
          beta_4z = 0.5 
        }
        else {
          beta_2z = 0
          beta_3z = -1
          beta_4z = 0.5  
        }
        dat = generate_data_model_b(beta_2z =beta_2z, beta_3z = beta_3z, beta_4z = beta_4z, beta_4y = 1,z=z)
      }
      else dat = generate_data_model_b(beta_2z = 0.5, beta_3z = -0.5, beta_4z = -10,beta_4y = 1, z=3)
      X[i,] = dat$x
      y[i] = dat$y
      z.vals[i] = dat$z
    }
  }
  else if (model == 'c'){
    # Model corresponding to Figure 1 (a) with the edge between Z and Y removed. 
    for(i in 1:n){
      if(!new) {
        if(more.envirs) z=sample(c(1,2,4), 1) # Either from domain 1 or 2
        else z=sample(1:2, 1) # Either from domain 1 or 2
        if(z==1) {
          beta_4z = 1
          beta_5z = 1
        }
        else if (z==2){
          beta_4z = -1
          beta_5z = -1
        }
        else{ # z=4
          beta_4z = -0.5
          beta_5z = -0.5
        }
        dat = generate_data_model_c(beta_4z = beta_4z, beta_5z = beta_5z, z=z)
      }
      else dat = generate_data_model_c(beta_4z = 10, beta_5z = -10,z=3) # test envir
      X[i,] = dat$x
      y[i] = dat$y
      z.vals[i] = dat$z
    }
  }else if (model == 'd'){
    # Model corresponding to Figure 1 (b) with an arrow from z to y
    for(i in 1:n){
      if(!new) {
        if(more.envirs) z=sample(c(1,2,4), 1) # Either from domain 1 or 2
        else z=sample(1:2, 1) # Either from domain 1 or 2
        if(z==1) {
          beta_yz = 0.2
          beta_2z = 0.3 
          beta_3z = 0.3 
          beta_4z = -0.5 
        }
        else if (z==2){
          beta_yz = -0.2
          beta_2z = -0.3 
          beta_3z = -0.3 
          beta_4z = 0.5 
        }
        else {
          beta_yz = -1
          beta_2z = -0.2
          beta_3z = 0.5
          beta_4z = 0.5  
        }
        dat = generate_data_model_d(beta_yz = beta_yz, beta_2z =beta_2z, beta_3z = beta_3z, beta_4z = beta_4z, , beta_4y = 1,z=z)
      }
      else dat = generate_data_model_d(beta_yz = 0.5, beta_2z = 0.5, beta_3z = -0.5, beta_4z = -10, , beta_4y = 1,z=3)
      X[i,] = dat$x
      y[i] = dat$y
      z.vals[i] = dat$z
    }
  }else if (model == 'e'){
    # Model corresponding to Figure 1 (b) with larger domain shift
    for(i in 1:n){
      if(!new) {
        if(more.envirs) z=sample(c(1,2,4), 1) # Either from domain 1 or 2
        else z=sample(1:2, 1) # Either from domain 1 or 2
        if(z==1) {
          beta_2z = 0.3
          beta_3z = 0.3
          beta_4z = -0.5
        }
        else if (z==2){
          beta_2z = -0.3
          beta_3z = -0.3
          beta_4z = 0.5  
        }
        else { # z=4
          beta_2z = 0
          beta_3z = -1
          beta_4z = 0.5  
        }
        dat = generate_data_model_b(beta_2z =beta_2z, beta_3z = beta_3z, beta_4z = beta_4z, beta_4y = 1, z=z)
      }
      else dat = generate_data_model_b(beta_2z = 1, beta_3z = -1, beta_4z = -20, beta_4y = 1, z=3)
      X[i,] = dat$x
      y[i] = dat$y
      z.vals[i] = dat$z
    }
  }
  res=list(X=X,y=y,z=z.vals)
  return(res)
}

generate_data_model_a= function(sigma=0.1, beta_31=5, beta_y3=100, beta_y2=100, beta_yz=0, 
                                beta_4z=1, beta_4y=1, beta_5z = 0, beta_53=5, z=1){
  x1 = rnorm(1,0,sigma)
  x2 = rnorm(1,0,sigma)
  x3 = rnorm(1, beta_31*x1, sigma)
  y = rbinom(1, 1, logistic_prob(beta_y3*x3 + beta_y2*x2 + beta_yz))
  x4 = rnorm(1, beta_4z + beta_4y*y, sigma)
  x5 = rnorm(1, beta_5z + beta_53*x3, sigma)
  x = c(x1, x2, x3, x4, x5)
  res = list(x=x, y=y, z=z)
  return(res)
}

generate_data_model_b= function(sigma=0.1, beta_31=5, beta_3z= 0,
                                beta_2z = 0, beta_y3=100, beta_y2=100, 
                                beta_4z=0, beta_4y=2, beta_53=5, z=1){
  x1 = rnorm(1,0,sigma)
  x3 = rnorm(1, beta_31*x1+beta_3z, sigma)
  x2 = rnorm(1,beta_2z, sigma)
  y = rbinom(1, 1, logistic_prob(beta_y3*x3 + beta_y2*x2))
  x4 = rnorm(1, beta_4z + beta_4y*y, sigma)
  x5 = rnorm(1, beta_53*x3, sigma)
  x = c(x1, x2, x3, x4, x5)
  res = list(x=x, y=y, z=z)
  return(res)
}

generate_data_model_c= function(sigma=0.1, beta_31=5, beta_y3=100, beta_y2=100,
                                beta_4z=0.5, beta_4y=1, beta_5z = 0.5, beta_53=5,z=1){
  x1 = rnorm(1,0,sigma)
  x2 = rnorm(1,0,sigma)
  x3 = rnorm(1, beta_31*x1, sigma)
  y = rbinom(1, 1, logistic_prob(beta_y3*x3 + beta_y2*x2))
  x4 = rnorm(1, beta_4z + beta_4y*y, sigma)
  x5 = rnorm(1, beta_5z + beta_53*x3, sigma)
  x = c(x1, x2, x3, x4, x5)
  res = list(x=x, y=y, z=z)
  return(res)
}

generate_data_model_d= function(sigma=0.1, beta_31=5, beta_3z= 0, beta_yz=0,
                                beta_2z = 0, beta_y3=100, beta_y2=100, 
                                beta_4z=0, beta_4y=2, beta_53=5, z=1){
  x1 = rnorm(1,0,sigma)
  x3 = rnorm(1, beta_31*x1+beta_3z, sigma)
  x2 = rnorm(1,beta_2z, sigma)
  y = rbinom(1, 1, logistic_prob(beta_y3*x3 + beta_y2*x2+beta_yz))
  x4 = rnorm(1, beta_4z + beta_4y*y, sigma)
  x5 = rnorm(1, beta_53*x3, sigma)
  x = c(x1, x2, x3, x4, x5)
  res = list(x=x, y=y, z=z)
  return(res)
}

logistic_prob = function(expr){
  return(1/(1+exp(-expr)))
}

