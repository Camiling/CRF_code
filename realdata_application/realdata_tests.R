library(dplyr)
library(reticulate)
source('CRF/CTree.R')
source('CRF/CRF.R')
source('realdata_application/realdata_evaluate.R')
# MUST RUN *_format_data.R for different datasets before this script can be run
load('realdata_application/data/ACS_2018_formatted.RData')
load('realdata_application/data/Diabetes_readmission_formatted.RData')
load('realdata_application/data/Financial_formatted.RData')
load('realdata_application/data/Accident_2020_formatted.RData')


run_income = T
run_mobility = T
run_pubcov = T
run_accidents = T
run_financial = T
run_diabetes = T

# Benchmark applications. Compare with existing methods

if(run_income){
  # ACS income ----
  features=colnames(train.income)[-which(colnames(train.income) %in% c('y', 'z'))]
  set.seed(111)
  train.income = train.income[train.income$z %in% c('ND',  'SD'),]
  train.income=clean_colnames(train.income)
  test.income=clean_colnames(test.income)
  train.income = train.income[sample(1:nrow(train.income), size=600),] 
  test.income = test.income[sample(1:nrow(test.income), size=300),]
  res_income = benchmark_evaluate(train.income, test.income, 'y', 'z', min.leaf.size=30, 
                                  lambdas = matrix(c(5, 5, 
                                                     10,10), nrow=2, byrow=T),
                                  gamma.anchor=c(1, 5, 10), lambda.irf = c(1, 5,10))
  print_method_results(res_income, test.income)
}

if(run_mobility){
  # ACS mobility ----
  train.domains = c('ND', 'SD')
  test.mobility = train.mobility[train.mobility$z %in% 'IA',]
  set.seed(111)
  train.mobility = train.mobility[train.mobility$z %in% train.domains,]
  train.mobility = train.mobility[c(sample(which(train.mobility$z==train.domains[1] & train.mobility$y==0), size=100), 
                                    sample(which(train.mobility$z==train.domains[1] & train.mobility$y==1), size=100),
                                    sample(which(train.mobility$z==train.domains[2] & train.mobility$y==0), size=100),
                                    sample(which(train.mobility$z==train.domains[2] & train.mobility$y==1), size=100)), ]
  test.mobility = test.mobility[c(sample(which(test.mobility$y==0), size=200),
                                  sample(which(test.mobility$y==1), size=200)),] 
  res_mobility = benchmark_evaluate(train.mobility, test.mobility, 'y', 'z', min.leaf.size=5, 
                                    lambdas = matrix(c(10,10), nrow=1, byrow=T),
                                    gamma.anchor=c(1, 5, 10), lambda.irf = c(1, 5,10))
  print_method_results(res_mobility,test.mobility)
}

# ACS pubcov ----

if(run_pubcov){
  train.domains = c('NC',  'SC')
  set.seed(111)
  train.pubcov = train.pubcov[train.pubcov$z %in% train.domains,]
  train.pubcov = train.pubcov[c(sample(which(train.pubcov$z==train.domains[1] & train.pubcov$y==0), size=100), 
                                    sample(which(train.pubcov$z==train.domains[1] & train.pubcov$y==1), size=100),
                                    sample(which(train.pubcov$z==train.domains[2] & train.pubcov$y==0), size=100),
                                    sample(which(train.pubcov$z==train.domains[2] & train.pubcov$y==1), size=100)), ]
  test.pubcov = test.pubcov[c(sample(which(test.pubcov$y==0), size=100),
                                  sample(which(test.pubcov$y==1), size=100)),] 
  res_pubcov = benchmark_evaluate(train.pubcov, test.pubcov, 'y', 'z', min.leaf.size=10, 
                                  lambdas = matrix(c(10,10), nrow=1, byrow=T),
                                  gamma.anchor=c(1, 5, 10), lambda.irf = c(1, 5,10))
  print_method_results(res_pubcov, test.pubcov)
}

if(run_accidents){
  # Accidents data ----
  train.domains = c('NC',  'SC')
  set.seed(111)
  accident_test = accident_train[accident_train$z %in% 'PA',]
  accident_train = accident_train[accident_train$z %in% train.domains,]
  accident_train = clean_colnames(accident_train)
  accident_test  = clean_colnames(accident_test)
  accident_train = accident_train[c(sample(which(accident_train$z==train.domains[1] & accident_train$y==0), size=200), 
                                    sample(which(accident_train$z==train.domains[1] & accident_train$y==1), size=200),
                                    sample(which(accident_train$z==train.domains[2] & accident_train$y==0), size=200),
                                    sample(which(accident_train$z==train.domains[2] & accident_train$y==1), size=200)), ]
  accident_test = accident_test[c(sample(which(accident_test$y==0), size=100),
                                  sample(which(accident_test$y==1), size=100)),] 
  res_accident = benchmark_evaluate(accident_train, accident_test, 'y', 'z', min.leaf.size=10, 
                                    lambdas = matrix(c(10,10), nrow=1, byrow=T), 
                                    gamma.anchor=c(1, 5, 10), lambda.irf = c(1, 5,10))
  print_method_results(res_accident, accident_test)
}

# Financial ----
if(run_financial){
  set.seed(1)
  finance_tr=clean_colnames(finance_tr)
  finance_te=clean_colnames(finance_te)
  res_finance_tr = benchmark_evaluate(finance_tr, finance_te, 'y', 'z', min.leaf.size=10, 
                                      lambdas = matrix(c(10,10), nrow=1, byrow=T), prune=F,
                                      gamma.anchor=c(1, 5, 10), lambda.irf = c(1, 5,10))
  print_method_results(res_finance_tr, finance_te)
}

# Technical ----
if(run_technical){
  set.seed(1)
  res_tech = benchmark_evaluate(train_tech, test_tech, 'y', 'z', min.leaf.size=10,
                                lambdas = matrix(c(10,10), nrow=1, byrow=T), 
                                gamma.anchor=c(1, 5, 10), lambda.irf = c(1, 5,10))
  print_method_results(res_tech, test_tech)
}

# Diabetes readmissions ----
if(run_diabetes){
  set.seed(1)
  train_diabetes_adm=clean_colnames(train_diabetes_adm)
  test_diabetes_adm=clean_colnames(test_diabetes_adm)
  res_diabetes_adm = benchmark_evaluate(train_diabetes_adm, test_diabetes_adm, 'y', 'z', min.leaf.size=5,
                                        lambdas = matrix(c(10,10), nrow=1, byrow=T),
                                        gamma.anchor=c(1, 5, 10), lambda.irf = c(1, 5,10))
  print_method_results(res_diabetes_adm, test_diabetes_adm)
}
