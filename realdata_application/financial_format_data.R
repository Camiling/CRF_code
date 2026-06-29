rm(list=ls())
source('realdata_application/realdata_evaluate.R')
library(data.table)


# FINANCIAL DATA SET 
# Downloaded from https://www.kaggle.com/datasets/cnic92/200-financial-indicators-of-us-stocks-20142018?resource=download&select=2018_Financial_Data.csv

data_finance_tr1 = read.csv("realdata_application/data/2016_Financial_Data.csv")
data_finance_tr2 = read.csv("realdata_application/data/2017_Financial_Data.csv")
data_finance_te = read.csv("realdata_application/data/2018_Financial_Data.csv")

finance_tr1 = data_finance_tr1[,-which(colnames(data_finance_tr1) %in% c('X', 'Class', "X2017.PRICE.VAR...."))]
finance_tr2 = data_finance_tr2[,-which(colnames(data_finance_tr2) %in% c('X', 'Class', "X2018.PRICE.VAR...."))]
finance_tr1$y = data_finance_tr1$Class
finance_tr2$y = data_finance_tr2$Class
finance_te= data_finance_te[,-which(colnames(data_finance_te) %in% c('X', 'Class', "X2019.PRICE.VAR...."))]
finance_te$y = data_finance_te$Class
finance_tr = rbind(finance_tr1, finance_tr2)
finance_tr$z = c(rep(1, nrow(finance_tr1)), rep(2, nrow(finance_tr2)))
finance_te$z = 3


# Remove NA entries
# First cols that are mainly NA
finance_tr = finance_tr[,-which(colnames(finance_tr) %in% c('cashConversionCycle', 'operatingCycle'))]
finance_tr = finance_tr[-which(apply(finance_tr, 1, FUN= function(s) any(is.na(s)))),]
finance_te = finance_te[,-which(colnames(finance_te) %in% c('cashConversionCycle', 'operatingCycle'))]
finance_te = finance_te[-which(apply(finance_te, 1, FUN= function(s) any(is.na(s)))),]

# one-hot-encoding sector
onehot_full = function(df) {
  # Convert numeric-like strings to numeric; keep non-numeric as character
  df = as.data.frame(lapply(df, \(x) type.convert(x, as.is = TRUE)))
  
  # Identify categorical (non-numeric) columns
  is_cat = !vapply(df, is.numeric, logical(1))
  
  # Make categoricals into factors *including NA as a level*
  df[is_cat] = lapply(df[is_cat], \(x) factor(x, exclude = NULL))
  
  # Full one-hot (no intercept)
  X = model.matrix(~ . - 1, data = df)
  
  as.data.frame(X, check.names = FALSE)
}

finance_tr = onehot_full(finance_tr)
finance_te = onehot_full(finance_te)

dim(finance_te) # 793 231
dim(finance_tr) # 1498  231

# Save results ---------------
save(finance_tr,finance_te,
     file='realdata_application/data/Financial_formatted.RData')

