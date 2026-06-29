rm(list=ls())


# DIABETES READMISSION
# Downloaded from https://huggingface.co/datasets/imodels/diabetes-readmission, used the "test data" for reduced size.
# Train: Admission Source Referral and Transfer. Test: Emergency

data.adm = read.csv("realdata_application/data/diabetes_admission.csv") # Must be downloaded and placed in "data" folder
colnames.all = colnames(data.adm)
adm.names = colnames.all[substr(colnames.all, 1, 19) == "admission_source_id"]

# Consider referral and transfer as training domains
data.use.adm = data.adm[,-which(colnames.all %in% c(adm.names, "readmitted"))]
data.use.adm$z = 0
data.use.adm$z[data.adm$admission_source_id.Referral ==1] = 1
data.use.adm$z[data.adm$admission_source_id.Transfer==1] = 2
data.use.adm$z[data.adm$admission_source_id.Emergency==1] = 3
data.use.adm$y = data.adm$readmitted

train_diabetes_adm = data.use.adm[data.use.adm$z %in% c(1,2), ]
test_diabetes_adm= data.use.adm[data.use.adm$z == 3, ]
  
dim(train_diabetes_adm)
dim(test_diabetes_adm)

# Downsize (but keep class balance)
set.seed(1)
train_diabetes_adm = train_diabetes_adm[c(
                          sample(which(train_diabetes_adm$z==1 & train_diabetes_adm$y==0), size=300),
                          sample(which(train_diabetes_adm$z==1 & train_diabetes_adm$y==1), size=300),
                          sample(which(train_diabetes_adm$z==2 & train_diabetes_adm$y==0), size=300),
                          sample(which(train_diabetes_adm$z==2 & train_diabetes_adm$y==1), size=300)),]
  
test_diabetes_adm = test_diabetes_adm[sample(1:nrow(test_diabetes_adm), size=500),]

# Save results ---------------
save(train_diabetes_adm, test_diabetes_adm,
     file='realdata_application/data/Diabetes_readmission_formatted.RData')


  