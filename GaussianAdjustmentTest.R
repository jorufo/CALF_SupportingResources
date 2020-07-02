library(CALF)
library(glmnet)


#Load the script that contains the class definitions for the adjustments
source("GaussianAdjustment.R")


#Load datasets: TODO: Update with your data set.
data<-read.table("data.csv",header=T,sep=",")




#Perform the LASSOGaussianAdjustment.  Pass in the results from the LASSO run and the
# data on which LASSO was applied.
lassoInput <- as.matrix(data[,1:ncol(data)])
y <- as.double(as.matrix(lassoInput[,1]))
x <- as.matrix(lassoInput[,-1])

model <- cv.glmnet(x,y,alpha=1, family="gaussian", type.measure="mse", grouped=FALSE)
result = predict(model,type="coefficients",s=0.065)

adjResults = LASSOGaussianAdjustment(result, data)
print(adjResults)





#Perform the CALFGaussianAdjustment.  Pass in the results from the CALF run and the
# data on which CALF was applied.
result <- calf(data, 5, "binary", "pval")
adjResults = CALFGaussianAdjustment(result, data)
print(adjResults)