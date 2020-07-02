#'@title CALFGaussianAdjustment
#'@description Performs a Gaussian adjustment on a result set from CALF.
#'@param calfObject The results returned from running a CALF() function call on data.
#'@param The data set on which CALF operated formatted as a data.table.
#'@export
CALFGaussianAdjustment<- function(calfObject, data) {
  
  
  #Use the data as the basis for a sparse vector, then zero it out, and populate
  # the respective markers with the weights returned from CALF
  sparse = data[1,-1]
  sparse[1,] = rep(0, length(sparse))
  
  #Fill the sparse vector with the results
  markerCount = 1
  markerList = as.character(calfObject$selection$Marker)
  repeat {
    
    sparse[markerList[markerCount]] = calfObject$selection$Weight[markerCount]
    
    markerCount <- markerCount + 1
    
    if (markerCount > length(markerList))
      break
  }
  
  
  
  
  #Convert table to list of rows so apply functions can be used
  rows <- split(data[,-1], seq(nrow(data[,-1])))
  
  #Convert first column into a dataframe
  dfTarget = as.data.frame(data[,1])
  colnames(dfTarget) <- names(data[][1])
  
  #Calculate the sum product
  sumProd = sapply(rows, function(row) sum(row * t(sparse)))
  dfSumProd = as.data.frame(sumProd)
  
  
  #Declare the formula based upon the marker and target
  # and run the linear regression
  formula <- as.formula(paste(colnames(dfTarget[1])," ~ sumProd"))
  dfMerged = cbind.data.frame(dfTarget, dfSumProd)
  regression <- lm(formula,dfMerged)
  
  
  #Get the adjustment vector
  adj = sapply(dfMerged[,2], function(row) regression$coefficients[2]*row+regression$coefficients[1])
  dfAdj = as.data.frame(adj)
  
  
  adjDiff = dfTarget - dfAdj
  dfDiffAdjSqrd = as.data.frame(adjDiff^2)
  colnames(dfDiffAdjSqrd) <- c("diffSqrd")
  dfMerged = cbind.data.frame(dfMerged, dfAdj, dfDiffAdjSqrd)
  
  mse = sum(adjDiff^2)/dim(dfTarget)[1]
  
  vars <- list(
    calculations = dfMerged,
    mseAdjusted = mse,
    m = as.vector(regression$coefficients[2]),
    b = as.vector(regression$coefficients[1])
  )
  
  
  class(vars) <- append(class(vars),"CALFGaussianAdjustment")
  return(vars)
  
}






#'@title LASSOGaussianAdjustment
#'@description Performs a Gaussian adjustment on a result set from glmnet's LASSO predict. 
#'@param lassoObject The results returned from a glmnet LASSO predict() function call on data.
#'@param The data set on which LASSO operated formatted as a data.table.
#'@export
LASSOGaussianAdjustment<- function(lassoObject, data) {
  
  
  mat = as.matrix(lassoObject)
  tranMat = t(mat)
  b = tranMat[1]
  sparse = as.list(tranMat[,-1])
  
  
  #Convert table to list of rows so apply functions can be used
  rows <- split(data[,-1], seq(nrow(data[,-1])))
  
  #Convert first column into a dataframe
  dfTarget = as.data.frame(data[,1])
  colnames(dfTarget) <- names(data[][1])
  
  #Calculate the sum product
  sumProd = sapply(rows, function(row) sum(row * t(sparse)))
  sumProdPlusIntercept = sumProd + b
  dfSumProd = as.data.frame(sumProdPlusIntercept)
  
  
  #Declare the formula based upon the marker and target
  # and run the linear regression
  formula <- as.formula(paste(colnames(dfTarget[1])," ~ sumProdPlusIntercept"))
  dfMerged = cbind.data.frame(dfTarget, dfSumProd)
  regression <- lm(formula,dfMerged)
  
  
  #Get the adjustment vector
  adj = sapply(dfMerged[,2], function(row) regression$coefficients[2]*row+regression$coefficients[1])
  dfAdj = as.data.frame(adj)
  
  
  adjDiff = dfTarget - dfAdj
  dfDiffAdjSqrd = as.data.frame(adjDiff^2)
  colnames(dfDiffAdjSqrd) <- c("diffSqrd")
  dfMerged = cbind.data.frame(dfMerged, dfAdj, dfDiffAdjSqrd)
  
  mse = sum(adjDiff^2)/dim(dfTarget)[1]
  
  vars <- list(
    calculations = dfMerged,
    mseAdjusted = mse,
    m = as.vector(regression$coefficients[2]),
    b = as.vector(regression$coefficients[1])
  )
  
  
  class(vars) <- append(class(vars),"LASSOGaussianAdjustment")
  return(vars)
  
}