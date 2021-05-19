library(mice)
library(missForest)
library(dplyr)

library(MASS)
library(Amelia)
library(missMDA)
library(softImpute)
library(tidyr)
library(ggplot2)
library(devtools)
library(gdata)
library(mltools)
library(parallel)

#Please modify the path o current code_path in your system
mainPath <- "/path/to/code"

#find the file that contain miss and full datasets
dataMainPath <- file.path(mainPath, "data_stored")
missMainPath <- file.path(dataMainPath, "data_miss")

missingMechanism <- list("MCAR", "MAR", "MNAR")
caseList <- c(1:10)# the case index list, new index could be added in by expanding the range.
missList <- as.numeric(list(10, 30, 50)) # missing ratio from ./Rcode/sample_miss.R
sampleTime <- 100 # num of training test defined in ./Rcode/sample_miss.R
multi_ImputationTime <- 50 # num of multiple imputation
maxiteration <- 30 # num of iterations
numTrees <- 200 # num of trees for Miss-Forest

dataMicePath <- file.path(dataMainPath, "data_mice_store")
dataMissForestPath <- file.path(dataMainPath, "data_missforest")
dir.create(dataMicePath)
dir.create(dataMissForestPath)
for(method_index in 1:length(missingMechanism)){
  method <- missingMechanism[method_index]
  methodPath <- file.path(missMainPath, method)
  methodMicePath <- file.path(dataMicePath, method)
  methodMissForestPath <- file.path(dataMissForestPath, method)
  dir.create(methodMicePath)
  dir.create(methodMissForestPath)
  #k is the case index for loop
  for(k in 1:length(caseList)){
    case_name <- paste("Case", toString(k), sep = "") 
    case_add <- file.path(methodPath, case_name)
    #initial the case path for store the imputed datasets from mice
    caseMicePath <- file.path(methodMicePath, case_name)
    caseMissForestPath <- file.path(methodMissForestPath, case_name)
    dir.create(caseMicePath)
    dir.create(caseMissForestPath)
    for(j in 1:length(missList)){
      
      #find the path of missing case in data file
      miss_name <- paste("miss", toString(missList[j]), sep = "")
      mis_add <- file.path(case_add, miss_name)
      missMicePath <- file.path(caseMicePath, miss_name)
      missMissForestPath <- file.path(caseMissForestPath, miss_name)
      
      dir.create(missMicePath)
      dir.create(missMissForestPath)
      
      mclapply(1:sampleTime, function(i) {
        file_name <- paste(toString(i-1), ".csv", sep = "")
        file_path <- file.path(mis_add, file_name)
        mis_df <- read.csv(file_path)
        
        if(k == 1){
          matrix <- c(rep("pmm", 93))
        }
        else if(k == 3){
          matrix <- c(rep("pmm", 57))
        }
        else if(k == 2){
          matrix <- c(rep("pmm", 8))
        }
        else if(k == 4){
          cat <- c(1:7)
          mis_df[cat] <- lapply(mis_df[cat], function(x) as.factor(x))
          matrix <- c(rep("logreg", 7))
        }
        else if(k == 5){
          cat <- c(1:180)
          mis_df[cat] <- lapply(mis_df[cat], function(x) as.factor(x))
          matrix <- c(rep("logreg", 180))
        }
        else if(k == 6){
          cat <- c(1, 2, 5, 7, 9, 10)
          mis_df[cat] <- lapply(mis_df[cat], function(x) as.factor(x))
          matrix <- c("polyreg", "logreg", "pmm", "pmm", "logreg", "pmm", "logreg", "pmm", "logreg", "polyreg")
        }
        else if(k == 7){
          cat <- c(4, 5, 6, 8, 11)
          mis_df[cat] <- lapply(mis_df[cat], function(x) as.factor(x))
          matrix <- c(rep("pmm", 3), "logreg", "logreg", "polyreg", "pmm", "logreg", "pmm", "pmm", "polyreg", "pmm")
        }
        #If new dataset was added in, after assigned the new data with a new integer name as case index.
        #you could add code below in "else if" block for querying;
        #else if(k == Index_new){
          #cat <- c({The index for the categorical features counted from initial 1})
          #mis_df[cat] <- lapply(mis_df[cat], function(x) as.factor(x))
          #matrix <- c(rep("pmm", 10), #here is list of regression methods for each categorical features#)
        #}
        #
        else if(k > 7){
          cat <- c(11, 12, 13, 14, 15)
          mis_df[cat] <- lapply(mis_df[cat], function(x) as.factor(x))
          matrix <- c(rep("pmm", 10), "polyreg", "polyreg", "logreg", "logreg", "logreg")
        }
        
        
        #project_imputed <- mice(mis_dat, m=1, maxit = 30, method = 'pmm', predictorMatrix=predM, seed = 500)
        project_imputed <- mice(mis_df, m = multi_ImputationTime, maxit = maxiteration, meth = matrix, print=FALSE)
        
        holderMissMicePath <- file.path(missMicePath, toString(i-1))
        dir.create(holderMissMicePath)
        for(index in 1:multi_ImputationTime){
          fileImputedName <-  paste(toString(index-1), ".csv", sep = "")
          fileImputedPath <- file.path(holderMissMicePath, fileImputedName)
          data_imputed <- complete(project_imputed, index)
          write.csv(data_imputed, fileImputedPath, row.names = FALSE)
        }
        
        dataImputedMissForest <- missForest(mis_df, maxiter=maxiteration, ntree = numTrees)$ximp
        fileImputedMissForest <- paste(toString(i-1), ".csv", sep = "")
        fileImputedMissForestPath <- file.path(missMissForestPath, fileImputedMissForest)
        write.csv(dataImputedMissForest, fileImputedMissForestPath, row.names = FALSE)
       
      }, mc.cores = sampleTime)# \# of cores for multi-processing of imputing process.
    }
  }
}