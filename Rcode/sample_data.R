library(missForest)
library(MASS)
library(mice)

library(Amelia)
library(missForest)
library(missMDA)
library(softImpute)
library(dplyr)
library(tidyr)
library(ggplot2)
library(devtools)
library(gdata)
library(mltools)

source('/home/suny/code/Rcode/amputation.R')

#Please modify the path o current code_path in your system
mainPath <- "/home/suny/code"

#find the file that contain miss and full datasets
dataMainPath <- file.path(mainPath, "data_stored")
originalPath <- file.path(dataMainPath, "data")
set.seed(123)
numRows <- as.numeric(list(100, 500, 10000))
for(i in 1:3){
  filename <- paste( toString(i+9), ".csv", sep = "")
  ful_add <- file.path(originalPath, filename)
  n <- numRows[i]
  p <- 15
  mu.X <- c(0,0,0,1,1,1,2,2,3,3,0,0,0,0,0)
  Sigma.X <- diag(0.5, ncol = 15, nrow = 15) + matrix(0.5, nrow = 15, ncol = 15)
  
  X <- mvrnorm(n, mu.X, Sigma.X)

  X <- as.data.frame(X)
  head(X)
  # generate categorical data fro V11~V15
  X$V11 = cut(X$V11, 
              breaks = c(-Inf, -1, 0, 1, Inf), 
              labels =c("a","b","c","d"))
  table(X$V11)

  X$V12 = cut(X$V12, 
              breaks = c(-Inf, -1, 0, Inf), 
              labels =c("a","b","c"))
  table(X$V12)

  X$V13 = cut(X$V13, 
              breaks = c(-Inf, -0.5, Inf), 
              labels =c("a","b"))
  table(X$V13)

  X$V14 = cut(X$V14, 
              breaks = c(-Inf, 0, Inf), 
              labels =c("a","b"))
  table(X$V14)

  X$V15 = cut(X$V15, 
              breaks = c(-Inf, 0.5, Inf), 
              labels =c("a","b"))
  table(X$V15)
  write.csv(X, ful_add, row.names = FALSE)
}