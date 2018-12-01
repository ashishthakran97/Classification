#Importing the file
file <- read.csv('file:///E:/ANALYTIXLABS/DATA SCIENCE USING R/BA CLASSES/5. ECOMMERCE CASE STUDY - CLASSIFICATION/train.csv')

file1 <- file[file==-1] <- NA

#UDF for understanding the distribution of data
mystat <- function(x){
  nmiss <- sum(is.na(x))
  a <- x[!is.na(x)]
  mean <- mean(a)
  l <- length(a)
  sd <- sd(a)
  min <- min(a)
  p1 <- quantile(a,.01)
  p5 <- quantile(a,.05)
  p10 <- quantile(a,.10)
  p25 <- quantile(a,.25)
  p50 <- quantile(a,.50)
  p75 <- quantile(a,.75)         
  p90 <- quantile(a,.90)
  p95 <- quantile(a,.95)
  p99 <- quantile(a,.99)
  max <- max(a)
  uc <- mean+3*sd
  lc <- mean-3*sd
  outlier <- max>p95|min<p5
  return(c(nmiss=nmiss,mean=mean,lenth=l,sd=sd,min=min,p1=p1,p5=p5,p10=p10,p25=p25,p50=p50,p75=p75,p90=p90,p95=p95,p99=p99,
           max=max,uc=uc,lc=lc,outlier=outlier))
  
}
#understanding the data
diag_stats <- t(data.frame(apply(file[,c(2:8,13:33)],2,FUN = mystat)))

#capping outliers.
m1_fun <- function(x){
  quantiles <- quantile(x,c(.05,.95),na.rm=TRUE)
  x[x<quantiles[1]] <- quantiles[1]
  x[x>quantiles[2]] <- quantiles[2]
  x
}

file2 <- data.frame(apply(file[,c(2:8,13:33)],2,FUN = m1_fun))
apply(is.na(file2[,]),2,sum)
#prop.table(table(file2$page1_exits))

#MISSING VALUE IMPUTATION
m2_fun <- function(x){
  x <- replace(x, is.na(x), which.max(prop.table(table(x))))
  x
}

file3 <- data.frame(apply(file2[,],2,m2_fun))
file3 <- cbind(file3,unique_id=file$unique_id)
file3$target <- factor(file3$target)

#splitting data
set.seed(5000)
train_ind <- sample(1:nrow(file3),size = 0.70*nrow(file3))
train <- file3[train_ind,]
test <- file3[-train_ind,]
#nrow(train)
#nrow(test)

#EXPORTING THE DATA FILES.
write.csv(train,'train.csv')
write.csv(test,'test.csv')
#################################################################
############  CASE STUDY STARTS FROM HERE    ################

library(h2o)   
library(caret)
require(graphics)
require(randomForest)

### with h2o ####
h2o.init(
  nthreads = -1,
  max_mem_size = '1G'
)
#load the file from disk
train1 <- h2o.importFile(path=normalizePath('F:/classification cs/train.csv'))
test1 <- h2o.importFile(path=normalizePath('F:/classification cs/test.csv'))

#assignment within h2o
train1 <- h2o.assign(train1,'train1.hex')
test1 <- h2o.assign(test1,'test1.hex')
train1$target <- as.factor(train1$target)
test1$target <- as.factor(test1$target)

#### with RANDOM FOREST #######
  rf1 <- h2o.randomForest(
    training_frame = train1,
    validation_frame = test1,
    x=1:27,
    y=28,                           
    model_id = 'rf1',              
     ntrees = 200,
    stopping_rounds = 10,
    score_each_iteration = TRUE,
    seed = 50000
  )

#performance evaluation

summary(rf1)
rf1@model$validation_metrics
rf1@model$validation_metrics@metrics$AUC #AUC
rf1@model$validation_metrics@metrics$Gini #Gini

#final predictions
final_pred_rf1 <- h2o.predict(
  object=rf1,
  newdata=test1)
mean(final_pred_rf1$predict==test1$target)   #88%


rf2 <- h2o.randomForest(
  training_frame = train1,
  validation_frame = test1,
  x=1:27,
  y=28,
  model_id = 'rf2',
  ntrees = 200,
  max_depth = 6,
  stopping_rounds = 15,
  stopping_tolerance = 0.001,
  score_each_iteration = TRUE,
  seed = 50000
)
summary(rf2)
rf2@model$validation_metrics
rf2@model$validation_metrics@metrics$AUC #AUC
rf2@model$validation_metrics@metrics$Gini #Gini

#final predictions
final_pred_rf2 <- h2o.predict(
  object=rf2,
  newdata=test1)
mean(final_pred_rf2$predict==test1$target)  #89.65%



#GBM
gbm1 <- h2o.gbm(
  training_frame = train1,
  validation_frame = test1,
  x=1:27,
  y=28,
  model_id = 'gbm1',
  seed = 50000
)
#PERFORMANCE EVALUATION
summary(gbm1)
gbm1@model$validation_metrics
h2o.varimp(gbm1)


#FINAL PREDICTION
final_pred_gbm1 <- h2o.predict(
  object = gbm1,
  newdata = test1
)
mean(final_pred_gbm1$predict==test1$target) ##### 90.01%  ###

final_pred <- data.frame(as.matrix(final_pred_gbm1))

write.csv(final_pred,'final_prediction.csv')


###############################################################################
############################################################################

gbm2 <- h2o.gbm(
  training_frame = train1,
  validation_frame = test1,
  x=1:27,
  y=28,
  model_id = 'gbm1',
  ntrees = 200,
  learn_rate = 0.01,
  max_depth = 6,
  sample_rate = 0.7,
  col_sample_rate = 0.8,
  stopping_rounds = 15,
  stopping_tolerance = 0.001,
  score_each_iteration = TRUE,
  seed = 50000
)

summary(gbm2)

final_pred_gbm1 <- h2o.predict(
  object = gbm1,
  newdata = test1
)
mean(final_pred_gbm1$predict==test1$target) #89.54%

#prefer random forest over gbm without CROSS-VALIDATION.

h2o.shutdown(prompt = FALSE)
