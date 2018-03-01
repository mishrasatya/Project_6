#----------------------Digit_Recognition-------------------#
# 1. Business Understanding
# 2. Data Understanding
# 3. Data Preparation
# 4. Model Building 
# 5  Cross validation 
#----------------------------------------------------------#

# 1. Business Understanding:

# From the field of pattern recognition, there is a problem that
# pertains to handwritten digit recognition. The goal is to develop
# a model that can correctly identify the digit (between 0-9) written in an image.
#----------------------------------------------------------#

# 2. Data Understanding:
# The dataset that is used to understand this concept is MNIST dataset
# The dataset is already divided into training and test datasets
# Number of instances in training : 60000
# Number of instances in testing : 10000
# Number of Attributes in both training and test: 785 (784 continuous, 1 nominal class label)
#----------------------------------------------------------#

# 3. Data Preparation:

#----------------------------------------------------------#

# Loading the required libraries
library(caret)
library(kernlab)
library(dplyr)
library(readr)
library(ggplot2)
library(gridExtra)
#----------------------------------------------------------#

# Loading the training and test dataset's
main_train <- read.csv("mnist_train.csv",header = FALSE, sep = ",")
main_test <- read.csv("mnist_test.csv",header = FALSE, sep = ",")

#----------------------------------------------------------#

# Checking the dimension of both the datasets
dim(main_train)
dim(main_test)

# Exploring both the datasets
summary(main_train)
summary(main_test)

# Checking if there are any missing value in both the datasets
sapply(main_train, function(x) sum(is.na(x)))
sapply(main_test, function(x) sum(is.na(x)))
#There were no missing value found in both the datasets

#---------------------------------------------------------#

# Now we will add the appropriate headers
# First we will subset the main data into two datasets
# One will have the first column only which will be named as Digit_Labels
# The second one will have all the other 784 columns which will be renamed as Pixel_***

# Subsetting first column and renaming it (for train dataset)
keep_col_1 <- subset(main_train, select = V1)
colnames(keep_col_1)[1] <- 'Digit_Labels'

# Subsetting all other columns and renaming them (for train dataset)
remove_col_1 <- subset( main_train, select = -V1 )
names(remove_col_1) <- paste("Pixel_", 1:784, sep="")

# Merging both the datasets to get the final with header dataset (for train dataset)
main_train_new <- bind_cols(keep_col_1,remove_col_1)

# Subsetting first column and renaming it (for test dataset)
keep_col_1_test <- subset(main_test, select = V1)
colnames(keep_col_1_test)[1] <- 'Digit_Labels'

# Subsetting all other columns and renaming them (for test dataset)
remove_col_1_test <- subset( main_test, select = -V1 )
names(remove_col_1_test) <- paste("Pixel_", 1:784, sep="")

# Merging both the datasets to get the final with header dataset (for test dataset)
main_test_new <- bind_cols(keep_col_1_test,remove_col_1_test)

#----------------------------------------------------------#

# Reshuffing both the datasets to remove bais before splitting the datasets into smaller parts
random_main_train <- main_train_new[sample(1:nrow(main_train_new)), ]
random_main_test <- main_test_new[sample(1:nrow(main_test_new)), ]

#----------------------------------------------------------#

# Checking the structure of first column (which is the output variable)
# of both the datasets and changing it to factor
str(random_main_train$Digit_Labels)
str(random_main_test$Digit_Labels)

# Changing it to factor type
random_main_train$Digit_Labels <- as.factor(random_main_train$Digit_Labels)
random_main_test$Digit_Labels <- as.factor(random_main_test$Digit_Labels)

#----------------------------------------------------------#

# As computationally it will be very difficult to run SVM on a local machine for a dataset
# that has more than 60000 records with 784 columns, hence splitting the
# dataset into subsets of 10000 rows each for better results.

# First we will create a vector that stores the denominator then we will
# capture the total number of rows in the dataset in another vector, we will
# then use the split function to subset the datasets into chunks of 10000 rows
# and store it as a list of 6 elements

split_number <- 10000
split_row <- nrow(random_main_train)
split_stored <- split(random_main_train, rep(1:ceiling(split_row/split_number), each=split_number, length.out=split_row))

# Now we will create 6 dataframes out of the list split_stored that has 6 elements
# We will use lapply with assign to assign each element to a dataframe all at once

lapply(seq_along(split_stored), function(x) {
  assign(c("tr1","tr2","tr3","tr4","tr5","tr6")[x], split_stored[[x]], envir=.GlobalEnv)
}
)

#----------------------------------------------------------#

# 4. Model Building

# First we will use a linear approach to see what is the accuracy
# Creating the first linear model using tr1 and vanilla Kernel
tr1_model_vanilla <- ksvm(Digit_Labels~ ., data = tr1, scale = FALSE, kernel = "vanilladot")
tr1_eval_vanilla <- predict(tr1_model_vanilla, random_main_test)

# Confusion matrix for first linear kernel
confusionMatrix(tr1_eval_vanilla,random_main_test$Digit_Labels)

# Accuracy : 0.9129
#                      Class: 0 Class: 1 Class: 2 Class: 3 Class: 4 Class: 5 Class: 6 Class: 7 Class: 8 Class: 9
# Sensitivity            0.9735   0.9841   0.8992   0.9020   0.9399   0.8363   0.9489   0.9232   0.8491   0.8573
# Specificity            0.9928   0.9953   0.9906   0.9840   0.9897   0.9875   0.9944   0.9886   0.9893   0.9912

#----------------------------------------------------------#

# Creating the second linear model using tr2 and vanilla Kernel
tr2_model_vanilla <- ksvm(Digit_Labels~ ., data = tr2, scale = FALSE, kernel = "vanilladot")
tr2_eval_vanilla <- predict(tr2_model_vanilla, random_main_test)

# Confusion matrix for second linear kernel
confusionMatrix(tr2_eval_vanilla,random_main_test$Digit_Labels)

# Accuracy : 0.9115
#                      Class: 0 Class: 1 Class: 2 Class: 3 Class: 4 Class: 5 Class: 6 Class: 7 Class: 8 Class: 9
# Sensitivity            0.9724   0.9833   0.9070   0.8990   0.9297   0.8744   0.9436   0.9134   0.8419   0.8385
# Specificity            0.9941   0.9939   0.9864   0.9855   0.9868   0.9908   0.9947   0.9890   0.9906   0.9899

#----------------------------------------------------------#

# Creating the third linear model using tr3 and vanilla Kernel
tr3_model_vanilla <- ksvm(Digit_Labels~ ., data = tr3, scale = FALSE, kernel = "vanilladot")
tr3_eval_vanilla <- predict(tr3_model_vanilla, random_main_test)

# Confusion matrix for third linear kernel
confusionMatrix(tr3_eval_vanilla,random_main_test$Digit_Labels)

# Accuracy : 0.9172
#                      Class: 0 Class: 1 Class: 2 Class: 3 Class: 4 Class: 5 Class: 6 Class: 7 Class: 8 Class: 9
# Sensitivity            0.9827   0.9903   0.9176   0.8871   0.9297   0.8700   0.9353   0.9047   0.8501   0.8910
# Specificity            0.9929   0.9945   0.9897   0.9846   0.9879   0.9864   0.9946   0.9944   0.9930   0.9900

#----------------------------------------------------------#

# Creating the fourth linear model using tr4 and vanilla Kernel
tr4_model_vanilla <- ksvm(Digit_Labels~ ., data = tr4, scale = FALSE, kernel = "vanilladot")
tr4_eval_vanilla <- predict(tr4_model_vanilla, random_main_test)

# Confusion matrix for fourth linear kernel
confusionMatrix(tr4_eval_vanilla,random_main_test$Digit_Labels)

# Accuracy : 0.9085
#                      Class: 0 Class: 1 Class: 2 Class: 3 Class: 4 Class: 5 Class: 6 Class: 7 Class: 8 Class: 9
# Sensitivity            0.9776   0.9833   0.8963   0.8832   0.9257   0.8655   0.9478   0.9047   0.8265   0.8622
# Specificity            0.9937   0.9929   0.9866   0.9845   0.9874   0.9889   0.9947   0.9918   0.9887   0.9892

#----------------------------------------------------------#

# Creating the fifth linear model using tr5 and vanilla Kernel
tr5_model_vanilla <- ksvm(Digit_Labels~ ., data = tr5, scale = FALSE, kernel = "vanilladot")
tr5_eval_vanilla <- predict(tr5_model_vanilla, random_main_test)

# Confusion matrix for fifth linear kernel
confusionMatrix(tr5_eval_vanilla,random_main_test$Digit_Labels)

# Accuracy : 0.9135
#                      Class: 0 Class: 1 Class: 2 Class: 3 Class: 4 Class: 5 Class: 6 Class: 7 Class: 8 Class: 9
# Sensitivity            0.9704   0.9859   0.9118   0.8871   0.9460   0.8711   0.9415   0.9086   0.8275   0.8722
# Specificity            0.9932   0.9936   0.9867   0.9858   0.9885   0.9859   0.9966   0.9928   0.9909   0.9900

#----------------------------------------------------------#

# Creating the sixth linear model using tr6 and vanilla Kernel
tr6_model_vanilla <- ksvm(Digit_Labels~ ., data = tr6, scale = FALSE, kernel = "vanilladot")
tr6_eval_vanilla <- predict(tr6_model_vanilla, random_main_test)

# Confusion matrix for sixth linear kernel
confusionMatrix(tr6_eval_vanilla,random_main_test$Digit_Labels)

# Accuracy : 0.9138
#                      Class: 0 Class: 1 Class: 2 Class: 3 Class: 4 Class: 5 Class: 6 Class: 7 Class: 8 Class: 9
# Sensitivity            0.9776   0.9789   0.9196   0.9059   0.9542   0.8430   0.9447   0.9047   0.8388   0.8563
# Specificity            0.9933   0.9939   0.9883   0.9823   0.9876   0.9898   0.9944   0.9929   0.9907   0.9911

#----------------------------------------------------------#

# We see that the accuracy is hovering around 91% for almost all the 6 models
# Now lets take a non linear appraoch and see what are the results
# We will be using RBF kernel to see if the accuracy goes up.

#----------------------------------------------------------#

# Creating the first non linear model using tr1 and RBF Kernel
tr1_model_RBF <- ksvm(Digit_Labels~ ., data = tr1, scale = FALSE, kernel = "rbfdot")
tr1_eval_RBF <- predict(tr1_model_RBF, random_main_test)

# Confusion matrix for first non linear kernel
confusionMatrix(tr1_eval_RBF,random_main_test$Digit_Labels)

# Accuracy : 0.9593
#                        Class: 0 Class: 1 Class: 2 Class: 3 Class: 4 Class: 5 Class: 6 Class: 7 Class: 8 Class: 9
# Sensitivity            0.9888   0.9903   0.9583   0.9535   0.9572   0.9406   0.9781   0.9416   0.9405   0.9395
# Specificity            0.9960   0.9968   0.9939   0.9943   0.9960   0.9959   0.9964   0.9961   0.9952   0.9941

#----------------------------------------------------------#

# Creating the second model using tr2 and RBF Kernel
tr2_model_RBF <- ksvm(Digit_Labels~ ., data = tr2, scale = FALSE, kernel = "rbfdot")
tr2_eval_RBF <- predict(tr2_model_RBF, random_main_test)

# Confusion matrix for second non linear kernel
confusionMatrix(tr2_eval_RBF,random_main_test$Digit_Labels)

# Accuracy : 0.9617
#                      Class: 0 Class: 1 Class: 2 Class: 3 Class: 4 Class: 5 Class: 6 Class: 7 Class: 8 Class: 9
# Sensitivity            0.9908   0.9912   0.9496   0.9564   0.9664   0.9552   0.9781   0.9484   0.9405   0.9376
# Specificity            0.9966   0.9973   0.9954   0.9949   0.9948   0.9956   0.9967   0.9963   0.9952   0.9947

#----------------------------------------------------------#

# Creating the third model using tr3 and RBF Kernel
tr3_model_RBF <- ksvm(Digit_Labels~ ., data = tr3, scale = FALSE, kernel = "rbfdot")
tr3_eval_RBF <- predict(tr3_model_RBF, random_main_test)

# Confusion matrix for third non linear kernel
confusionMatrix(tr3_eval_RBF,random_main_test$Digit_Labels)

# Accuracy : 0.9577
#                      Class: 0 Class: 1 Class: 2 Class: 3 Class: 4 Class: 5 Class: 6 Class: 7 Class: 8 Class: 9
# Sensitivity            0.9867   0.9877   0.9477   0.9554   0.9674   0.9540   0.9697   0.9416   0.9415   0.9227
# Specificity            0.9962   0.9975   0.9948   0.9942   0.9937   0.9954   0.9952   0.9952   0.9955   0.9953

#----------------------------------------------------------#

# Creating the fourth model using tr4 and RBF Kernel
tr4_model_RBF <- ksvm(Digit_Labels~ ., data = tr4, scale = FALSE, kernel = "rbfdot")
tr4_eval_RBF <- predict(tr4_model_RBF, random_main_test)

# Confusion matrix for fourth non linear kernel
confusionMatrix(tr4_eval_RBF,random_main_test$Digit_Labels)

# Accuracy : 0.9595
#                      Class: 0 Class: 1 Class: 2 Class: 3 Class: 4 Class: 5 Class: 6 Class: 7 Class: 8 Class: 9
# Sensitivity            0.9857   0.9894   0.9554   0.9653   0.9664   0.9372   0.9687   0.9436   0.9456   0.9326
# Specificity            0.9961   0.9971   0.9955   0.9933   0.9946   0.9958   0.9959   0.9961   0.9952   0.9953

#----------------------------------------------------------#

# Creating the fifth model using tr5 and RBF Kernel
tr5_model_RBF <- ksvm(Digit_Labels~ ., data = tr5, scale = FALSE, kernel = "rbfdot")
tr5_eval_RBF <- predict(tr5_model_RBF, random_main_test)

# Confusion matrix for fifth non linear kernel
confusionMatrix(tr5_eval_RBF,random_main_test$Digit_Labels)

# Accuracy : 0.9591
#                      Class: 0 Class: 1 Class: 2 Class: 3 Class: 4 Class: 5 Class: 6 Class: 7 Class: 8 Class: 9
# Sensitivity            0.9867   0.9894   0.9593   0.9574   0.9582   0.9439   0.9708   0.9387   0.9476   0.9346
# Specificity            0.9959   0.9973   0.9944   0.9944   0.9950   0.9964   0.9964   0.9958   0.9950   0.9940

#----------------------------------------------------------#

# Creating the sixth model using tr6 and RBF Kernel
tr6_model_RBF <- ksvm(Digit_Labels~ ., data = tr6, scale = FALSE, kernel = "rbfdot")
tr6_eval_RBF <- predict(tr6_model_RBF, random_main_test)

# Confusion matrix for sixth non linear kernel
confusionMatrix(tr6_eval_RBF,random_main_test$Digit_Labels)

# Accuracy : 0.9586
#                      Class: 0 Class: 1 Class: 2 Class: 3 Class: 4 Class: 5 Class: 6 Class: 7 Class: 8 Class: 9
# Sensitivity            0.9888   0.9885   0.9467   0.9564   0.9715   0.9462   0.9708   0.9455   0.9394   0.9286
# Specificity            0.9967   0.9972   0.9954   0.9939   0.9931   0.9954   0.9961   0.9961   0.9947   0.9954

#----------------------------------------------------------#

# Accuracy : 0.9593 - Model 1  using RBF Kernel
# Accuracy : 0.9617 - Model 2  using RBF Kernel
# Accuracy : 0.9577 - Model 3  using RBF Kernel
# Accuracy : 0.9595 - Model 4  using RBF Kernel
# Accuracy : 0.9591 - Model 5  using RBF Kernel
# Accuracy : 0.9586 - Model 6  using RBF Kernel

# we see that after using RBF kernal the accuracy has increased to
# 95% from 91%, hence going forward we will be using RBF kernel

#----------------------------------------------------------#

# We are able to build models with 10000 records lets see if we can run models
# having records of 20000 each having 784 columns.

split_number_1 <- 20000
split_row_1 <- nrow(random_main_train)
split_stored_1 <- split(random_main_train, rep(1:ceiling(split_row_1/split_number_1), each=split_number_1, length.out=split_row_1))

# Now we will create 3 dataframes out of the list split_stored_1 that has 3 elements
# We will use lapply with assign to assign each element to a dataframe all at once

lapply(seq_along(split_stored_1), function(x) {
  assign(c("tr7","tr8","tr9")[x], split_stored_1[[x]], envir=.GlobalEnv)
}
)

#--------------------------------------------------------#

# Creating the seventh model using tr7 and RBF Kernel
tr7_model_RBF <- ksvm(Digit_Labels~ ., data = tr7, scale = FALSE, kernel = "rbfdot")
tr7_eval_RBF <- predict(tr7_model_RBF, random_main_test)

# Confusion matrix for seventh non linear kernel
confusionMatrix(tr7_eval_RBF,random_main_test$Digit_Labels)

# Accuracy : 0.9681
#                      Class: 0 Class: 1 Class: 2 Class: 3 Class: 4 Class: 5 Class: 6 Class: 7 Class: 8 Class: 9
# Sensitivity            0.9898   0.9903   0.9661   0.9653   0.9715   0.9596   0.9802   0.9572   0.9507   0.9475
# Specificity            0.9970   0.9973   0.9952   0.9957   0.9968   0.9971   0.9972   0.9962   0.9958   0.9962

#--------------------------------------------------------#

# Creating the eighth model using tr8 and RBF Kernel
tr8_model_RBF <- ksvm(Digit_Labels~ ., data = tr8, scale = FALSE, kernel = "rbfdot")
tr8_eval_RBF <- predict(tr8_model_RBF, random_main_test)

# Confusion matrix for eighth non linear kernel
confusionMatrix(tr8_eval_RBF,random_main_test$Digit_Labels)

# Accuracy : 0.9673
#                      Class: 0 Class: 1 Class: 2 Class: 3 Class: 4 Class: 5 Class: 6 Class: 7 Class: 8 Class: 9
# Sensitivity            0.9878   0.9912   0.9632   0.9713   0.9735   0.9585   0.9697   0.9484   0.9641   0.9425
# Specificity            0.9972   0.9977   0.9953   0.9950   0.9955   0.9968   0.9976   0.9963   0.9962   0.9960

#--------------------------------------------------------#

# Creating the ninth model using tr9 and RBF Kernel
tr9_model_RBF <- ksvm(Digit_Labels~ ., data = tr9, scale = FALSE, kernel = "rbfdot")
tr9_eval_RBF <- predict(tr9_model_RBF, random_main_test)

# Confusion matrix for ninth non linear kernel
confusionMatrix(tr9_eval_RBF,random_main_test$Digit_Labels)

# Accuracy : 0.9658
#                      Class: 0 Class: 1 Class: 2 Class: 3 Class: 4 Class: 5 Class: 6 Class: 7 Class: 8 Class: 9
# Sensitivity            0.9878   0.9912   0.9622   0.9713   0.9674   0.9552   0.9729   0.9523   0.9548   0.9395
# Specificity            0.9968   0.9979   0.9953   0.9949   0.9952   0.9971   0.9975   0.9960   0.9957   0.9957

#--------------------------------------------------------#

# Accuracy : 0.9681 - Model 7 using RBF kernel and 20000 records
# Accuracy : 0.9673 - Model 8 using RBF kernel and 20000 records
# Accuracy : 0.9658 - Model 9 using RBF kernel and 20000 records

#--------------------------------------------------------#

# For training datasets with 10000 records we see that most of the times the accuracy
# come around 95%, while for datasets having 20000 records we see that the accuracy has
# moved to 96% in all the three cases.

# Lets now divide the dataset into two parts and see how does the accuracy behave.

#--------------------------------------------------------#

# Creating 2 datasets each having 30000 records

split_number_2 <- 30000
split_row_2 <- nrow(random_main_train)
split_stored_2 <- split(random_main_train, rep(1:ceiling(split_row_2/split_number_2), each=split_number_2, length.out=split_row_2))

# Now we will create 2 dataframes out of the list split_stored_2 that has 2 elements
# We will use lapply with assign to assign each element to a dataframe all at once

lapply(seq_along(split_stored_2), function(x) {
  assign(c("tr10","tr11")[x], split_stored_2[[x]], envir=.GlobalEnv)
}
)

#--------------------------------------------------------#

# Creating the tenth model using tr10 and RBF Kernel
tr10_model_RBF <- ksvm(Digit_Labels~ ., data = tr10, scale = FALSE, kernel = "rbfdot")
tr10_eval_RBF <- predict(tr10_model_RBF, random_main_test)

# Confusion matrix for tenth non linear kernel
confusionMatrix(tr10_eval_RBF,random_main_test$Digit_Labels)

# Accuracy : 0.9721
#                      Class: 0 Class: 1 Class: 2 Class: 3 Class: 4 Class: 5 Class: 6 Class: 7 Class: 8 Class: 9
# Sensitivity            0.9908   0.9912   0.9719   0.9723   0.9705   0.9697   0.9812   0.9601   0.9630   0.9485
# Specificity            0.9978   0.9980   0.9960   0.9967   0.9967   0.9978   0.9978   0.9963   0.9958   0.9962

#--------------------------------------------------------#

# Creating the eleventh model using tr11 and RBF Kernel
tr11_model_RBF <- ksvm(Digit_Labels~ ., data = tr11, scale = FALSE, kernel = "rbfdot")
tr11_eval_RBF <- predict(tr11_model_RBF, random_main_test)

# Confusion matrix for eleventh non linear kernel
confusionMatrix(tr11_eval_RBF,random_main_test$Digit_Labels)

# Accuracy : 0.9713
#                      Class: 0 Class: 1 Class: 2 Class: 3 Class: 4 Class: 5 Class: 6 Class: 7 Class: 8 Class: 9
# Sensitivity            0.9898   0.9921   0.9709   0.9802   0.9745   0.9596   0.9781   0.9543   0.9661   0.9445
# Specificity            0.9972   0.9977   0.9963   0.9956   0.9966   0.9981   0.9978   0.9963   0.9962   0.9962

#--------------------------------------------------------#

# Accuracy : 0.9721 - Model 10 using RBF kernel and 30000 records
# Accuracy : 0.9713 - Model 11 using RBF kernel and 30000 records

#--------------------------------------------------------#

# We see that by using 30000 records for training the accuracy has increased to 97%
# 1% jump from the previous traning datasets of 20000 records.

#-----------------------------------------------------------#

# Now lets try using the entire dataset and see if we can get better results
tr12_model_RBF <- ksvm(Digit_Labels~ ., data = random_main_train, scale = FALSE, kernel = "rbfdot")
tr12_eval_RBF <- predict(tr12_model_RBF, random_main_test)

# Confusion matrix for twelth non linear kernel
confusionMatrix(tr12_eval_RBF,random_main_test$Digit_Labels)

# Accuracy : 0.9772
#                      Class: 0 Class: 1 Class: 2 Class: 3 Class: 4 Class: 5 Class: 6 Class: 7 Class: 8 Class: 9
# Sensitivity            0.9929   0.9921   0.9738   0.9832   0.9756   0.9731   0.9823   0.9660   0.9743   0.9574
# Specificity            0.9978   0.9982   0.9970   0.9971   0.9974   0.9985   0.9983   0.9968   0.9969   0.9967

#-----------------------------------------------------------#

# The accuracy has increased by 0.50% when the whole data set was used.

#-----------------------------------------------------------#

# 5  Cross validation
# Hyperparameter tuning and Cross Validation

# We will be using the evaluation metric accuracy, for which we will be
# creating a vector by the name metric and assign it "Accuracy"
metric <- "Accuracy"

# We will now create another vector called grid_tuning and call the expand.grid function
# Expand.grid functions takes set of hyperparameters, that we shall pass to our model.
set.seed(10)
grid_tuning <- expand.grid(.sigma=c(0.025, 0.05), .C=c(0.1,0.5,1,2) )

# We will now create another vector called train_control using the traincontrol function
# traincontrol function Controls the computational nuances of the train function.
# method used will be cross validation and the number implies, how many folds
train_control <- trainControl(method="cv", number=2)

# The train function has three components 1.Target ~ Prediction, 2.data & 3.method = Algorithm
# Along with that we will add
# metric = metric vector
# tuneGrid = grid_tuning vector
# trcontrol = train_control vector
# We are using the tr2 dataset
fitting_svm <- train(Digit_Labels~., data=tr2, method="svmRadial", metric=metric, 
                 tuneGrid = grid_tuning, trControl = train_control)

print(fitting_svm)
plot(fitting_svm)

# Summary of sample sizes: 4999, 5001 
# Resampling results across tuning parameters:
  
#  sigma  C    Accuracy  Kappa
# 0.025  0.1  0.1093    0    
# 0.025  0.5  0.1093    0    
# 0.025  1.0  0.1093    0    
# 0.025  2.0  0.1093    0    
# 0.050  0.1  0.1093    0    
# 0.050  0.5  0.1093    0    
# 0.050  1.0  0.1093    0    
# 0.050  2.0  0.1093    0    

# Accuracy was used to select the optimal model using  the largest value.
# The final values used for the model were sigma = 0.05 and C = 0.1.


#----------------------END OF CODE---------------------#




