library(caTools)
library(class)
library(e1071)
library(rpart)
library(randomForest)
library(caret)
library(ggplot2)
library(tidyr)


strokeData <- read.csv("healthcare-dataset-stroke-data.csv")


sum(is.na(strokeData$bmi))  # 201 NA values
strokeData <- na.omit(strokeData) # removed it

# Converting columns to factors
strokeData$gender <- as.factor(strokeData$gender)
strokeData$hypertension <- as.factor(strokeData$hypertension)
strokeData$heart_disease <- as.factor(strokeData$heart_disease)
strokeData$ever_married <- as.factor(strokeData$ever_married)
strokeData$work_type <- as.factor(strokeData$work_type)
strokeData$Residence_type <- as.factor(strokeData$Residence_type)
strokeData$smoking_status <- as.factor(strokeData$smoking_status)
strokeData$stroke <- as.factor(strokeData$stroke)

# Splitting data
set.seed(123)
split <- sample.split(strokeData$stroke, SplitRatio = 0.7)
trainData <- subset(strokeData, split == TRUE)
testData <- subset(strokeData, split == FALSE)

# Scaling numeric columns
trainSetNum <- scale(trainData[sapply(trainData, is.numeric)])
testSetNum <- scale(testData[sapply(testData, is.numeric)])

# target for KNN
trainTarget <- trainData$stroke

# KNN 
k <- 5
knnPred <- knn(train = trainSetNum, test = testSetNum, cl = trainTarget, k = k)
knnConfMat <- confusionMatrix(knnPred, testData$stroke)
knnAccuracy <- knnConfMat$overall['Accuracy']
knnPrecision <- knnConfMat$byClass['Pos Pred Value']
knnRecall <- knnConfMat$byClass['Sensitivity']
knnF1 <- 2 * (knnPrecision * knnRecall) / (knnPrecision + knnRecall)

# Naive Bayes 
nbModel <- naiveBayes(stroke ~ ., data = trainData)
nbPred <- predict(nbModel, testData)
nbConfMat <- confusionMatrix(nbPred, testData$stroke)
nbAccuracy <- nbConfMat$overall['Accuracy']
nbPrecision <- nbConfMat$byClass['Pos Pred Value']
nbRecall <- nbConfMat$byClass['Sensitivity']
nbF1 <- 2 * (nbPrecision * nbRecall) / (nbPrecision + nbRecall)

# Decision Tree 
dtModel <- rpart(stroke ~ ., data = trainData, method = "class")
dtPred <- predict(dtModel, testData, type = "class")
dtConfMat <- confusionMatrix(dtPred, testData$stroke)
dtAccuracy <- dtConfMat$overall['Accuracy']
dtPrecision <- dtConfMat$byClass['Pos Pred Value']
dtRecall <- dtConfMat$byClass['Sensitivity']
dtF1 <- 2 * (dtPrecision * dtRecall) / (dtPrecision + dtRecall)

# Random Forest
rfModel <- randomForest(stroke ~ ., data = trainData)
rfPred <- predict(rfModel, testData)
rfConfMat <- confusionMatrix(rfPred, testData$stroke)
rfAccuracy <- rfConfMat$overall['Accuracy']
rfPrecision <- rfConfMat$byClass['Pos Pred Value']
rfRecall <- rfConfMat$byClass['Sensitivity']
rfF1 <- 2 * (rfPrecision * rfRecall) / (rfPrecision + rfRecall)

# model comparison summary
modelComparison <- data.frame(
  Model = c("KNN", "Naive Bayes", "Decision Tree", "Random Forest"),
  Accuracy = c(knnAccuracy, nbAccuracy, dtAccuracy, rfAccuracy),
  Precision = c(knnPrecision, nbPrecision, dtPrecision, rfPrecision),
  Recall = c(knnRecall, nbRecall, dtRecall, rfRecall),
  F1_Score = c(knnF1, nbF1, dtF1, rfF1)
)
print(modelComparison)

#for plotting purpose
modelComparison_long <- modelComparison %>%
  pivot_longer(cols = c(Accuracy, Precision, Recall, F1_Score), 
               names_to = "Metric", 
               values_to = "Value")


# Plotting all metrics in a single grouped bar chart :) 
ggplot(modelComparison_long, aes(x = Model, y = Value, fill = Metric)) + 
  geom_bar(stat = "identity", position = "dodge") + 
  scale_fill_manual(values = c("#432E54", "#4B4376", "#AE445A", "#E8BCB9")) +
  labs(title = "Comparison of Models", 
       x = "Model", 
       y = "Score", 
       fill = "Metric") +
  theme_minimal() + 
  theme(axis.text.x = element_text(angle = 0, hjust = 0.5))
