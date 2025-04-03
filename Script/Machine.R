library(caret)
library(ggplot2)
library(class)
library(e1071)

###############  (A) #######################
## Reading in the data
file <- read.table('C:\\Users\\user\\OneDrive\\Documents\\MATH501\\Coursework\\earthquake.txt', header
                 =TRUE)
attach(file)
summary(file)

# Type Distribution: There are more instances of earthquakes (equake: 26) than nuclear explosions (explosn: 11) in the earthquake dataset.

# Body-Wave Magnitude (Mb): The body-wave magnitude ranges from 4.65 to 6.47.
# The median is slightly higher than the mean, which suggests a slight left skew in the distribution.
# The first quartile (Q1) is at 5.22, and the third quartile (Q3) is at 5.94, indicating that the middle 50% of the data is spread over a range of 0.72 units.

# Surface-Wave Magnitude (Ms): The surface-wave magnitude ranges from 3.71 to 6.34.
# The mean and median are closer together compared to the body-wave magnitude, suggesting a more symmetric distribution around the center.
# The interquartile range (Q3 - Q1) is 0.69, similar to the body-wave magnitude, indicating a similar spread in the middle 50% of the data.


type_b <- rep(0, 37)   # a vector of 10000 values all equal to zero
type_b[type == 'equake'] = 1   # where default is equal to 'Yes', change 0 into 1
type_b <- as.factor(type_b)    # convert the type of the variable to be a factor

# Create a dataframe
eq <- data.frame(file, type_b)

# Convert type to a factor
eq$type <- as.factor(eq$type) # Convert to factor if it represents groups

# Visualization for Relationship
par(mfrow = c(1,1))
# Specify colors to use
cols <- c('blue', 'red')

# scatter plot of earthquake data with body-wave magnitude on the x-axis and surface-wave magnitude on the y-axis.
plot(eq$body, eq$surface, col =cols[eq$type], xlab = 'Body-Wave Magnitude (Mb)', 
     ylab = 'Surface-wave Magnitude (Ms)', pch = 18)
legend(x = 'topright', legend = c('equake', 'explosn'), col = c('blue','red'), pch = 18)

#### Based on the graph
# Earthquakes (equake): Represented by blue diamonds, these points are mostly scattered between Mb values of 5.0 to 6.0 and Ms values of 4.5 to 5.5. This suggests that earthquakes in the dataset tend to have a narrower range of surface-wave magnitudes compared to body-wave magnitudes.

# Explosions (explosn): Represented by red diamonds, these points are clustered more tightly and are found at higher Mb values (approximately 6.0 to 6.5) and lower Ms values (approximately 4.0 to 4.5). This indicates that explosions tend to have higher body-wave magnitudes relative to their surface-wave magnitudes when compared to earthquakes.

#Decision Boundaries: The plot seems to show a clear distinction between the two types of events, with a potential decision boundary that could be drawn around Mb = 5.75. Events with a higher Mb and lower Ms are more likely to be explosions, while those with a lower Mb and higher Ms are more likely to be earthquakes.

# Patterns and Misclassification Potential: The separation between the two types of events suggests that a classifier should perform well. However, there might be a small overlap or proximity around the decision boundary where misclassification could occur, especially for events with Mb values close to 5.75.



################# (B) #################################

# Suitability of Models: Both KNN and SVM models are suitable for this classification task as they can handle the binary classification and capture the non-linear decision boundary suggested by the data.
# The visualizations (plot of body wave against surface wave) provides evidence to support the effectiveness of the classifiers in distinguishing between the two types of seismic events. 

#select only the two predictors so we can scale our data later
data <- cbind(eq$body, eq$surface)
# Set column names
colnames(data) <- c("body", "surface")

# Set a random seed for reproducibility of the sample.
set.seed(1)

# Randomly sample 27 indices from the first 37 indices to create a training set.
train <- sample(37, 27)

# Select the data points for the training set using the sampled indices.
train_data <- data[train, ]   # select the data points for the training set
dim(train_data)   

# Select the data points for the test set using the indices not included in the training set.
test_data <- data[-train,]
dim(test_data)

cl_train <- eq$type_b[train]   # Class labels for training data
cl_test <- eq$type_b[-train]   # Class labels for test data

# The two predictors have different spreads as can be seen from the plot
# and so we need to **standardise**  them for the KNN method.

# Scale the training data by centering and scaling.
train_scale <- scale(train_data, center = TRUE, scale = TRUE)

# and for the test set too:
test_scale <- scale(test_data, 
                  center = attr(train_scale, "scaled:center"), 
                  scale = attr(train_scale, "scaled:scale"))


# Calculate the training error for K=5 using the KNN algorithm.
knn.k <- knn(train = train_scale, test = train_scale, cl = cl_train, k = 5) 
tab <- table(knn.k, cl_train) # confusion matrix
tab 

(tab[1,2] + tab[2,1]) / sum(tab)    # the training error

# This matrix indicates that:
#   
# 15 earthquakes were correctly classified as earthquakes (true positives).
# 10 explosions were correctly classified as explosions (true negatives).
# 2 earthquakes were incorrectly classified as explosions (false negatives).
# The training error rate calculated from the confusion matrix is approximately ( 0.074 ), or 7.4%. This is a relatively low error rate, suggesting that the classifier is performing well on the training data.

# Selecting best K using cross-validation

# We will use the function knn.cv()

# Define a function to calculate cross-validation error for different values of k.
cv.error <- function(k){
  knn.best <- knn.cv(train = train_scale, cl = cl_train, k = k) 
  tab <- table(knn.best, cl_train) 
  cv.error <- (tab[1,2] + tab[2,1]) / sum(tab) 
  return(cv.error) 
}

# Initialize an array to store cross-validation errors for k values from 1 to 100.
cv.errors <- rep(0, 100)
# Calculate cross-validation errors for each k and store them in the array.
for(i in 1:100) cv.errors[i] <- cv.error(k=i)

# Plot the cross-validation errors for k values from 1 to 30 for a more detailed view.
plot(cv.errors, xlab="K", ylab = "CV error", pch = 16)
plot(cv.errors[1:30], xlab="K", ylab = "CV error", pch = 16)

#Get the value for the best K
k.best <- min( which.min(cv.errors) )
k.best  

# The results from your K-Nearest Neighbors (KNN) classifier indicate that the best number of neighbors(k), to use is 3. This value of ( k ) was determined to minimize the cross-validation error.
# 


# Test error for the KNN classifier:

knn.test <- knn(train = train_scale, test = test_scale, cl = cl_train, k = k.best) 
tab <- table(knn.test, cl_test) 
test.error.knn <- (tab[1,2] + tab[2,1]) / sum(tab)
test.error.knn 

# When the optimal ( k ) value was applied to the test data, the confusion matrix and the test error rate were calculated as follows:
#This matrix indicates that:
# 8 earthquakes were correctly classified as earthquakes (true positives).
# 1 explosion was correctly classified as an explosion (true negative).
# 1 earthquake was incorrectly classified as an explosion (false negative).
# No explosions were incorrectly classified as earthquakes (false positives).
# The test error rate, calculated as the sum of false negatives and false positives divided by the total number of cases gave a value of 0.1 or 10%. This confirms that the classifier is quite accurate, correctly classifying 90% of the test cases

# 4. Visualisation of the KNN classifier

# First, we view the data to see what range they have:
plot(train_scale[,1], train_scale[,2], xlab = "Standardised body", 
     ylab = "Standardised surface", pch = 16)


# We visualise the resulting classification rule by drawing the class boundary in the scatter plot of the data. 

# Also check the minimum values
min(train_scale[, 1], na.rm = TRUE) # check max value of balance
min(train_scale[, 2], na.rm = TRUE) # check max value of income
min(train_scale[, 1], na.rm = TRUE) # check max value of balance
min(train_scale[, 2], na.rm = TRUE) # check max value of income
max(train_scale[, 1], na.rm = TRUE)
max(train_scale[, 2], na.rm = TRUE)
# We will use a grid of size 50x50.
len <- 50
xp.s <- seq(-3, 2, length = len)  
yp.s <- seq(-2, 3, length = len)  
xygrid.s <- expand.grid(body = xp.s, surface = yp.s) # create a grid of points

# Classify points on the grid:
grid_knn <- knn(train = train_scale, test = xygrid.s, cl = cl_train, k = k.best)

# Prepare colours to be plotted:
col3 <- rep("lightblue", len*len)
for (i in 1:(len*len)) if (grid_knn[i]== '0') col3[i] <- "pink"

plot(xygrid.s, col = col3, main = "KNN classifier with best K", xlab = "Std Body",
     ylab = "Std Surface", xlim = c(-3,2), ylim = c(-2,3), pch = 16, asp = 1)
contour(xp.s, yp.s, matrix(grid_knn, len), levels = 0.5, add = TRUE, lwd = 2)
points(train_scale[ ,1], train_scale[ ,2], col = cl_train, pch = 20)
# 
# The plot shows a clear separation between the two classes, with the decision boundary appearing to be well-placed in relation to the training data points. The classifier seems to have a good understanding of the underlying pattern, as indicated by the distinct regions for each class.

#### SVM with linear kernel
#train_data <- as.data.frame(train_data)

set.seed(5)
tune.out = tune(svm, type_b ~ body + surface, 
                data = eq[train, ], kernel ="linear", 
                ranges = list(cost = c(0.001, 0.01, 0.1, 1, 5, 10, 20, 50, 100) ))
plot(tune.out)
summary(tune.out)

# The results from tuning the Support Vector Machine (SVM) with a linear kernel indicate that the best cost parameter is 1, with a 10-fold cross-validation error of approximately 0.067, or 6.7%. This is a low error rate, suggesting that the SVM model with this cost parameter is performing well in distinguishing between earthquakes and explosions.

# value of cost that gives the lowest cross-validation error
bestmod = tune.out$best.model
plot(bestmod, eq[train, ], surface ~ body)

# Evaluate the classifier using test data
ypred = predict(bestmod, eq[-train, ])
tab <- table(predict = ypred, truth = type[-train] ) 
tab

(tab[1,2] + tab[2,1]) / sum(tab)

# The confusion matrix for the SVM classifier on the test data is as follows:
# This matrix indicates that:
# All 9 earthquakes were correctly classified as earthquakes (true positives).
# The single explosion was correctly classified as an explosion (true negative).
# There were no misclassifications (false positives or false negatives).
# The test error rate calculated from the confusion matrix is 0, which means there were no errors on the test set. This excellent result suggests that the SVM model has learned to classify the events in the test set perfectly.

####  Polynomial kernel

set.seed(5)
tune.out = tune(svm, type ~ body + surface,
                data = eq[train, ],
                kernel ="polynomial",
                ranges = list(cost = seq(from = 0.01, to = 20, length = 40),
                              degree = seq(from = 1, to = 5, length = 5) )
                )
bestmod <- tune.out$best.model
summary(bestmod)

# The summary of the SVM model with a polynomial kernel indicates that the best parameters found through tuning are a cost of approximately 1.035 (~1) and a degree of 1. The model used 12 support vectors (6 for each class) to construct the decision boundary.

# Visualise the best classifier:
plot(bestmod, eq[train, ], surface~body)

# Test error:
ypred = predict(bestmod, eq[-train, ])
tab = table(predict = ypred, truth = type[-train])
(tab[1,2] + tab[2,1]) / sum(tab) # test error

# This matrix indicates that:
# 
# All 9 earthquakes were correctly classified as earthquakes (true positives).
# The single explosion was correctly classified as an explosion (true negative).
# There were no misclassifications (false positives or false negatives).
# The test error rate calculated from the confusion matrix is 0, which means there were no errors on the test set. This excellent result suggests that the SVM model with a polynomial kernel has learned to classify the events in the test set perfectly.

### Radial Kernel

#perform cross-validation using tune() to select the best 
# choice of gamma and cost for an SVM with a radial kernel:

set.seed(5)
tune.out = tune(svm, type ~ body + surface,
                data = eq[train, ],
                kernel ="radial",
                ranges = list(cost = seq(from = 0.01, to = 20, length = 40),
                              gamma = seq(from = 1, to = 50, length = 20) )
)
#value of cost that gives the lowest cross-validation error
bestmod <- tune.out$best.model
summary(bestmod)
# 
# The summary of your SVM model with a radial kernel indicates that the best parameters found through tuning are a cost of approximately 0.523 and a gamma value that was part of the tuning process. This model used 17 support vectors (11 for earthquakes and 6 for explosions) to construct the decision boundary.

# Visualise the best classifier:
plot(bestmod, eq[train, ], surface~body)

# Test error:
ypred = predict(bestmod, eq[-train, ])
tab = table(predict = ypred, truth = type[-train])
(tab[1,2] + tab[2,1]) / sum(tab) # test error

# The confusion matrix for the test data using this SVM model is the same as the previous models. The test error rate calculated from the confusion matrix is 0, which means there were no errors on the test set. This result suggests that the SVM model with a radial kernel has learned to classify the events in the test set perfectly.


# The results suggest that both KNN and SVM classifiers are highly effective in distinguishing between earthquakes and explosions using the given predictors. The low error rates, especially the zero test error rate for SVM models, demonstrate the models’ strong predictive capabilities. However, it’s important to note that the test set size and class balance should be considered when interpreting these results. The perfect classification by SVM models may not generalize to a larger or more diverse dataset. Therefore, further validation on a broader set of data is recommended to confirm these findings. The visualizations provided clear and intuitive representations of the classifiers’ decision rules, which are valuable for understanding and communicating the models’ behavior.


#### Leave one out cross validation

# An alternative way to asses the test error.

n <- nrow(eq)  # the number of data points in the data set eq
cv.predictions <- rep('equake', n)

for(i in 1:n) { # start a loop over all data points
  # Fit a classification using all data except one data point:
  svm_fit <- svm(type ~ ., data = eq[-i, ]) 
  
  # Make a prediction for the excluded data point:
  cv.predictions[i] <- predict(svm_fit, newdata = eq[i,], type = "class")
} 

# Now, we compare the predictions with the actual class labels:
tab <- table(cv.predictions, type)
tab

cv.error = (tab[1,2] + tab[2,1]) / sum(tab) 
cv.error
######

###### FOR KNN leave one out cross validation

# Scale the full dataset
full_scale <- scale(data, center = TRUE, scale = TRUE)

# Perform LOOCV using the full scaled dataset
loocv_errors <- rep(0, nrow(full_scale))
for(i in 1:nrow(full_scale)) {
  loocv_predictions <- knn(train = full_scale[-i, ], test = full_scale[i, , drop = FALSE], cl = 
                             eq$type[-i], k = k.best)
  loocv_errors[i] <- ifelse(loocv_predictions != eq$type[i], 1, 0)
}
loocv_error_rate <- mean(loocv_errors)

# Perform LOOCV for KNN with the best K
loocv_predictions <- knn.cv(train = train_scale, cl = cl_train, k = k.best)

# Create a confusion matrix to compare LOOCV predictions with actual class labels
loocv_confusion_matrix <- table(Predicted = loocv_predictions, Actual = cl_train)

# Calculate LOOCV error
loocv_error <- (loocv_confusion_matrix[1,2] + loocv_confusion_matrix[2,1]) / sum(loocv_confusion_matrix)

loocv_error


######################## (C) ##########################

# KNN Advantages:
# KNN is a non-parametric method, which means it does not make any underlying assumptions about the distribution of the data.
# It is simple to understand and easy to implement.
# The model tuning is relatively straightforward, with the number of neighbors (( k )) being the primary hyperparameter.

# KNN Disadvantages:
# KNN can be computationally expensive, especially as the size of the dataset grows, because it needs to compute the distance between each test instance and all training instances.
# It may not perform well with high-dimensional data due to the curse of dimensionality.
# The method is sensitive to the scale of the data, requiring standardization or normalization.

# # SVM Advantages:
# SVM is effective in high-dimensional spaces and is capable of defining complex higher-order separation planes through kernel functions.
# It is robust against overfitting, especially in high-dimensional space.
# Different kernel functions can be specified for the decision function, allowing flexibility.

# SVM Disadvantages:
# Choosing the right kernel and tuning the model can be more complex than KNN.
# SVM models are less intuitive to understand compared to KNN.
# The training time for SVM can be long, especially for larger datasets.

# Considering the LOOCV results and the characteristics of each model, the SVM with a linear kernel is best for this dataset. Its simplicity, computational efficiency, and superior LOOCV performance make it the preferred choice. The linear kernel’s ability to achieve perfect classification, as good as more complex kernels, aligns with the principle of Occam’s razor, favoring simpler models when possible.
# 


##################  (D) ##########################

new_eq <- cbind(eq$body, eq$surface)
## set column names
colnames(new_eq) <- c("body", "surface")
body <- new_eq[, "body"]
surface <- new_eq[, "surface"]

## Scaling the data
sd.data <- scale(new_eq)

km.earthquake <- kmeans(sd.data, centers = 4, nstart = 20)

v <- rep(0, 20)
for(K in 1:20){
  km.earthquake <- kmeans(sd.data, centers = K, nstart = 20)
  v[K] <- km.earthquake$tot.withinss
}
plot(v, xlab = "Number of clusters, K", ylab = "Within-clusters variability")

km.earthquake.20 <- kmeans(sd.data, centers = 6, nstart = 20)

# plot usig the best K
plot(body, surface, col = km.earthquake.20$cluster, 
     pch = 16, main = "Five clusters")

#The scatter plot provided does not definitively distinguish between earthquakes and explosions. While there is some separation among clusters, it’s not clear-cut. 