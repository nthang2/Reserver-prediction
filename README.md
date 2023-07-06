![Template guide](https://github.com/udacity/machine-learning/blob/master/projects/capstone/capstone_report_template.md)

# 1. Introduction

## 1.1 Background

The hospitality industry has undergone a significant transformation with the advent of online hotel reservation channels. These platforms have revolutionized the way hotel bookings are made and have led to changes in customer behavior. As a result, hotels have had to adapt their strategies to accommodate these new booking trends.

## 1.2 Problem Statement

One of the challenges faced by hotels is the high number of reservation cancellations or no-shows. There are various reasons why guests cancel their reservations, such as changes in plans or conflicting schedules. To attract customers, many hotels offer flexible cancellation policies, allowing guests to cancel free of charge or at a low cost. While this benefits the guests, it can pose revenue challenges for hotels.

## 1.3 Objective

The objective of this analysis is to explore a dataset called "Hotel Reservation.csv" and leverage machine learning classification models to gain insights into reservation cancellations. By understanding the factors that contribute to cancellations, hotels can make informed decisions to optimize their revenue and operational strategies.

## 1.4 Data Overview

The "Hotel Reservation.csv" dataset contains 36275 rows and 19 columns. Each row represents a hotel reservation, and the columns provide information about the reservation details, guest demographics, and booking attributes. The dataset will serve as the foundation for our analysis and machine learning modeling.

# 2. Analysis

![Fig. 1: Variables Correlation table](images/var_correlation.png)
Fig. 1: Variables Correlation table

The table indicates that there are minimal correlations among the variables, except for the relationship between repeated_guests and their previous bookings, which is likely attributed to scheduling delays.

## 2.1 Data Exploration

### **Categorical Variables**

The analysis of categorical variables provides insights into the booking preferences and patterns of the hotel's guests.

![Fig. 2: Booking preferences 1](images/var2.png)
Fig. 2: Booking preferences 1

- **Rooms**: The majority of reservations are made for rooms with two adults and without children.

- **Booking Duration**: Reservations are primarily short-term, with fewer long-term bookings.

- **Meal Plan and Car Space**: Most guests select Meal Plan 1 and do not require a car space.

- **Room Types**: The most frequently booked room types are Types 1 and 4.

- **Booking Year**: The data predominantly corresponds to bookings made in 2018.

![Fig. 3: Booking preferences 2](images/var3.png)
Fig. 3: Booking preferences 2

- **Bookings by Month**: There is a gradual increase in bookings from January to October, followed by a drop in November and December.

- **Booking Method**: The majority of bookings are made online.

- **Guests' History**: Guests who have never been to the hotel and have never canceled a booking are more likely to make reservations.

- **Special Requests**: The available data decreases as the number of special requests increases.

- **Cancellation Rate**: Approximately 65% of reservations have not been canceled.

### **Continuous Variables**

Analyzing continuous variables helps understand the numerical aspects of the booking data.

![Fig. 4: Booking preferences 4](images/var3.png)
Fig. 4: Booking preferences 4

- **Lead Time**: Reservations are generally made without significant delays.

- **Lead Time and Reservations**: The longer the lead time, the lower the number of reservations.

- **Average Room Price**: The average price of a room is around 100 euros.

- **Cancellation History**: Guests typically have not canceled bookings before.

### **Bivariate Analysis: Relationship with Variables**

By examining the relationship between categorical variables and the likelihood of cancellation, several patterns emerge:

![Fig. 5](images/bivariate%20analysis.png) 
![Fig. 5](images/bivariate%20analysis%202.png)
![Fig. 5](images/bivariate%20analysis%203.png) 
![Fig. 5](images/bivariate%20analysis%204.png) 
![Fig. 5](images/bivariate%20analysis%205.png)
![Fig. 5](images/bivariate%20analysis%206.png) 

Fig. 5: Booking preferences and cancellation

- **Parking Space and Previous Stay**: Guests who request a parking space or have stayed previously at the hotel are less likely to cancel their reservations.

- **Special Requests**: Guests who request special requests are also less likely to cancel. Moreover, as the number of special requests increases, the likelihood of cancellation decreases.

- **Lead Time**: The lead time, or the time between booking and check-in, shows a consistent relationship with the likelihood of cancellation across various categorical variables:
  - Guests requesting a parking space or those who have stayed before tend to have shorter lead times.
  - Guests with longer stays (3 or 4 nights) generally have longer lead times.
  - Guests with Meal Plan 2 exhibit higher lead times.
  - Different room types have varying lead times.
  - As the number of special requests increases, the lead time tends to decrease.
  - Guests who are already familiar with the hotel tend to have lower lead times.

- **Cancellations and Room Prices**: Higher-priced rooms tend to have a higher number of cancellations, and this pattern remains consistent throughout the year.

### **Other Insights**

- Lead time tends to be shorter at the beginning and end of the year.

- Guests who have previously canceled their bookings and guests who have not stayed at the hotel before follow similar patterns, with shorter lead times.

- Higher lead time corresponds to higher prices for cancellations.

These findings provide valuable insights into the relationship between categorical and continuous variables, shedding light on the preferences, behavior, and trends of the hotel's guests.

## 2.2 Algorithms and Techniques

In our machine learning project on hotel reservation prediction, we experimented with seven different models: K Neighbors, Naive Bayes - Gaussian, Decision Tree, Random Forest, Extra Trees, Gradient Boosting, and Neural Network. Each of these models employs a unique algorithm or technique to make predictions based on the input data.

- **K Neighbors**: This algorithm predicts the class of a data point by considering the class labels of its nearest neighbors. The number of neighbors to consider is determined by the value of 'k'. **Speed**: moderate, **Accuracy**: high

- **Naive Bayes - Gaussian**: This algorithm is based on Bayes' theorem and assumes that the features are conditionally independent. It uses the Gaussian distribution to model the likelihood of each feature given the class labels. **Speed**: fast, **Accuracy**: moderate

- **Decision Tree**: Decision trees partition the feature space based on a series of binary decisions. Each internal node represents a test on a specific feature, while each leaf node represents a class label. The decision tree algorithm learns an optimal tree structure from the training data. **Speed**: fast, **Accuracy**: moderate to high

- **Random Forest**: Random Forest is an ensemble learning method that combines multiple decision trees. It creates a set of decision trees on random subsets of the training data and makes predictions by averaging the outputs of individual trees. **Speed**: moderate, **Accuracy**: high

- **Extra Trees**: Extra Trees, similar to Random Forest, is an ensemble learning method that combines multiple decision trees. However, it further randomizes the tree construction process by selecting random splits at each node, aiming to increase diversity among the trees. **Speed**: moderate to fast, **Accuracy**: high

- **Gradient Boosting**: Gradient Boosting is an iterative ensemble method that builds a strong predictive model by sequentially adding weak models. Each weak model is trained to correct the mistakes made by the previous models, focusing on the instances that were misclassified. **Speed**: moderate to slow, **Accuracy**: high

- **Neural Network**: Neural networks are a class of models inspired by the structure and functioning of biological neural networks. They consist of interconnected layers of artificial neurons that process input data and learn to make predictions through a process called backpropagation. **Speed**: moderate to slow, **Accuracy**: high

## 2.3 Benchmark

To evaluate the performance of our models in predicting hotel reservation cancellations, we followed a standard practice of splitting the dataset into a training set and a test set. The dataset was divided at a ratio of 70% for training and 30% for testing. This splitting strategy ensures that the models are trained on a substantial amount of data while also allowing us to assess their performance on unseen data.

We employed the scikit-learn library to calculate the accuracy score and create a confusion matrix for evaluating the models. The accuracy score measures the proportion of correctly predicted cancellations or non-cancellations out of the total predictions. It provides an overall assessment of the model's predictive accuracy.

Additionally, we utilized a confusion matrix to gain a more detailed understanding of the model's performance. The confusion matrix presents a tabular representation of predicted and actual classes, showing the true positive, true negative, false positive , and false negative values. This matrix allows us to calculate various evaluation metrics such as precision, recall, and F1 score and determine which model excels at predicting hotel reservation cancellations, where **precision** is the proportion of true positive predictions out of all positive predictions, **recall** is the proportion of true positive predictions out of all actual positive instances and **F1 score** is a measure that combines both precision and recall into a single metric.

# 3. Methodology

## 3.1 Data Preprocessing

In this section, we will describe the steps taken to preprocess the data for machine learning models. The preprocessing steps include removing irrelevant features, encoding categorical variables, balancing the classes, scaling the data, and splitting it into training and testing sets.

- **Feature Removal**:
The *'Booking_ID'* feature is removed from the dataset as it does not provide any meaningful information for predicting the booking status.

- **Label Encoding**:
Categorical variables in the dataset are encoded into numeric form using the LabelEncoder from the scikit-learn library. This transformation allows the machine learning models to work with categorical data.

- **Class Balancing**:
As the dataset may have imbalanced classes, where one class has significantly more instances than the others, we apply oversampling using the RandomOverSampler from the imbalanced-learn library. This technique increases the number of instances in the minority class to achieve a balanced representation.

![Fig. 6](images/balanced_status.png) 

Fig.6: Post-balancing booking statuses

- **Data Scaling**:
To ensure that all features are on a similar scale and prevent any particular feature from dominating the learning process, we apply standard scaling using the StandardScaler from the scikit-learn library. This transformation standardizes the features to have zero mean and unit variance.

- **Data Split**:
The preprocessed data is split into training and testing sets using the train_test_split function from the scikit-learn library. We allocate 30% of the data for testing, while the remaining 70% is used for training the machine learning models.

## 3.2 Implementation

### 1, K-NN

In this implementation, the K-Neighbors model was used to classify the data. The value of k determines the number of neighbors considered in the classification decision.

To ensure optimal performance on unseen data, we evaluate the model's performance using the cross-validation technique.

![Fig. 7](images/loss_acc.png) 

Fig. 7: Loss/accuracy for different values of k

As the value of k increases, both the training accuracy and validation accuracy tend to decrease slightly. This can be attributed to the fact that increasing the number of neighbors introduces more noise or mislabeled points from the training set, affecting the model's performance.

The model achieved the highest training accuracy of 0.9929 for k = 1. However, the validation accuracy for this value of k is slightly lower at 0.8734, indicating a possible overfitting situation. Overfitting occurs when a model performs exceptionally well on the training dataset but struggles to generalize to unseen data.

The validation accuracy stabilizes after k = 4, with slight fluctuations in subsequent values. This suggests that a range of k values around 4 could be considered for potential model selection, as they provide reasonably good performance without overfitting to the training data.

The following are the training and validation accuracies for different values of k:

For k = 1: Training Accuracy = 0.9929, Validation Accuracy = 0.8734
For k = 2: Training Accuracy = 0.9123, Validation Accuracy = 0.8236
For k = 3: Training Accuracy = 0.9156, Validation Accuracy = 0.8209
For k = 4: Training Accuracy = 0.8745, Validation Accuracy = 0.7975
For k = 5: Training Accuracy = 0.8722, Validation Accuracy = 0.7986
For k = 6: Training Accuracy = 0.8513, Validation Accuracy = 0.7865
For k = 7: Training Accuracy = 0.8493, Validation Accuracy = 0.7897
For k = 8: Training Accuracy = 0.8365, Validation Accuracy = 0.7802
For k = 9: Training Accuracy = 0.8342, Validation Accuracy = 0.7793

![Fig. 8](images/kn_matrix.png) 

Fig. 8: K-NN Confusion matrix

![Fig. 9](images/knn_result.png) 

Fig. 9: K-NN performance

**NOTE**: reservation cancelled - denoted by class 0, not cancelled, denoted by class 1

Based on the results obtained, the model appears to perform reasonably well. After considering the training and validation accuracies, as well as the precision, recall, F1-score, and overall accuracy, a value of k = 3 was chosen for the K-Neighbors model. This value provides a good balance between performance and potential overfitting.

***Overall accuracy : 82%***

### 2, Naive Bayes - Gaussian Model

![Fig. 10](images/gauss_smoothing.png) 

Fig. 10: Hyperparameters smoothing

In this section, we implemented the Naive Bayes algorithm with a Gaussian model. Naive Bayes is a probabilistic machine learning algorithm that applies Bayes' theorem with the assumption of independence between the features. The Gaussian model assumes that the features in the dataset follow a Gaussian (normal) distribution.

After testing different parameters for priors and var_smoothing, we decided to use priors=[0.01,0.99] and var_smoothing=1e-6. The priors represent the probability distribution of the classes in the dataset, and var_smoothing is a smoothing parameter that helps prevent numerical instabilities when computing probabilities.

The Naive Bayes Gaussian model calculates the likelihood of a data point belonging to a particular class by estimating the probability density function (PDF) of each feature given the class. It then combines these likelihoods with the prior probabilities of the classes to make predictions.

The model's accuracy on the training data was 0.748, while the accuracy on the test/validation data was 0.744. Comparing these accuracy scores, it seems that the model is not overfitting as the accuracy on the training data is relatively similar to the accuracy on the test/validation data.

![Fig. 11](images/gauss_matrix.png) 

Fig. 11: Confusion matrix

![Fig. 12](images/gauss_perf.png) 

Fig. 12: Performace

Regarding the metrics, we evaluated the model's performance for each class as well as an overall average (macro avg and weighted avg) across all classes. The precision, recall, and F1-score were relatively similar for both classes, indicating a balanced performance. However, the overall accuracy of 0.74 suggests that the model is correctly predicting approximately 74% of the instances in the dataset.

***Overall accuracy : 74%***

### 7, Neural Network
In this section, we employed a neural network (ANN) to address our problem. The neural network is inspired by biological neural networks in the human brain and consists of interconnected artificial neurons that process and learn from data.

During training, we initially set the model to train for 200 epochs with a learning rate of 0.01. We observed promising results in terms of training and validation loss up to 100 epochs. However, beyond that point, the loss and accuracy lines began to diverge, indicating overfitting, where the model becomes too specialized in the training data.

To combat overfitting, we decided to stop training early at 100 epochs to prevent the model from memorizing the training data and encourage generalization. Additionally, we reduced the learning rate to 0.001. This adjustment allowed for finer weight adjustments, resulting in slower convergence but potentially greater precision. Lowering the learning rate also improved training stability and reduced the risk of overshooting the optimal solution.

![Neural_loss](images/neural_loss.png) 
![Neural_acc](images/neural_acc.png) 

By training the neural network with 100 epochs and a reduced learning rate of 0.001, we aimed to strike a balance between convergence speed and precision. This approach helped us capture meaningful patterns from the data while avoiding overfitting and unstable training dynamics.

![Neural_loss](images/neural_loss.png) 
![Neural_acc](images/neural_acc.png) 

***Overall accuracy : 87.87%***

# 4. Results

## 4.1 Model Evaluation and Validation

## 4.2 Justification

# 5. Conclusion

## 5.1 Free-Form Visualization

## 5.2 Reflection

## 5.3 Improvement
