#!/usr/bin/env python
# coding: utf-8

# In[7]:


#importing necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[8]:


#importing the dataset
churn_data = pd.read_csv('/Users/jeevithaselvaraj/Downloads/Churn_Modelling.csv')


# In[9]:


# To understand the data
churn_data


# In[4]:


# to understand the data checking the first 5 rows of the dataset
churn_data.head()


# In[5]:


# to check how many rows in the datset 
churn_data.tail()


# In[6]:


# to check the number of rows and columns in a dataframe using shape attribute
churn_data.shape


# In[7]:


print("Number of Rows",churn_data.shape [0])
print("Number of Columns",churn_data.shape [1])


# In[8]:


churn_data.info()


# .info()function gives the precise summary of the dataset, which is crucial for carrying out an effective data analysis and preprocessing tasks.

# In[9]:


# checking the null values
churn_data.isnull()


# .isnull() is used to identify missing or null values in a DataFrame. This function is vital for data preprocessing and cleaning tasks. It gives the boolean value of the dataset indicating, whether each element is a null value or not.

# In[10]:


churn_data.isnull().sum()


# .isnull().sum() is used to calculate and display the count of missing values for each column in the DataFrame.It is essential for assessing the data quality, guiding data cleaning and imputation processes, for handling missing values in our analysis.

# In[82]:


# Dropping inappropriate columns that is not required for this analysis
churn_data.columns


# In[8]:


churn_data = churn_data.drop(['RowNumber', 'CustomerId', 'Surname'],axis=1)


# The irrelevant columns are dropped to reduce the memory usage and for the ease of analysis purposes

# In[15]:


churn_data.info()


# After dropping the irrelevant attributes there is a significant decrease in the memory usage from 1.1+ MB to 859.5+ KB, the former is the one, before dropping the irrelevant columns.

# In[16]:


# Encoding ctaegorical data
churn_data['Geography'].unique()


# In[84]:


churn_data['Gender'].unique()


# Before encoding categorical variables into numerical representations, we want to know the unique categories present in the column. To identify possible misspellings, inconsistencies, or unexpected values that need to be cleaned or corrected.Identifying unique values can help us in deciding the data preprocessing tasks

# In[10]:


# Encoding categorical data
# one hot/ dummy encoding
# Based on Pandas Get Dummies – pd.get_dummies() | Data Independent, 2023. [online]. Data Independent. Available from: https://dataindependent.com/pandas/pandas-get-dummies-pd-get_dummies/.
churn_data = pd.get_dummies(churn_data, drop_first = True)


# This is used to perform one-hot encoding. It transforms categorical variables into a binary representation (0 or 1) for each category, creating new columns (dummy variables) for each unique category in the original column. It is used for converting categorical variables into a numerical format suitable for machine learning algorithms.drop_first = True It is a boolean parameter that is used to remove the first dummy variable

# In[11]:


churn_data


# Checking the data after one hot encoding

# In[87]:


churn_data.Exited.plot.hist()


# In[19]:


churn_data['Exited'].value_counts()


#  The 'Exited' column is a categorical or binary variable representing whether a customer churned or not.Here value counts() function is used to find the distribution of targeted variable. We can understand the exzct count of instances falling into each category, which is important for understanding the class distribution and potential class imbalances. The output will be sorted in the descending order with the firat element being the frequently occurring value. 0 means the customer is not leaving the bank and 1 means the customer is leaving the bank.
# 

# In[120]:


# dropping the target variable and storing the data with independent variables in a new data frame 
churn_data1=churn_data.drop(columns='Exited')


# ## Correlation Matrix

# In[121]:


# Creating a correlation matrix between the independent and response variable
churn_data1.corrwith(churn_data['Exited']).plot.bar(figsize=(15,7), title='correlation With the exited variable', rot = 45,grid = True)


# Correlation Matrix is performed to identify the correlation between each feature and the target variable 'Exited'.It calculates the correlation between each column in churn_data1 and the 'Exited' column. The height of each bar represents the strength and direction of the correlation between that specific column and the 'Exited' variable.
# 
# A positive correlation coefficient close to 1 indicates that as the values of the column increases, the chance for the customer 'Exiting' also increases, which shows a positive relationship between the column and the churn rate. Here Age, Balance and Geography_Germany have a positive coorelation which are more likely to churn.
# 
# A negative correlation coefficient close to -1 indicates that as the values of the column increase, the likelihood of the customer 'Exiting' decreases. This suggests a negative relationship between the column and the churn rate.Here,IsActiveMember and Gender_Male is close to -1 and it is negatively correlated and means that they are less likely to churn.
# 
# Correlation coefficients close to 0 indicate a weak or no linear relationship between the column and the 'Exited' variable. Here, CreditScore, Tenure, NumOforoducts and Geography_spain are close to 0 which has no correlation between the 'exited' variable.

# ## Heat Map

# In[118]:


# Creating a heat map
corr=churn_data.corr()
plt.figure(figsize=(15,7))
sns.heatmap(corr,annot=True)


#  This heatmap provides a visual representation of the correlations between the numeric variables in the dataset. Positive correlations are displayed in warmer colors like reds, while negative correlations are shown in cooler colors.

# In[311]:


import warnings
warnings.filterwarnings('ignore')


# ### Count plot for identifying the distribution of the target variable

# In[312]:


# Visualising
sns.countplot(churn_data['Exited'])


# It is used to identify the distribution of the target variables. From this count plot we can find that the target variable is unevenly distributed and we can see that the data is highly imbalanced.

# ## Data Splitting

# In[12]:


#importing train_test_split 
from sklearn.model_selection import train_test_split
# Storing the independent variable in A
A = churn_data.drop(columns='Exited')
# storing the target variable in B
B = churn_data['Exited']


# In[30]:


# Splitting the dataset
A_train, A_test, B_train, B_test = train_test_split(A, B, test_size=0.2, random_state=42, stratify=B)


# Data splitting involves dividing the dataset into separate subsets for training, validation, and testing. This ensures that our model is trained on one set of data, validated on another, and evaluated on another set of data it has never seen before. This helps estimate the model's performance on new, unseen data.random_state allows to reproduce the same random behavior every time when we run our code.  Here stratify is used to enforce the same class balance on the train and test data as the original data.

# In[22]:


A_train.shape


# In[23]:


A_test.shape


# # Feature Scaling 

# In[14]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
A_train = scaler.fit_transform(A_train)
A_test = scaler.transform(A_test)


# Feature scaling is a preprocessing technique used  to standardize or normalize the range of independent variables (features) in our dataset. Scaling ensures that all features have similar magnitudes,  It is vital to perform Feature scaling where all the variables are brought to a similar scale, as the features with high value range dominates in calculating the distances between data.

# In[15]:


A_train


# # Logistic regression (LR)
# ### Analysis before applying SMOTE

# In[36]:


from sklearn.linear_model import LogisticRegression
# Creating instance for this model
logis = LogisticRegression()
# train the model
logis.fit(A_train,B_train)
# Predict the model
B_pred = logis.predict(A_test)


# LogisticRegression class was imported from scikit-learn and created an instance called logis. The .fit trains the logistic regression model (logis) using the training data where A_train for features and B_train for target variables. The trained logistic regression model make predictions on the test data which is A_test. The predicted labels for the test set are stored in the B_pred. 

# In[130]:


# Checking accuracy before applying SMOTE
# importing necessary libraries
from sklearn.metrics import accuracy_score
accuracy_score(B_test,B_pred)


# In[131]:


# Checking precision score before applying SMOTE
# importing necessary libraries
from sklearn.metrics import precision_score,recall_score,f1_score
precision_score(B_test,B_pred)


# In[132]:


# Recall score before applying SMOTE
recall_score(B_test,B_pred)


# In[133]:


# f1 score before applying SMOTE
f1_score(B_test,B_pred)


# Handling class imbalance is an important step, where one class (churned) might be significantly less frequent than the other (not churned).Beacuse of the imbalanced dataset, we are performing the performance metircs and there is a high difference in the performance of the logistic regression model. From the results, we can infer that, Logistic Regression model with an imbalance dataset predicted a biased model tending to predict the majority class more accurately while neglecting the minority class. The model did not learn the underlying patterns of the minority class, resulting in poor generalization. There is a high accuracy of 0.80 and precison of 0.58, recall of 0.18 which is very low and fi score of 0.28 which is again very low. It is crucial to apply resampling techniques to handle this imbalance.

# ## Bar plot for performance metrics for Logistic Regression before SMOTE

# In[208]:


# creating a bar plot for performance metrics for Logistic Regression
accuracy = 0.80
precision = 0.58
recall = 0.18
f1_score = 0.28
LR_df = pd.DataFrame({'Metric':['Accuracy','Precision','Recall','F1 Score'],
                      'Score':[accuracy,precision,recall,f1_score]})
plt.figure(figsize=(8,6))
plt.bar(LR_df['Metric'], LR_df['Score'], color=['gray','red','pink','brown'])
plt.ylim(0,1.0)
plt.xlabel('Metric')
plt.ylabel('Score')
plt.title('performance metrics for Logistic Regression in Forecasting Customer Churn')
plt.tight_layout()
plt.show()


# From the bar plot we can visualise the performance of the Logistic regression Model. The plot depicts a biased performance of the model with an imbalanced datset.It predicted a biased model tending to predict the majority class more accurately while neglecting the minority class. 

# # Applying Oversampling with SMOTE
# 

# In[17]:


# Applying Oversampling with SMOTE
# importing SMOTE
# Based on ML Handling Imbalanced Data with SMOTE and Near Miss Algorithm in Python, 2023. [online]. GeeksforGeeks. Available from: https://www.geeksforgeeks.org/ml-handling-imbalanced-data-with-smote-and-near-miss-algorithm-in-python/.
from imblearn.over_sampling import SMOTE
# Assigning in new variable
A_new,B_new = SMOTE().fit_resample(A,B)


# From imblearn over sampling, SMOTE was imported and the resampled data was stored in new variables A_new and B_new.To handle the imbalance we are applying an oversampling technique called SMOTE. It oversamples the minority class using replacement. It balances the class distribution by randomly increasing the minority class instances by replacing them  

# In[135]:


B_new.value_counts()


#  Value_counts() is used to check the exact count of the distribution of target variable. Now after applying SMOTE we can see that, there is an even distribution in majority and minority class 

# In[18]:


# Splitting the dataset with new dataset with SMOTE
A_train, A_test, B_train, B_test = train_test_split(A_new, B_new, test_size=0.2, random_state=42)


# In[284]:


# Create a DataFrame with resampled data
resampled_data = pd.DataFrame(A_new, columns=A.columns)
resampled_data["Churn"] = B_new

# Setting the style of seaborn
sns.set(style="whitegrid")

# Creating a count plot
plt.figure(figsize=(6, 4))
sns.countplot(x="Churn", data=resampled_data, palette="Set2")

# Adding labels and title
plt.xlabel("Exited")
plt.ylabel("Count")
plt.title("Distribution of Churn after SMOTE")

# Show the plot
plt.show()


# # Feature Scaling after SMOTE

# In[39]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
A_train = scaler.fit_transform(A_train)
A_test = scaler.transform(A_test)


# # Logistic Regression(LR) 
# ### After appying SMOTE

# In[19]:


from sklearn.linear_model import LogisticRegression
# Creating instance for this model
logis = LogisticRegression()
# train the model
logis.fit(A_train,B_train)
# Predict the model
B_pred = logis.predict(A_test)


# In[65]:


# accuracy after SMOTE
accuracy_score(B_test, B_pred)


# In[140]:


# precision score after SMOTE
precision_score(B_test,B_pred)


# In[141]:


# recall score after SMOTE
recall_score(B_test,B_pred)


# In[142]:


# f1 score after SMOTE
f1_score(B_test,B_pred)


# After applying SMOTE, the model's performance is fairly balanced, with similar precision, recall, and F1-scores which is aound 77% for both exited and non-exited classes. The accuracy of 78% suggests that the model is doing a decent job of predicting churn.

# ## roc_auc_score for Logistic Regression Model

# In[300]:


# Predict probabilities for positive class(exited)
from sklearn.metrics import roc_curve, roc_auc_score
# Based on BROWNLEE, J., 2021. How to Use ROC Curves and Precision-Recall Curves for Classification in Python. [online]. MachineLearningMastery.com. Available from: https://machinelearningmastery.com/roc-curves-and-precision-recall-curves-for-classification-in-python/#:~:text=The%20AUC%20for%20the%20ROC,skill%20and%20perfect%20skill%20respectively.https://machinelearningmastery.com/roc-curves-and-precision-recall-curves-for-classification-in-python/#:~:text=The%20AUC%20for%20the%20ROC,skill%20and%20perfect%20skill%20respectively.
B_prob = logis.predict_proba(A_test)[:, 1]
# Calculating ROC curve
fpr, tpr, thresholds = roc_curve(B_test,B_prob)
# Calculate AUC score
auc_score = roc_auc_score(B_test, B_prob)
print("AUC:",auc_score)


# Calculating the ROC curve and AUC score for a logistic regression model's predictions on a test dataset. 
# The AUC score is a measure of how well the model can discriminate between the positive and negative classes, with higher values indicating better performance. 
# An AUC score of 0.8627 is relatively high, indicating that the logistic regression model has a good ability to differentiate between positive and negative instances. The higher the AUC, the better the model's discriminatory power. 
# The model is making well-informed predictions and has learned meaningful patterns from the data. 
# A score of 0.8627 suggests that the model is achieving a good balance between correctly identifying positive instances and avoiding false positives.

# ## Confusion Matrix for evaluating the Logistic Regression Model

# In[213]:


# importing confusion matrix for evaluating the model using Logistic Regression
# Based on BHANDARI, A., 2023. Understanding &#038; Interpreting Confusion Matrices for Machine Learning (Updated 2023). [online]. Analytics Vidhya. Available from: https://www.analyticsvidhya.com/blog/2020/04/confusion-matrix-machine-learning/#:~:text=A%20confusion%20matrix%20is%20a%20performance%20evaluation%20tool%20in%20machine,false%20positives%2C%20and%20false%20negatives.
# Based on ML Handling Imbalanced Data with SMOTE and Near Miss Algorithm in Python, 2023. [online]. GeeksforGeeks. Available from: https://www.geeksforgeeks.org/ml-handling-imbalanced-data-with-smote-and-near-miss-algorithm-in-python/.
from sklearn.metrics import confusion_matrix, classification_report
# Calculating the confusion matrix
conf_matrix = confusion_matrix(B_test, B_pred)
# printing the confusion matrix
print("confusion_matrix:\n", conf_matrix)
# calculating the performance metrics for Logistic Regression Model
print("classification Report:]\n", classification_report(B_test, B_pred))


# From the above confusion matrix, we can infer that the True Positives have 1211instances those were actually exited which is positive class and were correctly predicted as exited. 1271 instances were not actually exited which is a negative class and were correctly predicted as not exited. The number of False Positives have 362 instances who were not actually churned but were wrongly predicted as exited. The False Negatives have 342 instances who were actually exited but were incorrectly predicted as not exited.

# ## Confusion Matrix Heat Map for Logistic Regression Model

# In[285]:


confusion_matrix = np.array([[1271, 362],[342,1211]])
# Creating a heat map using seaborn
sns.set()
sns.heatmap(confusion_matrix, annot=True, cmap="cividis", fmt=".2f",
            xticklabels=["Non-Churned", "Churned"],
            yticklabels=["Non-Churned", "Churned"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Logistic Regression Model's Confusion Matrix")
plt.show()


# Visual presentation of the confusion matrix. From the  confusion matrix heatmap, we can infer that the True Positives have 1211instances those were actually exited which is positive class and were correctly predicted as exited. 1271 instances were not actually exited which is a negative class and were correctly predicted as not exited. The number of False Positives have 362 instances who were not actually churned but were wrongly predicted as exited. The False Negatives have 342 instances who were actually exited but were incorrectly predicted as not exited.

# ## Bar plot for performance metrics for Logistic Regression

# In[133]:


# creating a bar plot for performance metrics for Logistic Regression
accuracy = 0.79
precision = 0.76
recall = 0.77
f1_score = 0.77
LR_df = pd.DataFrame({'Metric':['Accuracy','Precision','Recall','F1 Score'],
                     'Score':[accuracy,precision,recall,f1_score]})
plt.figure(figsize=(8,6))
plt.bar(LR_df['Metric'], LR_df['Score'], color=['lightblue','green','orange','gray'])
plt.ylim(0,1.0)
plt.xlabel('Metric')
plt.ylabel('Score')
plt.title('performance metrics for Logistic Regression in Forecasting Customer Churn')
plt.tight_layout()
plt.show()


# Visual Presentation of the LR model with the performance metrics scores. From this we infer that the model's performance is fairly balanced, with similar precision, recall, and F1-scores around 77% for both exited and non-exited classes. The accuracy of 78% suggests that the model is doing a decent job of predicting churn.

# ## Support Vector Classifier (SVC)

# In[159]:


# importing svm
from sklearn import svm
# creating instance
svm = svm.SVC(probability = True)
# training the model
svm.fit(A_train,B_train)


# The SVM classifier (svm) is trained using the .fit() method. The A_train represents the training data features, and B_train represents the corresponding target labels. The SVM classifier has been trained using the training data. This model will be used to make predictions on new, unseen data.Setting probability=True when creating an SVM model allows you to obtain probability estimates for the predicted classes. It is used to indicate whether the SVM should be trained to provide class probabilities in addition to making class predictions. 
# 
# 

# In[36]:


# predicting the model
B_pred1 = svm.predict(A_test)


# In[41]:


# checking the accuracy
from sklearn.metrics import accuracy_score
accuracy_score(B_test, B_pred1)


# In[42]:


# checking the precision score
from sklearn.metrics import precision_score
precision_score(B_test,B_pred1)


# In[43]:


# checking the recall score
from sklearn.metrics import recall_score
recall_score(B_test,B_pred1)


# In[44]:


# checking the f1 score
from sklearn.metrics import f1_score
f1_score(B_test,B_pred1)


# The SVC model demonstrates equitable performance, showing comparable precision, recall, and F1-scores for both churned and non-churned classes. The 79% accuracy indicates that the model is performing adequately in predicting churn.

# ## roc_auc_score for Support Vector Classifier

# In[202]:


# Predict probabilities for positive class(exited)
from sklearn.metrics import roc_curve, roc_auc_score
# Based on BROWNLEE, J., 2021. How to Use ROC Curves and Precision-Recall Curves for Classification in Python. [online]. MachineLearningMastery.com. Available from: https://machinelearningmastery.com/roc-curves-and-precision-recall-curves-for-classification-in-python/#:~:text=The%20AUC%20for%20the%20ROC,skill%20and%20perfect%20skill%20respectively.https://machinelearningmastery.com/roc-curves-and-precision-recall-curves-for-classification-in-python/#:~:text=The%20AUC%20for%20the%20ROC,skill%20and%20perfect%20skill%20respectively.
B_prob1 = svm.predict_proba(A_test)[:, 1]
# Calculating ROC curve
fpr, tpr, thresholds = roc_curve(B_test,B_prob1)
# Calculate AUC score
auc_score = roc_auc_score(B_test, B_prob1)
print("AUC:",auc_score)


# The calculated AUC score of approximately 0.8789. This indicates that the SVM classifier's performance in separating the positive and negative instances is relatively good, as the AUC score is reasonably close to 1. A higher AUC score implies better performance and its ability to discriminate between the two classes.

# # Confusion Matrix for evaluating Support Vector machine

# In[46]:


# importing confusion matrix for evaluating the model using Support Vector Classifier
# Based on BHANDARI, A., 2023. Understanding &#038; Interpreting Confusion Matrices for Machine Learning (Updated 2023). [online]. Analytics Vidhya. Available from: https://www.analyticsvidhya.com/blog/2020/04/confusion-matrix-machine-learning/#:~:text=A%20confusion%20matrix%20is%20a%20performance%20evaluation%20tool%20in%20machine,false%20positives%2C%20and%20false%20negatives.
# Based on ML Handling Imbalanced Data with SMOTE and Near Miss Algorithm in Python, 2023. [online]. GeeksforGeeks. Available from: https://www.geeksforgeeks.org/ml-handling-imbalanced-data-with-smote-and-near-miss-algorithm-in-python/.
from sklearn.metrics import confusion_matrix, classification_report
# Calculating the confusion matrix
conf_matrix = confusion_matrix(B_test, B_pred1)
# printing the confusion matrix for Support Vector machine
print("confusion_matrix:\n", conf_matrix)
print("classification Report:]\n", classification_report(B_test, B_pred1))


# ## Confusion Matrix Heat Map For Support Vector machine

# In[291]:


confusion_matrix = np.array([[1295, 338],[321,1232]])
# Creating a heat map using seaborn
sns.set()
sns.heatmap(confusion_matrix, annot=True, cmap="RdPu", fmt=".2f",
            xticklabels=["Non-Churned", "Churned"],
            yticklabels=["Non-Churned", "Churned"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Support vector Machine's Confusion Matrix")
plt.show()


# From the confusion matrix heat map, we can infer that the True Positives have 1232 instances those were actually exited which is positive class and were correctly predicted as exited. 1295 instances were actually not exited which is a negative class and were correctly predicted as not exited. The number of False Positives have 338 instances who were actually not churned but were wrongly predicted as exited. The False Negatives have 321 instances who were actually exited but were incorrectly predicted as not churned by the svc model.

# In[132]:


# creating a bar plot for performance metrics for support vector machine
accuracy = 0.79
precision = 0.78
recall = 0.79
f1_score = 0.78
SVC_df = pd.DataFrame({'Metric':['Accuracy','Precision','Recall','F1 Score'],
                     'Score':[accuracy,precision,recall,f1_score]})
plt.figure(figsize=(8,6))
plt.bar(SVC_df['Metric'], SVC_df['Score'], color=['brown','pink','purple','lightblue'])
plt.ylim(0,1.0)
plt.xlabel('Metric')
plt.ylabel('Score')
plt.title('performance metrics for Support Vector machine in Forecasting Customer Churn')
plt.tight_layout()
plt.show()


# The SVC model demonstrates equitable performance, showing comparable precision, recall, and F1-scores for both churned and non-churned classes. The 79% accuracy indicates that the model is performing adequately in predicting churn.

# ## KNearest Neighbor

# In[79]:


# importing KNeighbors classifier
from sklearn.neighbors import KNeighborsClassifier
# creating instance
knn = KNeighborsClassifier()
# training the model
knn.fit(A_train,B_train)


# The KNeighborsClassifier class from scikit-learn's neighbors module was imported and an instance of the k-NN classifier called knn was created. The .fit trains the KNN model using the training data where A_train for features and B_train for target variables. The trained Knn model make predictions on the test data which is A_test. The predicted labels for the test set are stored in the B_pred2.

# In[80]:


# Predicting the model
B_pred2 = knn.predict(A_test)


# In[81]:


# checking the accuracy
accuracy_score(B_test,B_pred2)


# In[82]:


# checking the precision score
precision_score(B_test,B_pred2)


# In[83]:


# checking the recall score
recall_score(B_test,B_pred2)


# In[84]:


# checking the f1 score
from sklearn.metrics import f1_score
f1_score(B_test,B_pred2)


# The model's performance is decent but could benefit from improvement, especially in terms of balancing precision and recall for both classes. It's important to consider the application implications of these results and potentially fine-tune the model or adjust the decision threshold to achieve the desired outcomes.

# ## roc_auc_score for KNearest Neighbor 

# In[203]:


# Predict probabilities for positive class(exited)
from sklearn.metrics import roc_curve, roc_auc_score
# Based on BROWNLEE, J., 2021. How to Use ROC Curves and Precision-Recall Curves for Classification in Python. [online]. MachineLearningMastery.com. Available from: https://machinelearningmastery.com/roc-curves-and-precision-recall-curves-for-classification-in-python/#:~:text=The%20AUC%20for%20the%20ROC,skill%20and%20perfect%20skill%20respectively.https://machinelearningmastery.com/roc-curves-and-precision-recall-curves-for-classification-in-python/#:~:text=The%20AUC%20for%20the%20ROC,skill%20and%20perfect%20skill%20respectively.
B_prob2 = knn.predict_proba(A_test)[:, 1]
# Calculating ROC curve
fpr, tpr, thresholds = roc_curve(B_test,B_prob2)
# Calculate AUC score
auc_score = roc_auc_score(B_test, B_prob2)
print("AUC:",auc_score)


# An AUC score of 0.8035 suggests that the k-NN classifier is achieving a good balance between correctly identifying positive instances True Positives and avoiding false positives False Positives. 

# ## Confusion Matrix for evaluating KNearest Neighbor

# In[85]:


#importing confusion matrix for evaluating the model using KNeighborsClassifier
# Based on BHANDARI, A., 2023. Understanding &#038; Interpreting Confusion Matrices for Machine Learning (Updated 2023). [online]. Analytics Vidhya. Available from: https://www.analyticsvidhya.com/blog/2020/04/confusion-matrix-machine-learning/#:~:text=A%20confusion%20matrix%20is%20a%20performance%20evaluation%20tool%20in%20machine,false%20positives%2C%20and%20false%20negatives.
from sklearn.metrics import confusion_matrix, classification_report
# Calculating the confusion matrix
conf_matrix = confusion_matrix(B_test, B_pred2)
# printing the confusion matrix for KNeighborsClassifier
print("confusion_matrix:\n", conf_matrix)
print("classification Report:]\n", classification_report(B_test, B_pred2))


# ## Confusion Matrix Heat Map for KNearest Neighbor

# In[287]:


confusion_matrix = np.array([[940, 693],[218,1335]])
# Creating a heat map using seaborn
sns.set()
sns.heatmap(confusion_matrix, annot=True, cmap="magma", fmt=".2f",
            xticklabels=["Non-Churned", "Churned"],
            yticklabels=["Non-Churned", "Churned"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("KNeighbor's Confusion Matrix")
plt.show()


# From the confusion matrix heat map, we infer that the model correctly predicted 940 instances as class 0 which is True Negatives. It incorrectly predicted 693 instances as class 1 when they were actually class 0 which is False Positives. It incorrectly predicted 218 instances as class 0 when they were actually class 1 which is False Negatives. It correctly predicted 1335 instances as class 1 which is True Positives.

# ## Bar plot for performance metrics for KNearest Neighbor
# 
# 

# In[290]:


# creating a bar plot for performance metrics for KNeighbors Classifier
accuracy = 0.71
precision = 0.65
recall = 0.85
f1_score = 0.74
KNN_df = pd.DataFrame({'Metric':['Accuracy','Precision','Recall','F1 Score'],
                     'Score':[accuracy,precision,recall,f1_score]})
plt.figure(figsize=(8,6))
plt.bar(KNN_df['Metric'], KNN_df['Score'], color=['gray','orange','lightblue','green'])
plt.ylim(0,1.0)
plt.xlabel('Metric')
plt.ylabel('Score')
plt.title('performance metrics for KNearest Neighbor in Forecasting Customer Churn')
plt.tight_layout()
plt.show()


# The model's performance is decent but could benefit from improvement, especially in terms of balancing precision and recall for both classes. It's important to consider the application implications of these results and potentially fine-tune the model or adjust the decision threshold to achieve the desired outcomes.

# # Decision Tree Model

# In[20]:


# importing DecisionTreeClassifier
from sklearn.tree import DecisionTreeClassifier
# creating instance
dt = DecisionTreeClassifier()
# training the model
dt.fit(A_train, B_train)


# The DecisionTreeClassifier class from scikit-learn's tree module and created an instance of the Decision Tree classifier called dt. The Decision Tree classifier (dt) using the training data where A_train is for features and B_train for target labels. The Decision Tree algorithm iteratively splits the data based on the selected features and their values to create a tree-like structure that makes predictions.

# In[90]:


# predicting the model
B_pred3 = dt.predict(A_test)


# In[91]:


# checking the accuracy
accuracy_score(B_test,B_pred3)


# In[93]:


# checking the precision score
precision_score(B_test,B_pred3)


# In[94]:


# checking the recall score
recall_score(B_test,B_pred3)


# In[96]:


# checking the f1 score
from sklearn.metrics import f1_score
f1_score(B_test,B_pred3)


# The Decision Tree model's performance is balanced, with similar precision, recall, and F1-score for both classes. The model demonstrates good predictive ability and is effectively classifying instances. However, further analysis and potential fine-tuning may be needed based on considerations.

# ## roc_auc_score for Decision Tree Model

# In[204]:


# Predict probabilities for positive class(exited)
from sklearn.metrics import roc_curve, roc_auc_score
# Based on BROWNLEE, J., 2021. How to Use ROC Curves and Precision-Recall Curves for Classification in Python. [online]. MachineLearningMastery.com. Available from: https://machinelearningmastery.com/roc-curves-and-precision-recall-curves-for-classification-in-python/#:~:text=The%20AUC%20for%20the%20ROC,skill%20and%20perfect%20skill%20respectively.https://machinelearningmastery.com/roc-curves-and-precision-recall-curves-for-classification-in-python/#:~:text=The%20AUC%20for%20the%20ROC,skill%20and%20perfect%20skill%20respectively.
B_prob3 = dt.predict_proba(A_test)[:, 1]
# Calculating ROC curve
fpr, tpr, thresholds = roc_curve(B_test,B_prob3)
# Calculate AUC score
auc_score = roc_auc_score(B_test, B_prob3)
print("AUC:",auc_score)


# The ROC-AUC score of 0.809 for the Decision Tree classifier indicates the model's performance in distinguishing between positive and negative instances. An AUC score of 0.809 is relatively high, indicating that the Decision Tree classifier has a good ability to differentiate between positive and negative instances. The AUC score of 0.809 suggests that the Decision Tree classifier is achieving a good balance between correctly identifying positive instances that is True Positives and avoiding false positives. 

# ## Confusion Matrix for evaluating the Decision Tree Model

# In[107]:


#importing confusion matrix for evaluating the model using Decision Tree Model
# Based on BHANDARI, A., 2023. Understanding &#038; Interpreting Confusion Matrices for Machine Learning (Updated 2023). [online]. Analytics Vidhya. Available from: https://www.analyticsvidhya.com/blog/2020/04/confusion-matrix-machine-learning/#:~:text=A%20confusion%20matrix%20is%20a%20performance%20evaluation%20tool%20in%20machine,false%20positives%2C%20and%20false%20negatives.
from sklearn.metrics import confusion_matrix, classification_report
# Calculating the confusion matrix
conf_matrix = confusion_matrix(B_test, B_pred3)
# printing the confusion matrix for Decision Tree Model
print("confusion_matrix:\n", conf_matrix)
print("classification Report:]\n", classification_report(B_test, B_pred3))


# ## Confusion Matrix Heat Map for Decision Tree Model

# In[288]:


confusion_matrix = np.array([[1267, 366],[245,1308]])
# Creating a heat map using seaborn
sns.set()
sns.heatmap(confusion_matrix, annot=True, cmap="inferno", fmt=".2f",
            xticklabels=["Non-Churned", "Churned"],
            yticklabels=["Non-Churned", "Churned"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Decision Tress Model's Confusion Matrix")
plt.show()


# From the confusion matrix heatmap, we can infer that, the model correctly predicted 1267 instances as class 0 which is True Negatives. It incorrectly predicted 366 instances as class 1 when they were actually class 0 which is False Positives. It incorrectly predicted 245 instances as class 0 when they were actually class 1, False Negatives. It correctly predicted 1308 instances as class 1, True Positives.

# ## Bar plot for performance metrics for Decision Tree Model
# 

# In[128]:


# creating a bar plot for performance metrics for Decision Tree Model
accuracy = 0.80
precision = 0.78
recall = 0.84
f1_score = 0.81
DT_df = pd.DataFrame({'Metric':['Accuracy','Precision','Recall','F1 Score'],
                     'Score':[accuracy,precision,recall,f1_score]})
plt.figure(figsize=(8,6))
plt.bar(DT_df['Metric'], DT_df['Score'], color=['lightgreen','maroon','yellow','purple'])
plt.ylim(0,1.0)
plt.xlabel('Metric')
plt.ylabel('Score')
plt.title('performance metrics for Decision Tree Model in Forecasting Customer Churn')
plt.tight_layout()
plt.show()


# The Decision Tree model's performance is balanced, with similar precision, recall, and F1-score for both classes. The model demonstrates good predictive ability and is effectively classifying instances. However, further analysis and potential fine-tuning may be needed based on considerations.

# # Random Forest Classifier

# In[31]:


# importing RandomForestClassifier
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier()
# training the model
rf.fit(A_train,B_train)


# The RandomForestClassifier was imported class from scikit-learn's ensemble module and an instance of the Random Forest classifier rf was created. The Random Forest classifier (rf) trains using the training data where, A_train for features and B_train for target labels. The Random Forest algorithm creates an ensemble of decision trees and combines their predictions to make final classifications.

# In[32]:


# predicting the model
B_pred4 = rf.predict(A_test)


# In[102]:


# checking the accuracy
accuracy_score(B_test,B_pred4)


# In[103]:


# checking the precision score
precision_score(B_test,B_pred4)


# In[104]:


# checking the recall score
recall_score(B_test,B_pred4)


# In[106]:


# checking the f1 score
from sklearn.metrics import f1_score
f1_score(B_test,B_pred4)


# The Random Forest model demonstrates good performance with high precision, recall, and F1-score for both classes. It effectively classifies instances and strikes a balance between minimizing false positives and false negatives. The results suggest that the model is reliable and could be a suitable choice for the project.

# ## roc_auc_score for Random Forest Classifier

# In[205]:


# Predict probabilities for positive class(exited)
# Based on BROWNLEE, J., 2021. How to Use ROC Curves and Precision-Recall Curves for Classification in Python. [online]. MachineLearningMastery.com. Available from: https://machinelearningmastery.com/roc-curves-and-precision-recall-curves-for-classification-in-python/#:~:text=The%20AUC%20for%20the%20ROC,skill%20and%20perfect%20skill%20respectively.https://machinelearningmastery.com/roc-curves-and-precision-recall-curves-for-classification-in-python/#:~:text=The%20AUC%20for%20the%20ROC,skill%20and%20perfect%20skill%20respectively.
from sklearn.metrics import roc_curve, roc_auc_score
# Based on BROWNLEE, J., 2021. How to Use ROC Curves and Precision-Recall Curves for Classification in Python. [online]. MachineLearningMastery.com. Available from: https://machinelearningmastery.com/roc-curves-and-precision-recall-curves-for-classification-in-python/#:~:text=The%20AUC%20for%20the%20ROC,skill%20and%20perfect%20skill%20respectively.https://machinelearningmastery.com/roc-curves-and-precision-recall-curves-for-classification-in-python/#:~:text=The%20AUC%20for%20the%20ROC,skill%20and%20perfect%20skill%20respectively.
B_prob4 = rf.predict_proba(A_test)[:, 1]
# Calculating ROC curve
fpr, tpr, thresholds = roc_curve(B_test,B_prob4)
# Calculate AUC score
auc_score = roc_auc_score(B_test, B_prob4)
print("AUC:",auc_score)


# An AUC score of 0.9349 is quite high, indicating that the Random Forest classifier has a strong ability to differentiate between positive and negative instances. An AUC score above 0.9349 signifies that the model is performing exceptionally well in separating the classes. The Random Forest model's AUC score of 0.9349 indicates that it excels in its ability to discriminate between positive and negative instances. This suggests that the model has identified complex patterns and relationships within the data, making a strong predictive task.

# ## ROC Curve For Random Forest Classifier

# In[162]:


# Based on BROWNLEE, J., 2021. How to Use ROC Curves and Precision-Recall Curves for Classification in Python. [online]. MachineLearningMastery.com. Available from: https://machinelearningmastery.com/roc-curves-and-precision-recall-curves-for-classification-in-python/#:~:text=The%20AUC%20for%20the%20ROC,skill%20and%20perfect%20skill%20respectively.https://machinelearningmastery.com/roc-curves-and-precision-recall-curves-for-classification-in-python/#:~:text=The%20AUC%20for%20the%20ROC,skill%20and%20perfect%20skill%20respectively.
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label='ROC curve')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for Churn Prediction (Random Forest)')
plt.legend(loc='lower right')
plt.show()


#  The AUC measures the overall performance of the classifier. A higher AUC indicates better discrimination between classes, an AUC of 1 represents a perfect classifier. From the above roc curve, we infer that the curve is close to 1 indicating that it performs exceptionally well in distinguishing between positive and negative instances. 

# ## Confusion Matrix for evaluating the Random Forest Model

# In[108]:


#importing confusion matrix for evaluating the model using Random Forest Model
# Based on BHANDARI, A., 2023. Understanding &#038; Interpreting Confusion Matrices for Machine Learning (Updated 2023). [online]. Analytics Vidhya. Available from: https://www.analyticsvidhya.com/blog/2020/04/confusion-matrix-machine-learning/#:~:text=A%20confusion%20matrix%20is%20a%20performance%20evaluation%20tool%20in%20machine,false%20positives%2C%20and%20false%20negatives.
# Based on ML Handling Imbalanced Data with SMOTE and Near Miss Algorithm in Python, 2023. [online]. GeeksforGeeks. Available from: https://www.geeksforgeeks.org/ml-handling-imbalanced-data-with-smote-and-near-miss-algorithm-in-python/.
from sklearn.metrics import confusion_matrix, classification_report
# Calculating the confusion matrix
conf_matrix = confusion_matrix(B_test, B_pred4)
# printing the confusion matrix for Random Forest Model
print("confusion_matrix:\n", conf_matrix)
print("classification Report:]\n", classification_report(B_test, B_pred4))


# ## Confusion Matrix Heat Map for Random Forest Model

# In[289]:


confusion_matrix = np.array([[1323, 310],[155,1398]])
# Creating a heat map using seaborn
sns.set()
sns.heatmap(confusion_matrix, annot=True, cmap="viridis", fmt=".2f",
            xticklabels=["Non-Churned", "Churned"],
            yticklabels=["Non-Churned", "Churned"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Random Forest Model's Confusion Matrix")
plt.show()


# The Model correctly predicted 1323 instances as class 0, True Negatives. It incorrectly predicted 310 instances as class 1 when they were actually class 0 False Positives. It incorrectly predicted 155 instances as class 0 when they were actually class 1, False Negatives. It correctly predicted 1398 instances as class 1, True Positives.

# ## Bar plot for performance metrics for Random Forest Model

# In[294]:


#creating a bar plot for performance metrics for Random Forest Model
accuracy = 0.85
precision = 0.81
recall = 0.90
f1_score = 0.85
RF_df = pd.DataFrame({'Metric':['Accuracy','Precision','Recall','F1 Score'],
                     'Score':[accuracy,precision,recall,f1_score]})
plt.figure(figsize=(8,6))
plt.bar(RF_df['Metric'], RF_df['Score'], color=['brown','gray','green','pink'])
plt.ylim(0,1.0)
plt.xlabel('Metric')
plt.ylabel('Score')
plt.title('performance metrics for Random Forest Model in Forecasting Customer Churn')
plt.tight_layout()
plt.show()


# The Random Forest model demonstrates good performance with high precision, recall, and F1-score for both classes. It effectively classifies instances and strikes a balance between minimizing false positives and false negatives. The results suggest that the model is reliable and could be a suitable choice for the project.

# # Gradient Boosting classifier

# In[114]:


# importing GradientBoostingClassifier
from sklearn.ensemble import GradientBoostingClassifier
gb = GradientBoostingClassifier()
# training the model
gb.fit(A_train,B_train)


# The GradientBoostingClassifier class was imported from scikit-learn's ensemble module and  gb, an instance of the Gradient Boosting classifier was created. The (gb) trains the Gradient Boosting classifier using the training data, A_train for features and B_train for target labels. Gradient Boosting is an ensemble learning method that combines multiple weak learners, usually decision trees to create a strong predictive model.

# In[115]:


# predicting the model
B_pred5 = gb.predict(A_test)


# In[116]:


# checking the accuracy
accuracy_score(B_test,B_pred5)


# In[117]:


# checking the precision score
precision_score(B_test,B_pred5)


# In[118]:


# checking the recall score
recall_score(B_test,B_pred5)


# In[120]:


# checking the f1 score
from sklearn.metrics import f1_score
f1_score(B_test,B_pred5)


# The Gradient Boosting model demonstrates balanced performance with similar precision, recall, and F1-score for both classes. It effectively classifies instances and strikes a balance between minimizing false positives and false negatives. 

# ## Confusion Matrix for evaluating the Gradient Boosting classifier

# In[121]:


#importing confusion matrix for evaluating the model using Gradient Boosting classifier
# Based on BHANDARI, A., 2023. Understanding &#038; Interpreting Confusion Matrices for Machine Learning (Updated 2023). [online]. Analytics Vidhya. Available from: https://www.analyticsvidhya.com/blog/2020/04/confusion-matrix-machine-learning/#:~:text=A%20confusion%20matrix%20is%20a%20performance%20evaluation%20tool%20in%20machine,false%20positives%2C%20and%20false%20negatives.
# Based on ML Handling Imbalanced Data with SMOTE and Near Miss Algorithm in Python, 2023. [online]. GeeksforGeeks. Available from: https://www.geeksforgeeks.org/ml-handling-imbalanced-data-with-smote-and-near-miss-algorithm-in-python/.
from sklearn.metrics import confusion_matrix, classification_report
# Calculating the confusion matrix
conf_matrix = confusion_matrix(B_test, B_pred5)
# printing the confusion matrix for Gradient Boosting classifier
print("confusion_matrix:\n", conf_matrix)
print("classification Report:]\n", classification_report(B_test, B_pred5))


# ## Confusion Matrix Heat Map For Gradient Boosting Classifier

# In[256]:


confusion_matrix = np.array([[1374, 259],[249,1304]])
# Creating a heat map using seaborn
sns.set()
sns.heatmap(confusion_matrix, annot=True, cmap="Blues", fmt=".2f",
            xticklabels=["Non-Churned", "Churned"],
            yticklabels=["Non-Churned", "Churned"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Gradient Boosting Model's Confusion Matrix")
plt.show()


# The model correctly predicted 1374 instances as class 0, True Negatives. It incorrectly predicted 259 instances as class 1 when they were actually class 0, False Positives. It incorrectly predicted 249 instances as class 0 when they were actually class 1, False Negatives. It correctly predicted 1304 instances as class 1, True Positives.

# ## Bar plot for performance metrics for Gradient Boosting classifier

# In[292]:


#creating a bar plot for performance metrics for Decision Tree Model
accuracy = 0.84
precision = 0.83
recall = 0.83
f1_score = 0.83
GB_df = pd.DataFrame({'Metric':['Accuracy','Precision','Recall','F1 Score'],
                     'Score':[accuracy,precision,recall,f1_score]})
plt.figure(figsize=(8,6))
plt.bar(GB_df['Metric'], GB_df['Score'], color=['yellow','lightgreen','brown','pink'])
plt.ylim(0,1.0)
plt.xlabel('Metric')
plt.ylabel('Score')
plt.title('performance metrics for Gradient Boosting Model in Forecasting Customer Churn')
plt.tight_layout()
plt.show()


# The Gradient Boosting model demonstrates balanced performance with similar precision, recall, and F1-score for both classes. It effectively classifies instances and strikes a balance between minimizing false positives and false negatives. The results suggest that the model is reliable and could be suitable for our project.

# ## roc_auc_score for Gradient Boosting Classifier

# In[206]:


# Predict probabilities for positive class(exited)
from sklearn.metrics import roc_curve, roc_auc_score
# Based on BROWNLEE, J., 2021. How to Use ROC Curves and Precision-Recall Curves for Classification in Python. [online]. MachineLearningMastery.com. Available from: https://machinelearningmastery.com/roc-curves-and-precision-recall-curves-for-classification-in-python/#:~:text=The%20AUC%20for%20the%20ROC,skill%20and%20perfect%20skill%20respectively.https://machinelearningmastery.com/roc-curves-and-precision-recall-curves-for-classification-in-python/#:~:text=The%20AUC%20for%20the%20ROC,skill%20and%20perfect%20skill%20respectively.
B_prob5 = gb.predict_proba(A_test)[:, 1]
# Calculating ROC curve
fpr, tpr, thresholds = roc_curve(B_test,B_prob5)
# Calculate AUC score
auc_score = roc_auc_score(B_test, B_prob5)
print("AUC:",auc_score)


# An AUC score of 0.9169 is quite high, indicating that the Gradient Boosting classifier has a strong ability to differentiate between positive and negative instances. It is performing exceptionally well in separating the classes. It has identified meaningful patterns from the data and can effectively rank instances. This suggests that the model has identified complex patterns and relationships within the data, making a strong prediction.

# In[293]:


# creating pandas data frame for storing accuracy of all the models
new_churn_data=pd.DataFrame({'Models':['LR','SVC','KNN','DT','RF','GB'],
                             'ACC':[accuracy_score(B_test, B_pred),
                                    accuracy_score(B_test, B_pred1),
                                    accuracy_score(B_test, B_pred2),
                                    accuracy_score(B_test, B_pred3),
                                    accuracy_score(B_test, B_pred4),
                                    accuracy_score(B_test, B_pred5)]})


# Storing the accuaracy scores of all the models in a data frame

# In[174]:


new_churn_data


# In[136]:


# visualising the models
import seaborn as sns

sns.barplot(new_churn_data['Models'], new_churn_data['ACC'])


# Visual Presentation of the accuracy scores of all the models in which Random Forest is the best performer with 85% accuracy which is slightly higher than the GB model which is 83%.

# ## ROC AUC Score for all the models

# In[301]:


auc_scores = [roc_auc_score(B_test,B_prob),
              roc_auc_score(B_test,B_prob1),
              roc_auc_score(B_test,B_prob2),
              roc_auc_score(B_test,B_prob3),
              roc_auc_score(B_test,B_prob4),
              roc_auc_score(B_test,B_prob5)]
new_churn_data=pd.DataFrame({'Models':['LR','SVC','KNN','DT','RF','GB'],
                             'ROC_AUC_Score':[roc_auc_score(B_test, B_prob),
              roc_auc_score(B_test,B_prob1),
              roc_auc_score(B_test,B_prob2),
              roc_auc_score(B_test,B_prob3),
              roc_auc_score(B_test,B_prob4),
              roc_auc_score(B_test,B_prob5)]})


# In[302]:


print(new_churn_data)


# In[303]:


sns.barplot(new_churn_data['Models'], new_churn_data['ROC_AUC_Score'])


# Visual presentation of ROC_AUC score of all the models. This depict that Random Forest has achieved a high ROC_AUC score of 0.93.

# In[305]:




# storing the performance metrics 
metrics_data = {
    'Model': ['Logistic Regression', 'Support Vector', 'K-Nearest Neighbors', 'Decision Tree', 'Random Forest', 'Gradient Boosting'],
    'ROC-AUC': [0.862695, 0.878976, 0.803531, 0.809057, 0.934979, 0.916948],
    'Precision': [0.754276, 0.784713, 0.658284, 0.781362, 0.818501, 0.834293],
    'Recall': [0.779781, 0.793303, 0.859626, 0.842240, 0.900193, 0.839665],  
    'F1-Score': [0.774792, 0.788984, 0.745601, 0.810660, 0.857405, 0.836970],  
    'Accuracy': [0.779033, 0.834275, 0.821092, 0.791274, 0.855932, 0.835217]
}

# Create a DataFrame 
metrics_df = pd.DataFrame(metrics_data)

# Display the DataFrame
print(metrics_df)


# Based on the evaluation of multiple performance metrics, including ROC-AUC, precision, recall, F1-score, and accuracy, we can draw the following conclusions:
# Random Forest achieved the highest ROC-AUC score of 0.93, indicating strong discriminatory power and effective separation of classes. It also demonstrated balanced precision of 0.81 and recall of 0.90, resulting in a high F1-score of 0.85 and accuracy with 0.85. While Gradient Boosting and Support Vector Machine performed well, Random Forest consistently outperformed other models in terms of multiple metrics, making it a strong qualifier. K-Nearest Neighbors showed high recall of 0.85 but comparatively lower precision and F1-score, suggesting potential room for improvement. Logistic Regression and Decision Tree exhibited reasonable performance but were outperformed by Random Forest in various metrics.
# 

# In[23]:


# Saving the model after feature scaling
A_new=scaler.fit_transform(A_new)
rf.fit(A_new,B_new)


# In[24]:


# Storing the model in joblib
import joblib
joblib.dump(rf,'customer_churn_forecast_model')


# The joblib library was imported, which provides functions for saving and loading Python objects, including machine learning models. The joblib.dump() function was used to save the Random Forest model (rf) to a file named "customer_churn_forecast_model". The model will be saved and stored in this file.

# In[25]:


# to use this model
model = joblib.load('customer_churn_forecast_model')


# In[45]:


churn_data.columns


# In[110]:


model.predict([[619,42,2,0.0,0,0,0,101348.88,0,0,0]])


# The saved model is predicted with inputting values for each column in the data and the saved model predicted 1 which means that the customer will leave the bank.

# The model predicted that the customer will leave the bank

# In[26]:


model = joblib.load("customer_churn_forecast_model")


# # Bayesian Optimization with Tree-structured Parzen Estimator
# ### To fine-tune the hyperparameters of the Random Forest model.

# In[52]:


pip install hyperopt


# In[33]:


import numpy as np
import joblib
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from hyperopt import fmin, tpe, hp
# Based on Bergstra, J., Yamins, D. and Cox, D.D., 2013, June. Hyperopt: A python library for optimizing the hyperparameters of machine learning algorithms. In Proceedings of the 12th Python in science conference (Vol. 13, p. 20).
# Based on Agrawal, T., 2021. Hyperparameter optimization in machine learning: make your machine learning and deep learning models more efficient. New York, NY, USA:: Apress.
# Based on Guest_blog, 2022. Alternative Hyperparameter Optimization Technique You need to Know &#8211; Hyperopt. [online]. Analytics Vidhya. Available from: https://www.analyticsvidhya.com/blog/2020/09/alternative-hyperparameter-optimization-technique-you-need-to-know-hyperopt/.
# Based on MANSUKHANI, S., 2023. HyperOpt: Bayesian Hyperparameter Optimization. [online]. Available from: https://domino.ai/blog/hyperopt-bayesian-hyperparameter-optimization.
# Based on Rendyk, 2021. Bayesian Optimization: bayes_opt or hyperopt. [online]. Analytics Vidhya. Available from: https://www.analyticsvidhya.com/blog/2021/05/bayesian-optimization-bayes_opt-or-hyperopt/.
# Loading my already saved RF model using joblib
model = joblib.load("customer_churn_forecast_model")

# Defining the objective function for Bayesian Optimization
def objective(params):
    rf = RandomForestClassifier(n_estimators=int(params['n_estimators']), max_depth=int(params['max_depth']),
        min_samples_split=int(params['min_samples_split']), random_state=42)
    rf.fit(A_train, B_train)
    y_pred = rf.predict(A_test)
    accuracy = accuracy_score(B_test, B_pred4)
    return -accuracy  # Negative because Hyperopt minimizes

# Defining the search space for hyperparameters
space = {'n_estimators': hp.quniform('n_estimators', 50, 300, 50),
    'max_depth': hp.quniform('max_depth', 10, 100, 10),
    'min_samples_split': hp.quniform('min_samples_split', 2, 10, 1),}

# Performing Bayesian Optimization using TPE
best = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=100)

# Extracting the best hyperparameters found by Bayesian Optimization
best_n_estimators = int(best['n_estimators'])
best_max_depth = int(best['max_depth'])
best_min_samples_split = int(best['min_samples_split'])

# Training and evaluating the RF model with the best hyperparameters
best_rf = RandomForestClassifier(n_estimators=best_n_estimators,max_depth=best_max_depth,
                                 min_samples_split=best_min_samples_split,random_state=42)
best_rf.fit(A_train, B_train)
y_pred = best_rf.predict(A_test)
best_accuracy = accuracy_score(B_test, B_pred4)

print("Best Hyperparameters:", best)
print("Best Accuracy:", best_accuracy)


#  100 trials were executed, each taking approximately 8 minutes.The "best accuracy: -0.8535" signifies that the optimization aimed to minimize the negative of a metric, in this case, the objective function representing accuracy."Best Hyperparameters: {'max_depth': 40.0, 'min_samples_split': 2.0, 'n_estimators': 50.0}". These optimal hyperparameters were determined through Bayesian optimization, where a max depth of 40, minimum samples split of 2, and 50 estimators yielded the best accuracy.

# ## The python data analysis for this project was performed based on the following tutorials, research papers and websites
# 

# In[37]:


# Based on Pandas Get Dummies – pd.get_dummies() | Data Independent, 2023. [online]. Data Independent. Available from: https://dataindependent.com/pandas/pandas-get-dummies-pd-get_dummies/.
# Based on BHANDARI, A., 2023. Understanding &#038; Interpreting Confusion Matrices for Machine Learning (Updated 2023). [online]. Analytics Vidhya. Available from: https://www.analyticsvidhya.com/blog/2020/04/confusion-matrix-machine-learning/#:~:text=A%20confusion%20matrix%20is%20a%20performance%20evaluation%20tool%20in%20machine,false%20positives%2C%20and%20false%20negatives.
# Based on ML Handling Imbalanced Data with SMOTE and Near Miss Algorithm in Python, 2023. [online]. GeeksforGeeks. Available from: https://www.geeksforgeeks.org/ml-handling-imbalanced-data-with-smote-and-near-miss-algorithm-in-python/.
# Based on Data Analysis with Python, 2023. [online]. GeeksforGeeks. Available from: https://www.geeksforgeeks.org/data-analysis-with-python/.
# Based on Guest_blog, 2022. Alternative Hyperparameter Optimization Technique You need to Know &#8211; Hyperopt. [online]. Analytics Vidhya. Available from: https://www.analyticsvidhya.com/blog/2020/09/alternative-hyperparameter-optimization-technique-you-need-to-know-hyperopt/.
# Based on MANSUKHANI, S., 2023. HyperOpt: Bayesian Hyperparameter Optimization. [online]. Available from: https://domino.ai/blog/hyperopt-bayesian-hyperparameter-optimization.
# Based on Rendyk, 2021. Bayesian Optimization: bayes_opt or hyperopt. [online]. Analytics Vidhya. Available from: https://www.analyticsvidhya.com/blog/2021/05/bayesian-optimization-bayes_opt-or-hyperopt/.
# Based on Bergstra, J., Yamins, D. and Cox, D.D., 2013, June. Hyperopt: A python library for optimizing the hyperparameters of machine learning algorithms. In Proceedings of the 12th Python in science conference (Vol. 13, p. 20).
# Based on Agrawal, T., 2021. Hyperparameter optimization in machine learning: make your machine learning and deep learning models more efficient. New York, NY, USA:: Apress.

