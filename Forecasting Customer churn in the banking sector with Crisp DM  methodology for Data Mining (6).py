#!/usr/bin/env python
# coding: utf-8

# In[1]:


#importing necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


#importing the dataset
churn_data = pd.read_csv('/Users/jeevithaselvaraj/Downloads/Churn_Modelling.csv')


# In[3]:


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


# In[9]:


# checking the null values
churn_data.isnull()


# The above .isnull() function gives the boolean value of the dataset

# In[10]:


churn_data.isnull().sum()


# In the above code we are performing sum() of true values to get the null values for both categorical and numerical values. If any missing values are there we should drop them. but here it is not.
# 

# In[11]:


churn_data.describe()


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


# unique() function is used to find the unique values in a series.
# 
# 

# In[13]:


# Encoding categorical data
# one hot/ dummy encoding
# https://dataindependent.com/pandas/pandas-get-dummies-pd-get_dummies/
churn_data = pd.get_dummies(churn_data, drop_first = True)


# In[86]:


churn_data


# In[87]:


churn_data.Exited.plot.hist()


# In[19]:


churn_data['Exited'].value_counts()


# Here value counts() function is used to find the distribution of targeted variable. The output will be sorted in the descending order with the firat element being the frequently occurring value. 0 means the customer is not leaving the bank and 1 means the customer is leaving the bank.
# 

# In[120]:


# dropping the target variable and storing the data with independent variables in a new data frame 
churn_data1=churn_data.drop(columns='Exited')


# ## Correlation Matrix

# In[121]:


# Creating a correlation matrix between the independent and response variable
churn_data1.corrwith(churn_data['Exited']).plot.bar(figsize=(15,7), title='correlation With the exited variable', rot = 45,grid = True)


# ## Heat Map

# In[118]:


# Creating a heat map
corr=churn_data.corr()
plt.figure(figsize=(15,7))
sns.heatmap(corr,annot=True)


# ### Count plot for identifying the distribution of the target variable

# In[11]:


# Visualising
sns.countplot(churn_data['Exited'])


# From this count plot we can find that the target variable is unevenly distributed and 
# we can see that the data is imbalanced.

# ## Feature Selection and Data Splitting

# In[14]:


#importing train_test_split 
from sklearn.model_selection import train_test_split
# Storing the independent variable in A
A = churn_data.drop(columns='Exited')
# storing the target variable in B
B = churn_data['Exited']


# Feature selection is performed to reduce the dimensionality and for easier interpretation. 

# In[15]:


# Splitting the dataset
A_train, A_test, B_train, B_test = train_test_split(A, B, test_size=0.2, random_state=42, stratify=B)


# Data split is carried to enhance the model's performance.Here stratify is used to enforce the same class balance on the train and test data as the original data.

# In[8]:


A_train.shape


# In[9]:


A_test.shape


# # Feature Scaling 

# In[16]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
A_train = scaler.fit_transform(A_train)
A_test = scaler.transform(A_test)


# It is vital to perform Feature scaling where all the variables are brought to a similar scale, as the features with high value range dominates in calculating the distances between data.

# In[128]:


A_train


# # Logistic regression (LR)
# ### Analysis before applying SMOTE

# In[129]:


from sklearn.linear_model import LogisticRegression
# Creating instance for this model
logis = LogisticRegression()
# train the model
logis.fit(A_train,B_train)
# Predict the model
B_pred = logis.predict(A_test)


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


# # Applying Oversampling with SMOTE
# 

# In[17]:


# Applying Oversampling with SMOTE
# importing SMOTE
# Based on ML Handling Imbalanced Data with SMOTE and Near Miss Algorithm in Python, 2023. [online]. GeeksforGeeks. Available from: https://www.geeksforgeeks.org/ml-handling-imbalanced-data-with-smote-and-near-miss-algorithm-in-python/.
from imblearn.over_sampling import SMOTE
# Assigning in new variable
A_new,B_new = SMOTE().fit_resample(A,B)


# In[135]:


B_new.value_counts()


# Now after applying SMOTE there is an even distribution in majority and minority class 

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

# In[73]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
A_train = scaler.fit_transform(A_train)
A_test = scaler.transform(A_test)


# # Logistic Regression(LR) 
# ### After appying SMOTE

# In[75]:


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


# ## Support Vector Classifier (SVC)

# In[159]:


# importing svm
from sklearn import svm
# creating instance
svm = svm.SVC(probability = True)
# training the model
svm.fit(A_train,B_train)


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


# ## KNearest Neighbor

# In[79]:


# importing KNeighbors classifier
from sklearn.neighbors import KNeighborsClassifier
# creating instance
knn = KNeighborsClassifier()
# training the model
knn.fit(A_train,B_train)


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


# ## Confusion Matrix for evaluating KNearest Neighbor

# In[85]:


#importing confusion matrix for evaluating the model using KNeighborsClassifier
# Based on BHANDARI, A., 2023. Understanding &#038; Interpreting Confusion Matrices for Machine Learning (Updated 2023). [online]. Analytics Vidhya. Available from: https://www.analyticsvidhya.com/blog/2020/04/confusion-matrix-machine-learning/#:~:text=A%20confusion%20matrix%20is%20a%20performance%20evaluation%20tool%20in%20machine,false%20positives%2C%20and%20false%20negatives.
# Based on ML Handling Imbalanced Data with SMOTE and Near Miss Algorithm in Python, 2023. [online]. GeeksforGeeks. Available from: https://www.geeksforgeeks.org/ml-handling-imbalanced-data-with-smote-and-near-miss-algorithm-in-python/.
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


# # Decision Tree Model

# In[89]:


# importing DecisionTreeClassifier
from sklearn.tree import DecisionTreeClassifier
# creating instance
dt = DecisionTreeClassifier()
# training the model
dt.fit(A_train, B_train)


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


# ## Confusion Matrix for evaluating the Decision Tree Model

# In[107]:


#importing confusion matrix for evaluating the model using Decision Tree Model
# Based on BHANDARI, A., 2023. Understanding &#038; Interpreting Confusion Matrices for Machine Learning (Updated 2023). [online]. Analytics Vidhya. Available from: https://www.analyticsvidhya.com/blog/2020/04/confusion-matrix-machine-learning/#:~:text=A%20confusion%20matrix%20is%20a%20performance%20evaluation%20tool%20in%20machine,false%20positives%2C%20and%20false%20negatives.
# Based on ML Handling Imbalanced Data with SMOTE and Near Miss Algorithm in Python, 2023. [online]. GeeksforGeeks. Available from: https://www.geeksforgeeks.org/ml-handling-imbalanced-data-with-smote-and-near-miss-algorithm-in-python/.
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


# # Random Forest Classifier

# In[100]:


# importing RandomForestClassifier
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier()
# training the model
rf.fit(A_train,B_train)


# In[101]:


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


# # Gradient Boosting classifier

# In[114]:


# importing GradientBoostingClassifier
from sklearn.ensemble import GradientBoostingClassifier
gb = GradientBoostingClassifier()
# training the model
gb.fit(A_train,B_train)


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


# ## ROC Curve for Gradient Boosting Classifier

# In[152]:


#Based on BROWNLEE, J., 2021. How to Use ROC Curves and Precision-Recall Curves for Classification in Python. [online]. MachineLearningMastery.com. Available from: https://machinelearningmastery.com/roc-curves-and-precision-recall-curves-for-classification-in-python/#:~:text=The%20AUC%20for%20the%20ROC,skill%20and%20perfect%20skill%20respectively.https://machinelearningmastery.com/roc-curves-and-precision-recall-curves-for-classification-in-python/#:~:text=The%20AUC%20for%20the%20ROC,skill%20and%20perfect%20skill%20respectively.
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='brown', lw=2, label='ROC curve')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for Churn Prediction (Gradient Boosting Classifier)')
plt.legend(loc='lower right')
plt.show()


# In[171]:


models = ['LR','SVC','KNN','DT','RF','GB']
accuracy_scores=[79,79,71,80,85,84]
precision_scores=[76,79,65,78,81,83]
recall_scores=[77,89,85,84,90,83]
f1_scores=[77,78,74,81,85,83]


# In[179]:



plt.figure(figsize=(10,6))
plt.bar(models,accuracy_scores, color='purple',label='Accuracy')
plt.bar(models,precision_scores, color='brown',label='Precision')
plt.bar(models,recall_scores, color='gray',label='Recall')
plt.bar(models,f1_scores, color='lightblue',label='f1')
plt.xlabel('Models')
plt.ylabel('Scores')
plt.title("Forecasting Customer Churn")
plt.legend()
plt.show()


# In[293]:


# creating pandas data frame for storing accuracy of all the models
new_churn_data=pd.DataFrame({'Models':['LR','SVC','KNN','DT','RF','GB'],
                             'ACC':[accuracy_score(B_test, B_pred),
                                    accuracy_score(B_test, B_pred1),
                                    accuracy_score(B_test, B_pred2),
                                    accuracy_score(B_test, B_pred3),
                                    accuracy_score(B_test, B_pred4),
                                    accuracy_score(B_test, B_pred5)]})


# In[174]:


new_churn_data


# In[136]:


# visualising the models
import seaborn as sns

sns.barplot(new_churn_data['Models'], new_churn_data['ACC'])


# In[187]:


# creating pandas data frame for storing accuracy of all the models
new_churn_data=pd.DataFrame({'Models':['LR','SVC','KNN','DT','RF','GB'],
                             'PRE':[precision_score(B_test, B_pred),
                                    precision_score(B_test, B_pred1),
                                    precision_score(B_test, B_pred2),
                                    precision_score(B_test, B_pred3),
                                    precision_score(B_test, B_pred4),
                                    precision_score(B_test, B_pred5)]})


# In[184]:


new_churn_data


# In[185]:


# visualising
sns.barplot(new_churn_data['Models'], new_churn_data['PRE'])


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


# In[179]:


# Saving the model after feature scaling
A_new=scaler.fit_transform(A_new)
rf.fit(A_new,B_new)


# In[180]:


# Storing the model in joblib
import joblib
joblib.dump(rf,'customer_churn_forecast_model')


# In[181]:


# to use this model
model = joblib.load('customer_churn_forecast_model')


# In[107]:


churn_data.columns


# In[110]:


model.predict([[619,42,2,0.0,0,0,0,101348.88,0,0,0]])


# The model predicted that the customer will leave the bank
