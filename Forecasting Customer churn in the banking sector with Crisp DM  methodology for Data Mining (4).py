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

# In[9]:


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

# In[3]:


# Visualising
sns.countplot(churn_data['Exited'])


# From this count plot we can find that the target variable is unevenly distributed and 
# we can see that the data is imbalanced.

# ## Feature Selection and Data Splitting

# In[20]:


#importing train_test_split 
from sklearn.model_selection import train_test_split
# Storing the independent variable in A
A = churn_data.drop(columns='Exited')
# storing the target variable in B
B = churn_data['Exited']


# Feature selection is performed to reduce the dimensionality and for easier interpretation. 

# In[124]:


# Splitting the dataset
A_train, A_test, B_train, B_test = train_test_split(A, B, test_size=0.2, random_state=42, stratify=B)


# Data split is carried to enhance the model's performance.Here stratify is used to enforce the same class balance on the train and test data as the original data.

# In[125]:


A_train.shape


# In[126]:


A_test.shape


# # Feature Scaling before SMOTE

# In[127]:


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

# In[134]:


# Applying Oversampling with SMOTE
# importing SMOTE
# Based on ML Handling Imbalanced Data with SMOTE and Near Miss Algorithm in Python, 2023. [online]. GeeksforGeeks. Available from: https://www.geeksforgeeks.org/ml-handling-imbalanced-data-with-smote-and-near-miss-algorithm-in-python/.
from imblearn.over_sampling import SMOTE
# Assigning in new variable
A_new,B_new = SMOTE().fit_resample(A,B)


# In[135]:


B_new.value_counts()


# Now after applying SMOTE there is an even distribution in majority and minority class 

# In[136]:


# Splitting the dataset with new dataset with SMOTE
A_train, A_test, B_train, B_test = train_test_split(A_new, B_new, test_size=0.2, random_state=42)


# # Feature Scaling after SMOTE

# In[137]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
A_train = scaler.fit_transform(A_train)
A_test = scaler.transform(A_test)


# # Logistic Regression(LR) 
# ### After appying SMOTE

# In[138]:


from sklearn.linear_model import LogisticRegression
# Creating instance for this model
logis = LogisticRegression()
# train the model
logis.fit(A_train,B_train)
# Predict the model
B_pred = logis.predict(A_test)


# In[139]:


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


# ## Bar plot for performance metrics for Logistic Regression

# In[195]:


# creating a bar plot for performance metrics for Logistic Regression
accuracy = 0.77
precision = 0.76
recall = 0.77
f1_score = 0.77
LR_df = pd.DataFrame({'Metric':['Accuracy','Precision','Recall','F1 Score'],
                     'Score':[accuracy,precision,recall,f1_score]})
plt.figure(figsize=(8,6))
plt.bar(LR_df['Metric'], LR_df['Score'], color=['blue','green','orange','purple'])
plt.ylim(0,1.0)
plt.xlabel('Metric')
plt.ylabel('Score')
plt.title('performance metrics for Logistic Regression in Forecasting Customer Churn')
plt.tight_layout()
plt.show()


# # Support Vector Classifier (SVC)

# In[143]:


# importing svm
from sklearn import svm
# creating instance
svm = svm.SVC()
# training the model
svm.fit(A_train,B_train)


# In[144]:


# predicting the model
B_pred1 = svm.predict(A_test)


# In[145]:


# checking the accuracy
accuracy_score(B_test,B_pred1)


# In[146]:


# checking the precision score
precision_score(B_test,B_pred1)


# In[147]:


# checking the recall score
recall_score(B_test,B_pred1)


# In[148]:


# checking the f1 score
f1_score(B_test,B_pred1)


# In[199]:


# creating a bar plot for performance metrics for support vector machine
accuracy = 0.83
precision = 0.82
recall = 0.83
f1_score = 0.83
LR_df = pd.DataFrame({'Metric':['Accuracy','Precision','Recall','F1 Score'],
                     'Score':[accuracy,precision,recall,f1_score]})
plt.figure(figsize=(8,6))
plt.bar(LR_df['Metric'], LR_df['Score'], color=['brown','pink','red','yellow'])
plt.ylim(0,1.0)
plt.xlabel('Metric')
plt.ylabel('Score')
plt.title('performance metrics for Support Vector Classifier in Forecasting Customer Churn')
plt.tight_layout()
plt.show()


# # KNeighbors Classifier

# In[149]:


# importing KNeighbors classifier
from sklearn.neighbors import KNeighborsClassifier
# creating instance
knn = KNeighborsClassifier()
# training the model
knn.fit(A_train,B_train)


# In[150]:


# Predicting the model
B_pred2 = knn.predict(A_test)


# In[151]:


# checking the accuracy
accuracy_score(B_test,B_pred2)


# In[152]:


# checking the precision score
precision_score(B_test,B_pred2)


# In[153]:


# checking the recall score
recall_score(B_test,B_pred2)


# In[154]:


# checking the f1 score
f1_score(B_test,B_pred2)


# ## Scatter plot with Color Coding for KNN

# In[206]:


B_pred2 = knn.predict(A_test)
plt.scatter(A_test[B_pred2 == 1][:,0],A_test[B_pred2 == 1][:,1],color='red', marker='o', label='Predicted Exited')
plt.scatter(A_test[B_pred2 == 0][:,0],A_test[B_pred2 == 0][:,1],color='purple', marker='o', label='Predicted Not Exited')
plt.scatter(A_test[B_test == 1][:,0],A_test[B_test == 1][:,1],color='orange', marker='x', label='Actual Exited')
plt.scatter(A_test[B_test == 0][:,0],A_test[B_test == 0][:,1],color='gray', marker='x', label='Actual Not Exited')
plt.xlabel('Predicted')
plt.ylabel('Actuals')
plt.title('Scatter plot with color coding for KNN Forecasting Customer Churn')
plt.legend()
plt.grid()
plt.show()


# In[ ]:





# # Decision Tree Model

# In[155]:


# importing DecisionTreeClassifier
from sklearn.tree import DecisionTreeClassifier
# creating instance
dt = DecisionTreeClassifier()
# training the model
dt.fit(A_train, B_train)


# In[156]:


# predicting the model
B_pred3 = dt.predict(A_test)


# In[157]:


# checking the accuracy
accuracy_score(B_test,B_pred3)


# In[158]:


# checking the precision score
precision_score(B_test,B_pred3)


# In[159]:


# checking the recall score
recall_score(B_test,B_pred3)


# In[160]:


# checking the f1 score
f1_score(B_test,B_pred3)


# # Random Forest Classifier

# In[161]:


# importing RandomForestClassifier
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier()
# training the model
rf.fit(A_train,B_train)


# In[162]:


# predicting the model
B_pred4 = rf.predict(A_test)


# In[163]:


# checking the accuracy
accuracy_score(B_test,B_pred4)


# In[164]:


# checking the precision score
precision_score(B_test,B_pred4)


# In[165]:


# checking the recall score
recall_score(B_test,B_pred4)


# In[166]:


# checking the f1 score
f1_score(B_test,B_pred4)


# # Gradient Boosting classifier

# In[167]:


# importing GradientBoostingClassifier
from sklearn.ensemble import GradientBoostingClassifier
gb = GradientBoostingClassifier()
# training the model
gb.fit(A_train,B_train)


# In[168]:


# predicting the model
B_pred5 = gb.predict(A_test)


# In[169]:


# checking the accuracy
accuracy_score(B_test,B_pred5)


# In[170]:


# checking the precision score
precision_score(B_test,B_pred5)


# In[171]:


# checking the recall score
recall_score(B_test,B_pred5)


# In[172]:


# checking the f1 score
f1_score(B_test,B_pred5)


# In[173]:


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


# In[175]:


# visualising the models
import seaborn as sns
sns.barplot(new_churn_data['Models'], new_churn_data['ACC'])


# In[176]:


# creating pandas data frame for storing accuracy of all the models
new_churn_data=pd.DataFrame({'Models':['LR','SVC','KNN','DT','RF','GB'],
                             'PRE':[precision_score(B_test, B_pred),
                                    precision_score(B_test, B_pred1),
                                    precision_score(B_test, B_pred2),
                                    precision_score(B_test, B_pred3),
                                    precision_score(B_test, B_pred4),
                                    precision_score(B_test, B_pred5)]})


# In[177]:


new_churn_data


# In[178]:


# visualising
sns.barplot(new_churn_data['Models'], new_churn_data['PRE'])


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
