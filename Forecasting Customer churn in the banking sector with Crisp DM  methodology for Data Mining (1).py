#!/usr/bin/env python
# coding: utf-8

# In[80]:


#importing necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[81]:


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
#https://www.miamioh.edu/cads/students/coding-tutorials/python/data-cleaning/index.html
churn_data.isnull()


# The above .isnull() function gives the boolean value of the dataset

# In[10]:


#https://www.miamioh.edu/cads/students/coding-tutorials/python/data-cleaning/index.html
churn_data.isnull().sum()


# In the above code we are performing sum() of true values to get the null values for both categorical and numerical values. If any missing values are there we should drop them. but here it is not.
# 

# In[11]:


churn_data.describe()


# In[82]:


# Dropping inappropriate columns that is not required for this analysis
churn_data.columns


# In[83]:


#https://www.listendata.com/2019/06/pandas-drop-columns-from-dataframe.html
#(“How to Drop One or Multiple Columns from Pandas Dataframe” n.d.)
churn_data = churn_data.drop(['RowNumber', 'CustomerId', 'Surname'],axis=1)


# The irrelevant columns are dropped to reduce the memory usage and for the ease of analysis purposes

# In[15]:


churn_data.info()


# After dropping the irrelevant attributes there is a significant decrease in the memory usage from 1.1+ MB to 859.5+ KB, the former is the one, before dropping the irrelevant columns.

# In[16]:


# Encoding ctaegorical data
# https://www.educative.io/answers/what-is-the-unique-function-in-pandas
churn_data['Geography'].unique()


# In[84]:


churn_data['Gender'].unique()


# unique() function is used to find the unique values in a series.
# 
# 

# In[85]:


# Encoding ctaegorical data
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

# In[107]:


churn_data1=churn_data.drop(columns='Exited')


# In[111]:


# Creating a correlation matrix between the independent an dresponse variable
churn_data1.corrwith(churn_data['Exited']).plot.bar(figsize=(17,9), title='correlation With the exited variable', rot = 45,grid = True)


# In[116]:


# Creating a heat map
corr=churn_data.corr()
plt.figure(figsize=(17,9))
sns.heatmap(corr,annot=True)


# In[3]:


# Visualising
sns.countplot(churn_data['Exited'])


# In[ ]:


From this count plot we can find that the target variable is unevenly distributed and we can see that the data is imbalanced.


# In[5]:


#importing train_test_split 
# https://www.freecodecamp.org/news/what-is-stratified-random-sampling-definition-and-python-example/#:~:text=In%20machine%20learning%2C%20stratified%20sampling,here%20to%20download%20the%20dataset.
from sklearn.model_selection import train_test_split
# Storing the independent variable in A
A = churn_data.drop(columns='Exited')
# storing the target variable in churn_data2
B = churn_data['Exited']


# In[120]:


# Splitting the dataset
# https://www.freecodecamp.org/news/what-is-stratified-random-sampling-definition-and-python-example/#:~:text=In%20machine%20learning%2C%20stratified%20sampling,here%20to%20download%20the%20dataset.
# https://towardsdatascience.com/why-do-we-set-a-random-state-in-machine-learning-models-bb2dc68d8431
# https://www.askpython.com/python/examples/split-data-training-and-testing-set#:~:text=The%20most%20common%20split%20ratio,works%20well%20with%20large%20datasets.
A_train, A_test, B_train, B_test = train_test_split(A, B, test_size=0.2, random_state=42)


# It is vital to perform Feature scaling where all the variables are brought to a similar scale, as the features with high value range dominates in calculating the distances between data.

# In[123]:


A_train.shape


# In[124]:


from sklearn.preprocessing import StandardScaler


# In[125]:


scaler = StandardScaler()


# In[127]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
A_train = scaler.fit_transform(A_train)
A_test = scaler.transform(A_test)


# In[128]:


A_train


# In[136]:


# Analysis before SMOTE 
# Logistic regression Model
from sklearn.linear_model import LogisticRegression
# Creating instance for this model
logis = LogisticRegression()
# train the model
logis.fit(A_train,B_train)
# Predict the model
B_pred = logis.predict(A_test)


# In[142]:


# Checking accuracy before applying SMOTE
# importing necessary libraries
from sklearn.metrics import accuracy_score
accuracy_score(B_test,B_pred)


# In[144]:


# Checking precision score before applying SMOTE
# importing necessary libraries
from sklearn.metrics import precision_score,recall_score,f1_score
precision_score(B_test,B_pred)


# In[145]:


# Recall score before applying SMOTE
recall_score(B_test,B_pred)


# In[146]:


# f1 score before applying SMOTE
f1_score(B_test,B_pred)


# In[3]:


# Applying Oversampling with SMOTE
# importing SMOTE
from imblearn.over_sampling import SMOTE
# Assigning in new variable
A_new,B_new = SMOTE().fit_resample(A,B)


# In[ ]:




