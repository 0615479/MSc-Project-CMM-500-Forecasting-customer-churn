#!/usr/bin/env python
# coding: utf-8

# In[ ]:





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


# In[12]:


#https://www.w3resource.com/pandas/dataframe/dataframe-describe.php
churn_data.describe(include = 'all')


# It displays all the columns of the data frame regardless of its datatype and the overall statistical values of the categorical and numerical columns.

# In[13]:


# Dropping inappropriate columns that is not required for this analysis
churn_data.columns


# In[14]:


#https://www.listendata.com/2019/06/pandas-drop-columns-from-dataframe.html
#(“How to Drop One or Multiple Columns from Pandas Dataframe” n.d.)
churn_data = churn_data.drop(['RowNumber', 'CustomerId', 'Surname'],axis=1)


# The irrelevant columns are dropped to reduce the memory usage and for the ease of analysis purposes

# In[20]:


churn_data.head()


# After dropping the irrelevant attributes there is a significant decrease in the memory usage from 1015.8+ KB to 859.5+ KB, where the former is the one, before dropping the irrelevant columns.

# In[16]:


# Encoding ctaegorical data
churn_data['Geography'].unique()


# In[17]:


# Encoding ctaegorical data
# one hot/ dummy encoding
# https://dataindependent.com/pandas/pandas-get-dummies-pd-get_dummies/
churn_data = pd.get_dummies(churn_data, drop_first = True)


# In[19]:


churn_data['Exited'].value_counts()


# Here value counts() function is used to find the distribution of targeted variable. The output will be sorted in the descending order with the firat element being the frequently occurring value. 0 means the customer is not leaving the bank and 1 means the customer is leaving the bank.
# 

# In[27]:


# Visualising
sns.countplot(churn_data['Exited'])


# In[ ]:




