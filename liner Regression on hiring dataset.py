#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd 
import numpy as np
import seaborn as sns
import pickle


# In[2]:


hire = pd.read_csv('hiring.csv')


# In[3]:


hire.head()


# In[4]:


hire.info()


# In[5]:


hire.shape


# In[6]:


hire.describe()


# In[7]:


hire.isnull().sum()


# In[8]:


hire.corr()


# In[9]:


hire.drop(['Index'], axis = 1, inplace=True)


# In[10]:


hire.head()


# In[11]:


from word2number import w2n


# In[12]:


hire['experience'] = hire['experience'].apply(w2n.word_to_num)


# In[13]:


hire.head()


# In[14]:


x = hire.iloc[:,:-1]


# In[15]:


y = hire['salary']


# In[16]:


x.shape, y.shape


# In[17]:


from sklearn.linear_model import LinearRegression


# In[18]:


regression = LinearRegression()


# In[19]:


regression.fit(x, y)


# In[20]:


# Saving model to disk
pickle.dump(regression, open('model.pkl','wb'))

# Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))
print(model.predict([[2, 9, 6]]))


# In[ ]:




