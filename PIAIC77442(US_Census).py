#!/usr/bin/env python
# coding: utf-8

# In[5]:


import numpy as np
import pandas as pd


# In[6]:


import glob

# path = r'C:\Users\lab\Anaconda3\Scripts\jupyter.exe'
all_files = glob.glob( "./*.csv")

li = []

for filename in all_files:
    df =pd.read_csv(filename, index_col = None, header = 0)
    li.append(df)
    
us_census = pd.concat(li, axis = 0, ignore_index = True) 


# In[7]:


us_census.drop(columns = ['Unnamed: 0'], inplace=True)


# In[8]:


us_census[['Male','Female']] = us_census.GenderPop.str.split("_",expand=True)


# In[9]:


us_census.drop(columns = ['GenderPop'], inplace=True)


# In[10]:


us_census['Income'] = us_census.Income.str.strip('$')


# In[11]:


us_census = us_census.replace('%','', regex=True)


# In[12]:


us_census = us_census.replace('F','', regex=True)


# In[13]:


us_census = us_census.replace('M','', regex=True)


# In[14]:


us_census.loc[:,'Hispanic':'Income'] = round(us_census.loc[:,'Hispanic':'Income'].apply(pd.to_numeric),2)


# In[16]:


us_census['Male'] = us_census['Male'].astype(int)


# In[17]:


us_census.drop(columns = ['Female'], inplace=True)


# In[18]:


us_census['Female'] = us_census['TotalPop'] - us_census['Male']


# In[19]:


us_census.dtypes


# In[20]:


import matplotlib.pyplot as plt
plt.scatter(us_census['Female'],us_census['Income'])
plt.xlabel('Salary $', fontsize=18)
plt.ylabel('Female_Population', fontsize=16)
plt.show()


# In[21]:


us_census.duplicated()


# In[22]:


us_census.drop_duplicates(inplace= True)


# In[23]:


plt.scatter(us_census['Female'],us_census['Income'])
plt.xlabel('Salary $', fontsize=18)
plt.ylabel('Female_Population', fontsize=16)
plt.show()


# In[24]:


us_census.duplicated()


# #### Ploting Histograms for races

# In[25]:


us_census


# In[26]:


us_census.describe()


# In[27]:


histo =  round(us_census.loc[:,'Hispanic':'Pacific'].apply(lambda x:x*us_census['TotalPop']/100))


# In[28]:


histo.head()


# In[29]:


histo.fillna(method='bfill', inplace = True)


# In[30]:


histo.astype(int)


# In[31]:


histo['total_pop'] =us_census['TotalPop']
histo['state'] = us_census['State']


# In[32]:


histo.hist(column='Hispanic')


# In[33]:


histo.hist(column='White')


# In[34]:


histo.hist(column='Black')


# In[35]:


histo.hist(column='Native')


# In[36]:


histo.hist(column='Asian')


# In[37]:


histo.hist(column='Pacific')


# In[ ]:




