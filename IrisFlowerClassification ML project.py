#!/usr/bin/env python
# coding: utf-8

# ## Lets Grow More LGM VIP Internship August(2022)
# ### Task-1: Iris Flower Classification ML Project 
# ### Author: Gore Shivani Kailas

# Importing library

# In[51]:


import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd
import seaborn as sns


# ## Loading dataset

# In[52]:


# load the iris dataset
df=pd.read_csv('C:/Users/Hp/OneDrive/Desktop/Iris .csv')
df.head()


# Dataset Info

# In[53]:


df.info()


# Finding null values

# In[54]:


df.isnull().sum


# Drop Id values

# In[55]:


df=df.drop(columns=['Id'])
df.head()


# In[56]:


df.describe()


# In[57]:


df.shape


# In[58]:


df['Species'].value_counts()


# In[59]:


df['SepalLengthCm'].hist()


# In[60]:


df['SepalWidthCm'].hist()


# In[61]:


df['PetalWidthCm'].hist()


# In[62]:


#scatter plot
colors = ['red','orange','blue']
species =['Iris-virginica','Iris-versicolor','Iris-setosa']


# In[63]:


fig=plt.figure()
ax=fig.add_axes([0,0,1,1])
ax.axis('equal')
l = ['Versicolor','Setosa','Virginica']
s = [50,50,50]
ax.pie(s, labels = l,autopct= '%1.2f%%')
plt.show()


# In[64]:


for i in range(3):
    x = df[df['Species']== species[i]]
    plt.scatter(x['SepalLengthCm'],x['SepalWidthCm'],c=colors[i],label=species[i])
    plt.xlabel("Sepal Length")
    plt.ylabel("Sepal Width")
    plt.legend()
    
    


# In[65]:


for i in range(3):
    x = df[df['Species']== species[i]]
    plt.scatter(x['PetalLengthCm'],x['PetalWidthCm'],c=colors[i],label=species[i])
    plt.xlabel("Petal Length")
    plt.ylabel("Petal Width")
    plt.legend()
    


# In[66]:


for i in range(3):
    x = df[df['Species']== species[i]]
    plt.scatter(x['SepalLengthCm'],x['PetalLengthCm'],c=colors[i],label=species[i])
    plt.xlabel("Sepal Length")
    plt.ylabel("Petal Length")
    plt.legend()
    


# ##  checking for outliers

# In[67]:


import matplotlib.pyplot as plt
plt.figure(1)
plt.boxplot([df['SepalLengthCm']])
plt.figure(2)
plt.boxplot([df['SepalWidthCm']])
plt.show()


# In[68]:


df.plot(kind ='density',subplots = True,layout =(3,3),sharex = False)


# In[69]:


df.plot(kind ='box',subplots = True,layout=(2,5),sharex = False)


# In[70]:


sns.pairplot(df,hue='Species')


# In[71]:


df.corr()


# In[72]:


corr = df.corr()
fig,ax = plt.subplots(figsize=(5,4))
sns.heatmap(corr,annot=True, ax=ax, cmap ='coolwarm')

