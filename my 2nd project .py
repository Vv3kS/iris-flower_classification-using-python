#!/usr/bin/env python
# coding: utf-8

# In[41]:


import pandas as pd
import numpy as np
import plotly
import plotly.express as px
import plotly.offline as pyo
import cufflinks as cf
from plotly.offline import init_notebook_mode,plot,iplot
import graphviz

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.metrics import accuracy_score
import os


# In[4]:


pyo.init_notebook_mode(connected=True)
cf.go_offline()


# In[5]:


iris=pd.read_csv(r'V:\my project\my 2nd project\Iris.csv')


# In[6]:


iris


# In[8]:


iris.shape


# In[8]:


iris.drop('Id',axis=1,inplace=True)


# In[9]:


iris


# In[10]:


#performing some visualization


# In[11]:


px.scatter(iris,x='Species',y='PetalWidthCm',size='PetalWidthCm')


# In[12]:


plt.bar(iris['Species'],iris['PetalWidthCm'])


# In[13]:


px.bar(iris,x='Species',y='PetalWidthCm')
iris.iplot(kind='bar',x=['Species'],y=['PetalWidthCm'])


# In[14]:


px.line(iris,x='Species',y='PetalWidthCm')
iris.rename(columns={'SepalLengthCm':'SepalLength','SepalWidthCm':'SepalWidth','PetalWidthCm':'PetalWidth','PetalLengthCm':'PetalLength'},inplace=True)
iris


# In[15]:


px.scatter_matrix(iris,color='Species',title='Iris',dimensions=['SepalLength','SepalWidth','PetalWidth','PetalLength'])


# In[16]:


iris


# In[17]:


X=iris.drop(['Species'],axis=1)
X


# In[20]:


y=iris['Species']
y


# In[21]:


from sklearn.preprocessing import LabelEncoder

le=LabelEncoder()

y=le.fit_transform(y)


# In[22]:


y


# In[23]:


X


# In[24]:


X=np.array(X)
X


# In[25]:


y


# In[26]:


from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=0)


# In[27]:


X_test


# In[28]:


X_test.size


# In[29]:


iris


# In[30]:


# calling my machine learning algorithm Decision Tree


# In[31]:


from sklearn import tree

DT=tree.DecisionTreeClassifier()
DT.fit(X_train,y_train)


# In[32]:


y_train.size


# In[33]:


prediction_DT=DT.predict(X_test)
accuracy_DT=accuracy_score(y_test,prediction_DT)*100


# In[34]:


accuracy_DT


# In[35]:


y_test


# In[36]:


prediction_DT


# In[37]:


os.environ["PATH"]+= os.pathsep+(r'C:\Graphviz\bin')


# In[39]:


vvek_data=tree.export_graphviz(DT,out_file=None,feature_names=iris.drop(['Species'],axis=1).keys(),class_names=iris['Species'].unique(),filled=True,rounded=True,special_characters=True)


# In[40]:


graphviz.Source(vvek_data)


# In[ ]:





# In[ ]:




