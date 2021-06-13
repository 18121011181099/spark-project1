#!/usr/bin/env python
# coding: utf-8

# # *Linear Regression with Python Scikit Learn*
# In this section we will see how the Python Scikit-Learn library for machine learning can be used to implement regression functions. We will start with simple linear regression involving two variables.
# 
# # *Simple Linear Regression*
# In this regression task we will predict the percentage of marks that a student is expected to score based upon the number of hours they studied. This is a simple linear regression task as it involves just two variables.

# In[37]:


import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model

get_ipython().run_line_magic('matplotlib', 'inline')


# In[38]:


# Reading data from the given link
url = "http://bit.ly/w-data"
data= pd.read_csv(url)
print("Data imported successfully")

data.head(8)


# # Plot the given data points on 2-D graph to eyeball our dataset and see if we can manually find any relationship between the data. We can create the plot with the following script:

# In[42]:


plt.figure(dpi=120)
k=range(0,len(data))
x=data['Hours']
y=data['Scores']
plt.scatter(x,y,color='r',label='hours vs Percentage')
plt.xlabel('Hours stdied')
plt.ylabel(' percentage score')
plt.title('percentage prediction using hours of study')
plt.legend()


# # From the above graph we can notice that there is a positive linear relation between the hours studied and percentage score.

# In[43]:


## lets divide the data in to two 'attributes' and 'labels'
X=df1[['Hours']]
y=df1['score']


# # now we have our attributes and labels, the next step is to split this data into training and test sets. We'll do this by using Scikit-Learn's built-in train_test_split() method:

# In[44]:


from sklearn.model_selection import train_test_split
model=train_test_split
X_train,X_test,y_train,y_test =train_test_split(X,y,test_size=.2,random_state=0)


# # now after train and test the data its time to train our algorithm

# In[45]:


from sklearn.linear_model import LinearRegression
lin_reg=LinearRegression()
lin_reg=linear_model.LinearRegression()
lin_reg.fit(X_train, y_train)


# In[47]:


## ploting the linear regression model in graph 


# In[48]:


line = lin_reg.coef_*X+lin_reg.intercept_  ## y=mx+c

# Plotting for the test data
plt.scatter(X, y)
plt.plot(X, line,color='r');
plt.show()


# # Making predections
# 

# In[49]:


print(X_test) # Testing data - In Hours
y_pred = lin_reg.predict(X_test) # Predicting the scores


# In[ ]:


## Comparing actual and predicted data


# In[50]:


df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})  
df 


# # What will be predicted score if a student studies for 9.25 hrs/ day?

# In[51]:


Hours = 9.25
lin_reg.predict([[9.25]])


# # Evaluating the model by Mean square error method .This step is particularly important to compare how well different algorithms perform on a particular dataset.

# In[52]:


from sklearn import metrics  
print('Mean Absolute Error:', 
      metrics.mean_absolute_error(y_test, y_pred)) 


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




