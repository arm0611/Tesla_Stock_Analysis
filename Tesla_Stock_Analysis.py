#!/usr/bin/env python
# coding: utf-8

# In[111]:


import pandas as pd
import numpy as np
from sklearn import metrics
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt  
import math
from sklearn.datasets import load_iris
from factor_analyzer import FactorAnalyzer
from scipy.stats import skew,kurtosis,norm,skewtest,kurtosistest


# In[112]:


pd.__version__


# In[113]:


data = pd.read_csv('D:/Akshay/1NMIMS/Semester 3/Computational Statistics/Project/TSLA.csv')


# In[114]:


data.head()


# In[115]:


data['Date'] = pd.to_datetime(data.Date)


# In[46]:


data.drop('Adj Close', axis = 1, inplace = True)


# In[47]:


data.head()


# In[48]:


data.isnull().sum()


# In[49]:


data.isna().any()


# In[50]:


data.info()


# In[51]:


data.describe()


# In[52]:


from sklearn.model_selection import train_test_split

# Dependent(Y) and Independent Variables(X)
X  = data[['Open','High','Low','Volume']]
Y = data['Close']


# In[53]:


train, test = train_test_split(data, test_size=0.3, shuffle=False)


# In[54]:


# Training data:
X_train  = train[['Open','High','Low','Volume']]
Y_train = train['Close']


# In[55]:


# testing data
X_test  = test[['Open','High','Low','Volume']]
Y_test = test['Close']


# In[56]:


X_train.shape


# In[57]:


X_test.shape


# In[58]:


from sklearn.linear_model import LinearRegression
from sklearn.metrics import confusion_matrix, accuracy_score


# In[59]:


reg = LinearRegression()


# In[60]:


reg.fit(X_train, Y_train)


# In[61]:


print(reg.coef_)


# In[62]:


print(reg.intercept_)


# In[63]:


pred_value = reg.predict(X_test)


# In[64]:


print (X_test)


# In[65]:


d = pd.DataFrame({'Actual Price': Y_test, 'Predicted Price': pred_value })


# In[66]:


d.head(20)


# In[67]:


reg.score(X_test, Y_test)


# In[68]:


print('Mean Absolute Error: ', metrics.mean_absolute_error(Y_test, pred_value))


# In[69]:


print ('Mean Squared Error: ', metrics.mean_squared_error(Y_test, pred_value))


# In[70]:


print ('Root Mean Squared Error: ', math.sqrt(metrics.mean_squared_error(Y_test, pred_value)))


# In[71]:


graph = d.head(20)


# In[72]:


graph.plot(kind = 'bar')


# In[73]:


graph.plot()


# In[74]:


data.drop('Date', axis = 1, inplace = True)
data.head()


# In[75]:


data.info()


# In[76]:


from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity
chi_square_value,p_value=calculate_bartlett_sphericity(data)
chi_square_value, p_value


# In[77]:


from factor_analyzer.factor_analyzer import calculate_kmo
kmo_all,kmo_model=calculate_kmo(data)


# In[78]:


kmo_model


# In[79]:


fa = FactorAnalyzer(n_factors = 2, rotation = "varimax")


# In[80]:


fa.fit(data)


# In[81]:


loadings = fa.loadings_


# In[82]:


ev, v = fa.get_eigenvalues()


# In[83]:


ev


# In[84]:


xvals = range(1, data.shape[1]+1)


# In[85]:


plt.scatter (xvals, ev)
plt.plot (xvals, ev)
plt.title ('Scree Plot')
plt.xlabel('Factor')
plt.ylabel('Eigenvalue')
plt.grid()
plt.show()


# In[117]:


fa.loadings_


# In[87]:


fa.get_factor_variance()


# In[90]:


cl = data['Close']


# In[106]:


dt = data['Date']


# In[110]:


plt.figure(figsize=(10,4))
plt.title('Close price data')
plt.plot(dt, cl)
plt.xlabel('Date')
plt.ylabel('Price')


# In[94]:


returns = cl.pct_change(1).dropna()


# In[95]:


returns


# In[96]:


plt.hist(returns,bins="rice",label="Daily close price")
plt.legend()
plt.show()


# In[97]:


plt.boxplot(returns,labels=["Daily close price"])
plt.show()


# In[98]:


np.mean(returns)


# In[99]:


np.std(returns)


# In[100]:


np.quantile(returns,0.5)


# In[101]:


skew(returns)


# In[102]:


kurtosis(returns)


# In[ ]:




