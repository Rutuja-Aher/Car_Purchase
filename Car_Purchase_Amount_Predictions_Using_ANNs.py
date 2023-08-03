#!/usr/bin/env python
# coding: utf-8

# # STEP #0: LIBRARIES IMPORT
# 

# In[97]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# # STEP #1: IMPORT DATASET

# In[98]:


car_df = pd.read_csv('Car_Purchasing_Data.csv', encoding = 'ISO-8859-1')


# In[99]:


car_df


# In[100]:


car_df.head()


# # STEP #2: VISUALIZE DATASET

# In[101]:


sns.pairplot(car_df)


# # STEP #3: CREATE TESTING AND TRAINING DATASET/DATA CLEANING

# In[102]:


x = car_df.drop(['Customer Name','Customer e-mail','Country', 'Car Purchase Amount'], axis = 1)


# In[103]:


x


# In[104]:


y = car_df['Car Purchase Amount']


# In[105]:


y


# In[106]:


x.shape


# In[107]:


y.shape


# In[108]:


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
x_scaled = scaler.fit_transform(x)


# In[109]:


x_scaled.shape


# In[110]:


scaler.data_max_


# In[111]:


scaler.data_min_


# In[112]:


print(x_scaled[:,0])


# In[113]:


y.shape


# In[114]:


y = y.values.reshape(-1,1)


# In[115]:


y.shape


# In[116]:


y_scaled = scaler.fit_transform(y)


# In[117]:


y_scaled


# # STEP#4: TRAINING THE MODEL

# In[118]:


x_scaled.shape


# In[119]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x_scaled, y_scaled, test_size = 0.25)


# In[120]:


x_train.shape


# In[121]:


x_test.shape


# In[122]:


pip install tensorflow


# In[123]:


pip install tensorflow==<version>


# In[124]:


import tensorflow.keras
from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import MinMaxScaler

model = Sequential()
model.add(Dense(25, input_dim=5, activation='relu'))
model.add(Dense(25, activation='relu'))
model.add(Dense(1, activation='linear'))
model.summary()


# In[125]:


model.compile(optimizer = 'adam', loss = 'mean_squared_error')


# In[126]:


epochs_hist = model.fit(x_train, y_train, epochs = 20, batch_size = 25, verbose = 1, validation_split = 0.2)


# # STEP#5: EVALUATING THE MODEL 

# In[127]:


epochs_hist.history.keys()


# In[128]:


plt.plot(epochs_hist.history['loss'])
plt.plot(epochs_hist.history['val_loss'])

plt.title('Model Loss Progression During Training/Validation')
plt.ylabel('Training and Validation Losses')
plt.xlabel('Epoch Number')
plt.legend(['Training Loss', 'Validation Loss'])


# In[129]:


# Gender, Age, Annual Salary, Credit Card Debt, Net Worth

X_Testing = np.array([[1, 50, 50000, 10985, 629312]])


# In[130]:


y_predict = model.predict(X_Testing)
y_predict.shape


# In[131]:


print('Expected Purchase Amount=', y_predict[:,0])

