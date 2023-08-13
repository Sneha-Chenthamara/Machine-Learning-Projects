#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import os
from zipfile import ZipFile
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


# In[2]:


file_name = "expression_images_dataset.zip"
destination_path = "expression"

with ZipFile(file_name,"r") as zip:
    print("Trying extract all the files")
    if (not os.path.exists(destination_path)):
        zip.extractall(path = destination_path)
        print("Extraction completed")
    else :
        print("Destination directory already exists")


# In[3]:


im = Image.open("expression/images/KA.AN2.40.tiff")
plt.matshow(im)


# In[4]:


df = pd.read_csv("expression/data.csv")
df.head()


# In[5]:


df.info()


# In[6]:


list(df.columns)


# In[7]:


df['facial_expression'].unique()


# In[8]:


df.groupby("facial_expression")["facial_expression"].count()


# In[9]:


sum(df.groupby("facial_expression")["facial_expression"].count())


# In[10]:


df['facial_expression'].value_counts()


# In[11]:


x = []
for i in range(213):
    path = "expression/"+df.iloc[i][0]
    im = Image.open(path)
    imarray = np.array(im).reshape(65536)
    x.append(imarray)


# In[12]:


len(x)


# In[13]:


len(x[0])


# In[14]:


x[0]


# In[15]:


y = df["facial_expression"]
y


# In[16]:


encoder = LabelEncoder()
y = encoder.fit_transform(df["facial_expression"])
y


# In[17]:


scaler = StandardScaler()
x = scaler.fit_transform(x)
x


# In[18]:


x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2, random_state = 14)


# In[19]:


model = SVC(kernel = "linear")
model.fit(x_train,y_train)
y_pred = model.predict(x_test)
accuracy_score(y_test,y_pred)


# In[20]:


im_test = Image.open("expression/images/NM.HA1.95.tiff")
im_array = np.array(im_test).reshape(65536)
im_array


# In[21]:


test = scaler.transform([im_array])
model.predict(test)


# In[22]:


encoder.inverse_transform([3])


# In[23]:


plt.matshow(im_test)


# In[49]:


im_test_1 = Image.open("expression/images/KR.SA3.79.tiff")
im_array_1 = np.array(im_test_1).reshape(65536)
im_array_1


# In[50]:


test_1 = scaler.transform([im_array_1])
model.predict(test_1)


# In[51]:


encoder.inverse_transform([5])


# In[52]:


plt.matshow(im_test_1)


# In[24]:


stf = StratifiedKFold(n_splits = 10, shuffle = True, random_state = 5)


# In[25]:


cross_val_score(SVC(kernel = "linear"),x,y,cv = stf)


# In[26]:


np.mean(cross_val_score(SVC(kernel = "linear"),x,y,cv = stf))


# In[30]:


param = {"gamma" : ["scale", "auto"], "C" : [3,30,300]}


# In[31]:


tune_dt = GridSearchCV(estimator = SVC(kernel = "linear"), param_grid=param, cv = 10)


# In[32]:


tune_dt.fit(x,y)


# In[33]:


tune_dt.best_params_


# In[34]:


stf = StratifiedKFold(n_splits = 10, shuffle = True, random_state = 5)
np.mean(cross_val_score(SVC(kernel = "linear", C = 3, gamma = "scale"),x,y,cv = stf))


# In[ ]:




