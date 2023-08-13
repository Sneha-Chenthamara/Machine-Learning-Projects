#!/usr/bin/env python
# coding: utf-8

# In[41]:


import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
from sklearn.decomposition import NMF
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors


# In[2]:


df = pd.read_csv("spotify (1).xls",index_col = [0])


# In[3]:


df.head()


# In[4]:


df.info()


# In[5]:


x = df.values


# In[6]:


nmf = NMF(n_components = 100, max_iter = 200, random_state = 234)


# In[7]:


nmf.fit(x)


# In[8]:


user_matrix = nmf.transform(x)


# In[9]:


user_matrix


# In[10]:


song_matrix = nmf.components_.T


# In[11]:


song_matrix


# In[12]:


user_matrix.shape, song_matrix.shape


# # Recommendation System I

# In[13]:


# In this method we will follow a user centeric approach i.e we will use the user matrix to build the recommendation

# Steps :
# 1. Consider recommending songs to user_1 who is located at row location 0.
# 2. In order to do this we need to find the Euclidean distance between user_1 and the remaining 999 users.
# 3. After finding the distance we will pick up five closest users to user_1
# 4. We will then find the songs heard by these five users and recommend them to user_1
# 5. We will repeat the above 4 steps again and again to recommend songs to all the users


# In[14]:


def calculate_distance(user_1,user_2) :
    return pow(sum(pow(user_1[x] - user_2[x],2) for x in range(len(user_1))),0.5)
    


# In[15]:


calculate_distance(user_matrix[0],user_matrix[1])


# In[16]:


calculate_distance(user_matrix[27],user_matrix[23])


# In[17]:


def distance_all_users(base_user,user_matrix):
    distance  = []
    for i in range(len(user_matrix)):
        if base_user != i:
            distance.append(calculate_distance(user_matrix[base_user],user_matrix[i]))
    return distance
            


# In[18]:


distance = distance_all_users(0,user_matrix)


# In[19]:


top_five_users = np.argsort(distance)[0:5]


# In[20]:


for i in top_five_users :
    temp = pd.DataFrame(df.iloc[i])
    temp = temp[temp.values != 0] 
    print("Songs heard by user ",i)
    print(temp.index)


# In[21]:


def recommend_songs(closest_users,df,number_of_songs):
    
    # Picking out the songs heard by the closest users
    temp = df.iloc[closest_users]
    
    # Making a dictonary of songs heard maximum times by the closest users
    max_heard = temp.max().to_dict()
    
    # Sorting the above dictionary in order to find out the maximum heard songs which can be recommended
    sorted_dictionary = sorted(max_heard.items(),key = lambda keyvalue : (keyvalue[1],keyvalue[0]),reverse = True)
    
    # Picking up only the number of songs required
    required_songs = sorted_dictionary[0:number_of_songs]
    
    
    return [x[0] for x in required_songs]


# In[22]:


recommend_songs(top_five_users,df,10)


# # Recommendation System II

# In[23]:


# In this method we will follow a song centeric approach i.e we will use the song matrix to build the recommendation

# Steps :
# 1. First we wil build clusters on the song matrix.
# 2. Then we will consider any user listening to a particular song. For example song_5
# 3. We will try to figure out in which cluster is song_5 present in. For example cluster_2
# 4. After that we will try to find out other songs present in cluster_2.
# 5. We will then recommend those songs to the user.


# In[27]:


# Building clusters

wcss = {}
for k in range(1,50):
    kmeans = KMeans(n_clusters = k, random_state = 98).fit(song_matrix)
    wcss[k] = kmeans.inertia_


# In[28]:


wcss


# In[30]:


plt.plot(wcss.keys(),wcss.values())


# In[50]:


def song_recommendation(df,n_cluster,song_matrix,song_name,no_of_songs):
    
    # Making clusters of the song
    kmeans = KMeans(n_clusters=n_cluster, random_state=34).fit(song_matrix)
    
    # Assigning cluster values to every song
    all_song_in_cluster = list(kmeans.predict(song_matrix))
    
    # Finding the position of the user's song
    index_of_song = df.columns.to_list().index(song_name)
    
    # Picking out the values of the song from the song matrix
    song_vector = song_matrix[index_of_song]
    
    # Finding out which other songs are present in the same cluster where the given song is present
    song_cluster = [x for x in range(len(all_song_in_cluster)) if all_song_in_cluster[x] == kmeans.predict([song_vector])]
    
    # Values of the songs present in the same cluster
    song_values = song_matrix[song_cluster]
    
    # Applying nearest neighbor algorithm
    nn = NearestNeighbors(n_neighbors=no_of_songs)
    nn.fit(song_values)
    
    # Picking out the nearest song
    recommend_song = nn.kneighbors([song_matrix[index_of_song]])
    
    # Creating a list of all songs
    song_list = df.columns
    
    return[song_list[x] for x in recommend_song[1][0]]


# In[51]:


song_recommendation(df,10,song_matrix,"song_6",5)


# In[ ]:




