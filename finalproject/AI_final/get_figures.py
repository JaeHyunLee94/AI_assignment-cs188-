#!/usr/bin/env python
# coding: utf-8

# In[5]:


import matplotlib.pyplot as plt

import pickle


# In[13]:


with open('./cmu_weights.pkl', 'rb') as f:
    data = pickle.load(f)


# In[30]:


fig = plt.figure(figsize=(13,10))
place = [221,222,223,224]
color = ['lightpink',  'orange','seagreen', 'royalblue']
for i, (place,keys) in enumerate(zip(place,data.keys())):
    y = data[keys]
    ax = fig.add_subplot(place)
    ax.plot(y, color=color[i])
    ax.title.set_text(keys)
fig.savefig('log_Q_weights.png')


# In[ ]:




