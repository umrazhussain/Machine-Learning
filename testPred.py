#!/usr/bin/env python
# coding: utf-8

# In[5]:


import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns; sns.set()
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.datasets import fetch_20newsgroups
data=fetch_20newsgroups()
data.target_names


# In[14]:


cat=['alt.atheism','comp.graphics','comp.os.ms-windows.misc','comp.sys.ibm.pc.hardware','comp.sys.mac.hardware','comp.windows.x',
     'misc.forsale','rec.autos','rec.motorcycles','rec.sport.baseball','rec.sport.hockey','sci.crypt','sci.electronics',
     'sci.med','sci.space','soc.religion.christian','talk.politics.guns','talk.politics.mideast','talk.politics.misc','talk.religion.misc']
train=fetch_20newsgroups(subset='train', categories=cat)
test=fetch_20newsgroups(subset='test', categories=cat)

print(train.data[5])
len(train.data)


# In[15]:


print(len(test.data))


# In[17]:


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

model=make_pipeline(TfidfVectorizer(),MultinomialNB())

model.fit(train.data,train.target)

labels=model.predict(test.data)


# In[21]:


from sklearn.metrics import confusion_matrix
mat=confusion_matrix(test.target,labels)
sns.heatmap(mat.T,square=True,annot=True,fmt='d',cbar='False',
           xticklabels=train.target_names,
           yticklabels=train.target_names)

plt.xlabel('true label')
plt.ylabel('predicted label')


# In[25]:


def pred(S,train=train,model=model):
    pre=model.predict([S])
    return train.target_names[pre[0]]


# In[34]:


pred('number 1 fastest car in the world')


# In[ ]:




