from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn import metrics
%matplotlib inline
digits=load_digits()

print("img data shape",digits.data.shape)
print("label data shape",digits.target.shape)

import numpy as np
import matplotlib.pyplot as plt

plt.figure(figsize=(20,4))
for index,(image,label) in enumerate(zip(digits.data[0:5],digits.target[0:5])):
    plt.subplot(1,5,index+1)
    plt.imshow(np.reshape(image,(8,8)),cmap=plt.cm.gray)
    plt.title('training: %i\n'%label,fontsize=20)

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(digits.data,digits.target,test_size=0.23,random_state=2)

print(x_train.shape,x_test.shape,y_train.shape,x_test.shape,)

from sklearn.linear_model import LogisticRegression

lg=LogisticRegression()
lg.fit(x_train,y_train)

lg.predict(x_test[0].reshape(1,-1))

lg.predict(x_test[0:10])

pred=lg.predict(x_test)
pred

score=lg.score(x_test,y_test)
score

import matplotlib.pyplot as plt
import seaborn as sns 
from sklearn import metrics


cm =metrics.confusion_matrix(y_test,pred)
print(cm)

plt.figure(figsize=(9,9))
sns.heatmap(cm,annot=True,fmt=".3f",linewidth=0.5,square = True,cmap = 'Blues_r')
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
all_sample_title='Accuracy Score : {0}'.format(score)
plt.title(all_sample_title,size=15)

index = 0
classifiedIndex = []
for predict,actual in zip(pred,y_test):
    if predict==actual:
        classifiedIndex.append(index)
    index +=1
plt.figure(figsize=(20,3))
for plotIndex,wrong in enumerate(classifiedIndex[0:4]):
    plt.subplot(1,4,plotIndex +1)
    plt.imshow(np.reshape(x_test[wrong],(8,8)),cmap=plt.cm.gray)
    plt.title('Predicted: {}, Actual: {}'.format(pred[wrong],y_test[wrong]),fontsize=20)