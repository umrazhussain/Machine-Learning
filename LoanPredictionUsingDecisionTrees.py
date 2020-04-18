import numpy as np
import pandas as pd
from sklearn.model.selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.matrics import accuracy_score
from sklearn import tree

bd=pd.read_csv('C:/Users/Umraz/Desktop/loan.csv',sep=',',header=0)

print("Dataset length is::",len(bd))
print("Dataset length is::",bd.shape)

print("DataSet is below::::")
bd.head()

X=bd.values[:,0:7]
X

Y=bd.values[:,0]
Y

X_train, X_test, Y_train, Y_test=train_test_split(X,Y,test_size=0.5,random_state=100)

bd_entropy=DecisionTreeClassifier(criterion="entropy", random_state=100, max_depth=10, min_samples_leaf=7)

bd_entropy.fit(X_train,Y_train)

res=bd_entropy.predict(X_test)

print("decision tree REsult is ::",res)

print("model accuracy is ::",accuracy_score(Y_test,res)*100)