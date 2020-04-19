import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score, accuracy_score

d=pd.read_csv("C:/Users/Umraz/Desktop/diabetes.csv")
d.head()

print(len(d))

d.isna().sum()

z=['Glucose','BloodPressure','SkinThickness','BMI','Insulin']

for column in z:
    d[column]=d[column].replace(0,np.NaN)
    mean=int(d[column].mean(skipna=True))
    d[column]=d[column].replace(np.NaN,mean)

x=d.iloc[:,0:8]
y=d.iloc[:,8]
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=0,test_size=0.2)

sc=StandardScaler()
x_train=sc.fit_transform(x_train)
x_test=sc.fit_transform(x_test)

import math
math.sqrt(len(y_test))

classifier=KNeighborsClassifier(n_neighbors=11, p=2, metric='euclidean')

classifier.fit(x_train,y_train)

y_pred=classifier.predict(x_test)
y_pred

cm=confusion_matrix(y_test,y_pred)
print(cm)

print(f1_score(y_test,y_pred))

print(accuracy_score(y_test,y_pred))