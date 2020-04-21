import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
%matplotlib inline

d=pd.read_csv('C:/Users/Umraz/Desktop/stratups.csv')
x=d.iloc[:, :-1].values
y=d.iloc[:,4].values

d.head()

sns.heatmap(d.corr(), linewidth=0.5, annot=True)

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
le=LabelEncoder()
x[:,3]=le.fit_transform(x[:,3])

ohe=OneHotEncoder()
x=ohe.fit_transform(x).toarray()
print(x)

x=x[:,1:]

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.linear_model import LinearRegression
reg=LinearRegression()
reg.fit(x_train,y_train)

ypre=reg.predict(x_test)
ypre

plt.scatter(y_test, ypre)

print(reg.coef_)

print(reg.intercept_)

from sklearn.metrics import r2_score,accuracy_score
r2_score(y_test,ypre)

sns.distplot((y_test-ypre))