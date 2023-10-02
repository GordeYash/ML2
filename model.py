import pandas as pd
import numpy as np

df=pd.DataFrame(pd.read_csv('Iris Data/Iris.csv'))
df.drop('Id',inplace=True,axis=1)
x=df.iloc[:,0:4].values
y=df.iloc[:,-1].values

print(x)
print(y)

from sklearn.preprocessing import LabelEncoder
lb=LabelEncoder()
y=lb.fit_transform(y)

print(y)

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)
print(x_train[0])
print(y_train[0])
from sklearn.svm import SVC
sc=SVC(kernel='linear')
sc.fit(x_train,y_train)
y_pred=sc.predict(x_test)

print(sc.predict([[6.5,3. ,5.2, 2.]]))

import pickle
pickle.dump(sc,open('model.pkl','wb'))