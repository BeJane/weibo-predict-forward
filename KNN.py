import pandas as pd
import numpy as np

from matplotlib.colors import ListedColormap

import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

from sklearn.neighbors import KNeighborsClassifier
data = pd.read_csv('/home/aistudio/work/traindata3.csv')
#test = list(data.fcs)
#print(test)
#data = data[~data['fcs'].isin([0])]
x_train = data[['uid','tid','month','wday','fid']]
#x_train = data[['tid','fid']]
y_train = data[['fcs']]//100 *100
#y_train=data[['lcs']]
testdata = pd.read_csv('/home/aistudio/work/testdata1.csv')
#testdata=testdata[~testdata['fcs'].isin([0])]

x_test = testdata[['uid','tid','month','wday','fid']]
#x_test = testdata[['tid','fid']]
y_test = testdata['fcs']//100*100
#print(y_test)
transfer = StandardScaler()
x_train = transfer.fit_transform(x_train)
x_test = transfer.fit_transform(x_test)

estimator = KNeighborsClassifier()
estimator.fit(x_train,np.array(y_train).ravel())
score = estimator.score(x_test,y_test)
print(score)
classes = estimator.classes_
print(classes)
print(len(classes))


y=estimator.predict(x_test)
print(y.max())

plt.plot(y_test,color='red',label="test")

plt.plot(y,color = 'blue',label='predict')
plt.legend(loc="upper right")

plt.xlabel("")
plt.ylabel('fcs')
plt.show()