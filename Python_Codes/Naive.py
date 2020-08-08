import numpy as np
X = np.array([[-1,-1],[-2,-1]
Y = np.array([1,1,1,2,2,2])
from sklearn.naive_bayes import
clf = GaussianNB()
clf.fit(X,Y)
ssianNB()
print(clf.predict([[-0.8,-1]]))
print(clf.predict([[-0.8,-1]]))