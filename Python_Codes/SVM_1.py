import numpy as np
from sklearn import svm
X = np.array([["is", "so"," sad", "for", "my" ,"APL" ,"friend"],["i", "missed","the", "new", "moon" ,"trailer","badly"],["i","think","my","bf","is","cheating","on"],["i","am","so","happy","that","i","cried"],["today","was","a","great","day","for","me"],["they","enjoy","today","a","lot","because","now"]])
Y = np.array([0,0,0,1,1,1])
model = svm.svc(kernel='linear', c=1, gamma=1) 
model.fit(X, Y)
#model.score(X, y)
#Predict Output
print(model.predict([["so","i","am","feeling","very","good","today"]]))