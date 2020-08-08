from collections import defaultdict
import csv
from sklearn import svm
from sklearn.metrics import accuracy_score
import pickle
from sklearn.externals import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.datasets import load_files
from sklearn.naive_bayes import MultinomialNB
from sklearn.cross_validation import train_test_split
from sklearn import metrics
from sklearn.feature_extraction.text import TfidfVectorizer
model = joblib.load('comment_trainer.pkl') 
import pickle
from sklearn.externals import joblib
tf = joblib.load('tfidf_matrix.pkl') 
import codecs
import re

episodes = defaultdict(list)
episodes = joblib.load('episodes.pkl')
        
path = r'F:\Rafi\My_Study\4_1\AI_Lab\aclImdb\train'
dataset = load_files(path, shuffle= False, decode_error='ignore', random_state=None,load_content=True)
trainData,testData,trainTarget,testTarget = train_test_split(dataset.data,dataset.target,train_size  = 0.9, test_size=0.1,random_state=42);
testMatrix = tf.transform(testData)
predicted = model.predict(testMatrix)
counter = 0.0 
for prediction,data,target in zip(predicted,testData,testTarget):
	polarityScore = 0.0
	for word in re.findall(r'\w+', data):
		values = episodes[word.lower()]
        for val in values:
            polarityScore += float(val)
	if polarityScore < 0 and prediction == 0 and target == 0:
		counter+=1.0
	elif polarityScore > 0 and prediction == 1 and target == 1:
		counter+=1.0    
print(counter)
print(len(predicted))


