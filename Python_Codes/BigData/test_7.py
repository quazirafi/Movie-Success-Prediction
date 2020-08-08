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
from collections import defaultdict
import re
episodes = defaultdict(list)
episodes = joblib.load('episodes.pkl')
model = joblib.load('comment_trainer.pkl') 
import pickle
from sklearn.externals import joblib
path = r'F:\Rafi\My_Study\4_1\AI_Lab\aclImdb\train'
dataset = load_files(path, shuffle= False, decode_error='ignore', random_state=None,load_content=True)
trainData,testData,trainTarget,testTarget = train_test_split(dataset.data,dataset.target,train_size  = 0.5, test_size=0.5,random_state=42);
tf = joblib.load('tfidf_matrix.pkl') 
counter2 = 0
import codecs
new_doc_tfidf_matrix = tf.transform(testData)
predicted = model.predict(new_doc_tfidf_matrix)
counter = 0
totalPolarity = 0.0
for i in range(len(predicted)):
	polarity = 0.0
   	if predicted[i] == 0:
   		counter+=1