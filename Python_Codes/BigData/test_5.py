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
filenames = [
'10,000BC.txt',
'AmericasSweethearts.txt',
'Cats&Dogs.txt',
'Flubber.txt',
'greenhornet.txt',
'Heaven Is for Real.txt',
'inspectorgadget.txt',
'KindergartenCop.txt',
'MaidinManhattan.txt',
'michael.txt',
'objective.txt',
'Ride Along 2.txt',
'ScaryMovie4.txt',
'sexandcity.txt',
'TheHaunting.txt',
'ThePrincessDiaries2RoyalEngagement.txt',
'Twilight.txt',
]
tf = joblib.load('tfidf_matrix.pkl') 
counter2 = 0
import codecs
for filename in filenames:
	test = []
	s = "F:\\Rafi\\My_Study\\4_1\\AI_Lab\\AIData\\neutral2\\neg\\"+filename
	with codecs.open(s, "r",encoding='utf-8', errors='ignore') as infile2:
		for line in infile2:
			test.append(line)
	new_doc_tfidf_matrix = tf.transform(test)
	predicted = model.predict(new_doc_tfidf_matrix)
	counter = 0
	for i in range(len(predicted)):
   		if predicted[i] == 1:
   			counter+=1
   	print(s)
   	print('\n')
	print((float(counter)/float(len(predicted)))*100.0)