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
filenames =[
'2001 A Space Odyssey.txt',
'3 Idiots.txt',
'A Clockwork Orange.txt',
'Bicycle Thieves.txt',
'Double Indemnity.txt',
'Eternal Sunshine of the Spotless Mind.txt',
'Full Metal Jacket.txt',
'Inglourious Basterds.txt',
'Lawrence of Arabia.txt',
'Monty Python and the Holy Grail.txt',
'Singin in the Rain.txt',
'Snatch.txt',
'Taxi Driver.txt',
'The Kid.txt',
'The Sting.txt',
'To Kill a Mockingbird.txt',
'Toy Story 3.txt',
'Toy Story.txt',
'Your Name.txt',
]
tf = joblib.load('tfidf_matrix.pkl') 
counter2 = 0
import codecs
for filename in filenames:
	test = []
	s = "F:\\Rafi\\My_Study\\4_1\\AI_Lab\\AIData\\last_ones\\top\\"+filename
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