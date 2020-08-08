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
test = []
# for filename in filenames:
# s = "F:/Rafi/My_Study/4_1/AI_Lab/POSTESTDATA/gravity.txt"
s = "F:/Rafi/My_Study/4_1/AI_Lab/AIData/Test/disastermovie1.8.txt"
# print(filename+"\n")
with open(s, 'r') as infile2:
    for line in infile2:
        print(line+"\n")
        # new_doc_tfidf_matrix = tf.transform(line)
        test.append(line)
        # predicted = model.predict(new_doc_tfidf_matrix)
tf = joblib.load('vectorizer.pkl') 
new_doc_tfidf_matrix = tf.transform(test)
predicted = model.predict(new_doc_tfidf_matrix)
print(predicted)
counter = 0
for i in range(len(predicted)):
    print(predicted[i])
    if predicted[i] == 1:
        counter+=1
print('percentage of positive comments ')
print(counter)




print(len(predicted))
print((float(counter)/float(len(predicted)))*100.0)
