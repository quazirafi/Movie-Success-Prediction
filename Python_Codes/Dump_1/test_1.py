from collections import defaultdict
import csv
corpus = []
labels = []

filenames = []
with open('F:/Rafi/My_Study/4_1/AI_Lab/Python_Codes/Version1/pos2.txt', 'r') as infile:
    for line in infile:
        filenames.append(line.rstrip())
doc = ""
for filename in filenames:
    s = "F:/Rafi/My_Study/4_1/AI_Lab/aclImdb/train/pos/" + filename
    with open(s, 'r') as infile2:
        for line in infile2:
            doc += line
        corpus.append(doc)
        labels.append(1)
        doc = ""
print(len(corpus))
doc = ""
filenames = []
with open('F:/Rafi/My_Study/4_1/AI_Lab/Python_Codes/Version1/neg2.txt', 'r') as infile:
    for line in infile:
        filenames.append(line.rstrip())
doc = ""
for filename in filenames:
    s = "F:/Rafi/My_Study/4_1/AI_Lab/aclImdb/train/neg/" + filename
    with open(s, 'r') as infile2:
        for line in infile2:
            doc += line
        corpus.append(doc)
        labels.append(0)
        doc = ""
print(len(corpus))
print(len(labels))

filenames = []
with open('F:/Rafi/My_Study/4_1/AI_Lab/aclImdb/test/mixed/posnfull.txt', 'r') as infile:
    for line in infile:
        filenames.append(line.rstrip())

filenamesneg = []
with open('F:/Rafi/My_Study/4_1/AI_Lab/aclImdb/test/mixed/negnfull.txt', 'r') as infile:
    for line in infile:
        filenamesneg.append(line.rstrip())

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
tf = TfidfVectorizer(analyzer='word', ngram_range=(1,2), min_df = 0, max_features = 100000, stop_words = 'english')
tfidf_matrix = tf.fit_transform(corpus)
print(tfidf_matrix.shape)
training = tfidf_matrix[:3400]
l_train = labels[:3400]
joblib.dump(training, 'tfidf_matrix.pkl') 


model = svm.SVC(kernel='rbf', C=2, gamma=1) 
model.fit(training, l_train)
import pickle
from sklearn.externals import joblib
joblib.dump(model, 'comment_trainer.pkl') 
test = []
# for filename in filenames:
s = "F:/Rafi/My_Study/4_1/AI_Lab/POSTESTDATA/gravity.txt"
# print(filename+"\n")
with open(s, 'r') as infile2:
    for line in infile2:
        print(line+"\n")
        # new_doc_tfidf_matrix = tf.transform(line)
        test.append(line)
        # predicted = model.predict(new_doc_tfidf_matrix)

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
# for filename in filenamesneg:
#     s = "F:/Rafi/My_Study/4_1/AI_Lab/NEGTESTDATA/" + filename
#     print(filename+"\n")
#     with open(s, 'r') as infile2:
#         for line in infile2:
#             new_doc_tfidf_matrix2 = tf.transform(line)
#             predicted = model.predict(new_doc_tfidf_matrix2)
#             print(predicted)
#             print(" ")
#     print("\n")



