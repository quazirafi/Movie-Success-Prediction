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
joblib.dump(tf, 'vectorizer.pkl')


