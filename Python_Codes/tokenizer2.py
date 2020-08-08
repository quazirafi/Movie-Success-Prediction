from collections import defaultdict
import csv
corpus = []
labels = []

filenames = []
with open('F:/Rafi/My_Study/4_1/AI_Lab/Python_Codes/pos2.txt', 'r') as infile:
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
with open('F:/Rafi/My_Study/4_1/AI_Lab/Python_Codes/neg2.txt', 'r') as infile:
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
with open('F:/Rafi/My_Study/4_1/AI_Lab/aclImdb/test/mixed/posn.txt', 'r') as infile:
    for line in infile:
        filenames.append(line.rstrip())

filenamesneg = []
with open('F:/Rafi/My_Study/4_1/AI_Lab/aclImdb/test/mixed/negn.txt', 'r') as infile:
    for line in infile:
        filenamesneg.append(line.rstrip())


test = []
t_labels = []
doc = ""
for filename in filenames:
    s = "F:/Rafi/My_Study/4_1/AI_Lab/aclImdb/test/mixed/pos/" + filename
    with open(s, 'r') as infile2:
        for line in infile2:
            doc += line
        test.append(doc)
        t_labels.append(1)
        doc = ""

doc = ""
for filename in filenamesneg:
    s = "F:/Rafi/My_Study/4_1/AI_Lab/aclImdb/test/mixed/neg/" + filename
    with open(s, 'r') as infile2:
        for line in infile2:
            doc += line
        test.append(doc)
        t_labels.append(0)
        doc = ""


from sklearn.feature_extraction.text import TfidfVectorizer
tf = TfidfVectorizer(analyzer='word', ngram_range=(1,2), min_df = 0, max_features = 100000, stop_words = 'english')
tfidf_matrix = tf.fit_transform(corpus)
new_doc_tfidf_matrix = tf.transform(test)
print(tfidf_matrix.shape)
training = tfidf_matrix[:11930]
l_train = labels[:11930]

from sklearn import svm
from sklearn.metrics import accuracy_score

model = svm.SVC(kernel='rbf', C=2, gamma=1) 
model.fit(training, l_train)
predicted = model.predict(new_doc_tfidf_matrix)
print(accuracy_score(t_labels,predicted))
print(predicted)



