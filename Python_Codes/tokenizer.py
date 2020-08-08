from collections import defaultdict
import csv
episodes = defaultdict(list)
labels = []
with open("F:/Rafi/My_Study/4_1/AI_Lab/Sentiment Analysis Dataset Small.csv", "r") as sentences_file:
    reader = csv.reader(sentences_file, delimiter=',')
    reader.next()
    for row in reader:
        episodes[row[0]].append(row[3])		
f = open("F:/Rafi/My_Study/4_1/AI_Lab/Sentiment Analysis Dataset Small.csv")
for row in csv.reader(f):
    labels.append(row[1])
del labels[0]
# print(len(labels))
# for i in labels:
# 	print(i)
for episode_id, text in episodes.iteritems():
    episodes[episode_id] = "".join(text)
corpus = []
for id, episode in sorted(episodes.iteritems(), key=lambda t: int(t[0])):
    corpus.append(episode)

import re
text = ''.join(open('test.txt').readlines())
sentences = re.split(r' *[\.\?!][\'"\)\]]* *', text)
f_sentences = []
testing_set = []
for s in sentences:
	if not s:
		continue
	f_sentences.append(s)
print(len(f_sentences))
for s in f_sentences:
	print(s+"\n")
	testing_set.append(s)
print(len(testing_set))

from sklearn.feature_extraction.text import TfidfVectorizer
tf = TfidfVectorizer(analyzer='word', ngram_range=(1,2), min_df = 0, stop_words = 'english')
tfidf_matrix = tf.fit_transform(corpus)
new_doc_tfidf_matrix = tf.transform(testing_set)
print(tfidf_matrix.shape)
training, test = tfidf_matrix[:8989], tfidf_matrix[8980:]
l_train = labels[:8989]

from sklearn import svm
from sklearn.metrics import accuracy_score

model = svm.SVC(kernel='rbf', C=2, gamma=1) 
model.fit(training, l_train)
# # #model.score(tfidf_matrix, labels) //this is not needed
# # #Predict Output
predicted = model.predict(new_doc_tfidf_matrix)
print(predicted)



