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
from sklearn.feature_extraction.text import TfidfVectorizer
tf = TfidfVectorizer(analyzer='word', ngram_range=(1,2), min_df = 0, stop_words = 'english')
tfidf_matrix = tf.fit_transform(corpus)
training, test = tfidf_matrix[:9000], tfidf_matrix[9000:]
l_train, l_test = labels[:9000],labels[9000:]
#feature_names = tf.get_feature_names()
#print(tfidf_matrix)
#SVM implementation starts from here
#Import Library
from sklearn import svm
from sklearn.metrics import accuracy_score
#Assumed you have, X (predictor) and Y (target) for training data set and x_test(predictor) of test_dataset
# Create SVM classification object 
model = svm.SVC(kernel='rbf', C=2, gamma=1) 
# there is various option associated with it, like changing kernel, gamma and C value. Will discuss more # about it in next section.Train the model using the training sets and check score
model.fit(training, l_train)
#model.score(tfidf_matrix, labels)
#Predict Output
predicted = model.predict(test)
print(accuracy_score(predicted, l_test))
