from collections import defaultdict
import csv
from sklearn.metrics import confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.datasets import load_files
from sklearn.naive_bayes import MultinomialNB
from sklearn.cross_validation import train_test_split
from sklearn import metrics
path = r'F:\\Rafi\\My_Study\\4_1\\AI_Lab\\aclImdb\\train'

dataset = load_files(path, shuffle= False, decode_error='ignore', random_state=None,load_content=True)
trainData,testData,trainTarget,testTarget = train_test_split(dataset.data,dataset.target,train_size  = 0.5, test_size=0.5,random_state=42);

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
from sklearn.metrics import classification_report
tf = TfidfVectorizer(analyzer='word', ngram_range=(1,2), min_df = 0, max_features = 100000, stop_words = 'english')
tfidf_matrix = tf.fit_transform(trainData)
print(tfidf_matrix.shape)
model = svm.SVC(kernel='rbf', C=2, gamma=1) 
model.fit(tfidf_matrix, trainTarget)
import pickle
from sklearn.externals import joblib
new_doc_tfidf_matrix = tf.transform(testData)
predicted = model.predict(new_doc_tfidf_matrix)
print(accuracy_score(testTarget, predicted)) 
from sklearn.metrics import confusion_matrix
target_names = ['negative', 'positive']



print(confusion_matrix(testTarget, predicted, labels=[0,1]))
print(classification_report(testTarget, predicted, target_names=target_names))
