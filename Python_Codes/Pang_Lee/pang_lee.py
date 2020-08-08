from collections import defaultdict
import csv
from sklearn.metrics import confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.datasets import load_files
from sklearn.naive_bayes import MultinomialNB
from sklearn.cross_validation import train_test_split
from sklearn import metrics
from sklearn import metrics
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
path = r'F:\\Rafi\\My_Study\\4_1\\AI_Lab\\txt_sentoken'

dataset = load_files(path, shuffle= False, decode_error='ignore', random_state=None,load_content=True)
trainData,testData,trainTarget,testTarget = train_test_split(dataset.data, dataset.target, train_size  = 0.99, test_size = 0.01, random_state=42)

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
tf = TfidfVectorizer(analyzer='word', ngram_range=(1,2), min_df = 0, max_features = 30000, stop_words = 'english')
tfidf_matrix = tf.fit_transform(trainData)
print(tfidf_matrix.shape)
print(len(trainTarget))
model = svm.SVC(kernel='rbf', C=2, gamma=1) 
scores = cross_val_score(model, tfidf_matrix, trainTarget, cv=5)
print(scores)