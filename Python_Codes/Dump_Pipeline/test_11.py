from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn import preprocessing
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
from collections import defaultdict
import re

episodes = defaultdict(list)
episodes = joblib.load('episodes.pkl')

class polarityScoreCalculator(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        polarities = []
        for doc in X:
            polarityScore = 0.0
            for word in re.findall(r'\w+', doc):
                values = episodes[word.lower()]
                for val in values:
                    polarityScore += float(val)
            polarities.append(polarityScore)
        return [[x] for x in polarities] 

path = r'F:\Rafi\My_Study\4_1\AI_Lab\aclImdb\train'
dataset = load_files(path, shuffle= False, decode_error='ignore', random_state=None,load_content=True)
trainData,testData,trainTarget,testTarget = train_test_split(dataset.data,dataset.target,train_size  = 0.5, test_size=0.5,random_state=42);
classifier = Pipeline([
    ('feats', FeatureUnion([
       ('tfidf', TfidfVectorizer(analyzer='word', ngram_range=(1,2), min_df = 0, max_features = 100000, stop_words = 'english')),
       ('pol', polarityScoreCalculator())
    ])),
    ('clf', svm.SVC(kernel='rbf', C=2, gamma=1))
])
classifier.fit(trainData,trainTarget)
predicted = classifier.predict(testData)
print(accuracy_score(testTarget, predicted)) 
joblib.dump(classifier,'classifier.pkl')