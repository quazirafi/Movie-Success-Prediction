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
                print("%s %f"%(word,polarityScore))
            polarities.append(polarityScore)
        return [[x] for x in polarities]
X_test = ['nice day in nyc',
            'welcome to london',
            'london is rainy',
            'it is raining in britian',
            'it is raining in britian and the big apple',
            'it is raining in britian and nyc',
            'hello welcome to new york. enjoy it here and london too']
classifier = joblib.load('classifier.pkl')
predicted = classifier.predict(X_test)
for sen,pred in zip(X_test,predicted):
    print("%s --> %d"%(sen,pred))