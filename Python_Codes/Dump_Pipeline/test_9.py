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

class LengthTransformer(BaseEstimator, TransformerMixin):

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


X_train = ["new york is a hell of a town",
            "new york was originally dutch",
            "the big apple is great",
            "new york is also called the big apple",
            "nyc is nice",
            "people abbreviate new york city as nyc",
            "the capital of great britain is london",
            "london is in the uk",
            "london is in england",
            "london is in great britain",
            "it rains a lot in london",
            "london hosts the british museum",
            "new york is great and so is london",
            "i like london better than new york"]
y_train_text = [1,1,1,1,1,1,0,0,0,0,0,0,
                1,0]

X_test = ['nice day in nyc',
            'welcome to london',
            'london is rainy',
            'it is raining in britian',
            'it is raining in britian and the big apple',
            'it is raining in britian and nyc',
            'hello welcome to new york. enjoy it here and london too']
classifier = Pipeline([
    ('feats', FeatureUnion([
       ('tfidf', TfidfVectorizer()),
       ('len', LengthTransformer())
    ])),
    ('clf', svm.SVC(kernel='rbf', C=2, gamma=1))
])

classifier.fit(X_train, y_train_text)
predicted = classifier.predict(X_test)
for item, labels in zip(X_test, predicted):
    print("%s --> %d"%(item,labels))
joblib.dump(classifier,'classifier.pkl')