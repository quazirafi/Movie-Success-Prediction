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
from sklearn import metrics
from sklearn.metrics import classification_report
path = r'F:\\Rafi\\My_Study\\4_1\\AI_Lab\\aclImdb\\train'
dataset = load_files(path, shuffle= False, decode_error='ignore', random_state=None,load_content=True)
trainData,testData,trainTarget,testTarget = train_test_split(dataset.data,dataset.target,train_size  = 0.5, test_size=0.5,random_state=42);
model = joblib.load('comment_trainer.pkl') 
print("trainer_loaded")
import pickle
tf = joblib.load('tfidf_matrix.pkl') 
print("tf_idf_loaded")
new_doc_tfidf_matrix = tf.transform(testData)
predicted = model.predict(new_doc_tfidf_matrix)
print(accuracy_score(testTarget, predicted))
target_names = ['negative', 'positive']
print(classification_report(testTarget, predicted, target_names=target_names))
# from sklearn.model_selection import cross_val_score
# from sklearn.linear_model import LogisticRegression
# scores = cross_val_score(model, new_doc_tfidf_matrix, testTarget, cv=10)
# print(scores)