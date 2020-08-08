from collections import defaultdict
import csv
from sklearn.metrics import confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.datasets import load_files
from sklearn.naive_bayes import MultinomialNB
from sklearn.cross_validation import train_test_split
from sklearn import metrics
from sklearn.metrics import confusion_matrix
y_true = [1,2,1,1,2]
y_pred = [1,1,2,1,2]
print(confusion_matrix(y_true, y_pred, labels=[1,2]))
