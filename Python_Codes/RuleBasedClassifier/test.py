import pickle
from sklearn.externals import joblib
import codecs
from collections import defaultdict
import csv
import sys
import csv
csv.field_size_limit(500000)
episodes = defaultdict(list)
episodes = joblib.load('episodes.pkl')
target = open('test2.txt', 'w')
for k,v in episodes.iteritems():
	target.write("key = %s value = %s"%(k,v))
	target.write("\n")
target.close()
print(len(episodes))