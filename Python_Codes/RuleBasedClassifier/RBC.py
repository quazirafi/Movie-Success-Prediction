import pickle
from sklearn.externals import joblib
import codecs
from collections import defaultdict
import csv
import sys
import csv
csv.field_size_limit(500000)
episodes = defaultdict(list)
#build the dictionary
with open("unigrams-pmilexicon.txt", "r") as sentences_file:
    reader = csv.reader(sentences_file, delimiter='\t')
    reader.next()
    i = 0
    for row in reader:
        episodes[row[0].lower()].append(row[1])
joblib.dump(episodes,'episodes.pkl')
# episodes = joblib.load('episodes.pkl')
# fileNames = ['m2.txt']
# import re
# for fileName in fileNames:
# 	with open(fileName) as f:
# 		polarityScore = 0.0
# 		for line in f:
# 			for word in re.findall(r'\w+', line):
# 				values = episodes[word.lower()]
# 				for val in values:
# 					polarityScore += float(val)
# print(polarityScore)