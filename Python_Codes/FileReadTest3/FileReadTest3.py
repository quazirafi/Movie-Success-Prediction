import pickle
from sklearn.externals import joblib
import codecs
from collections import defaultdict
import csv
episodes = defaultdict(list)
# #build the dictionary
# with open("Test2.txt", "r") as sentences_file:
#     reader = csv.reader(sentences_file, delimiter=' ')
#     reader.next()
#     i = 0
#     for row in reader:
#         episodes[row[0].lower()].append(row[1])
# joblib.dump(episodes,'episodes.pkl')
episodes = joblib.load('episodes.pkl')
fileNames = ['m2.txt']
import re
for fileName in fileNames:
	with open(fileName) as f:
		polarityScore = 0.0
		for line in f:
			for word in re.findall(r'\w+', line):
				values = episodes[word.lower()]
				for val in values:
					polarityScore += float(val)
print(polarityScore)