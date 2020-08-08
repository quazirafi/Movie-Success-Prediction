import pickle
from sklearn.externals import joblib
import codecs
from collections import defaultdict
import csv
import os
episodes = defaultdict(list)
episodes = joblib.load('episodes.pkl')
fileNames = os.listdir('F:\\Rafi\\My_Study\\4_1\\AI_Lab\\aclImdb\\train\\pos')
print(len(fileNames))
for fileName in fileNames:
	print(fileName)
	break
import re
totalPolarity = 0.0
for fileName in fileNames:
	s = ""
	s = "F:\\Rafi\\My_Study\\4_1\\AI_Lab\\aclImdb\\train\\pos\\"
	with open(s+fileName) as f:
		polarityScore = 0.0
		for line in f:
			for word in re.findall(r'\w+', line):
				values = episodes[word.lower()]
				for val in values:
					polarityScore += float(val)
		# print("%s = %f"%(fileName,polarityScore))
		totalPolarity+=polarityScore
print("avg ploraity = %f"%(totalPolarity/21466.0))