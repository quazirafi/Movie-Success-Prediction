import codecs
from collections import defaultdict
import csv
episodes = defaultdict(list)
with open("Test2.txt", "r") as sentences_file:
    reader = csv.reader(sentences_file, delimiter=' ')
    reader.next()
    i = 0
    for row in reader:
        episodes[row[0]].append(row[2])
totalVal = 0.0
values = episodes['Rafi']
for val in values:
	totalVal += float(val)
print(totalVal)