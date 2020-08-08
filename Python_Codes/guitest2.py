from Tkinter import *
def printit():
	from collections import defaultdict
	import csv
	corpus = []
	labels = []

	filenames = []
	with open('F:/Rafi/My_Study/4_1/AI_Lab/Python_Codes/Version1/pos2.txt', 'r') as infile:
    	for line in infile:
    		filenames.append(line.rstrip())
	doc = ""
	for filename in filenames:
    	s = "F:/Rafi/My_Study/4_1/AI_Lab/aclImdb/train/pos/" + filename
    	with open(s, 'r') as infile2:
        	for line in infile2:
            	doc += line
        	corpus.append(doc)
        	labels.append(1)
    	    doc = ""
	print(len(corpus))
	doc = ""
	filenames = []
	with open('F:/Rafi/My_Study/4_1/AI_Lab/Python_Codes/Version1/neg2.txt', 'r') as infile:
    	for line in infile:
    	    filenames.append(line.rstrip())
	doc = ""
	for filename in filenames:
    	s = "F:/Rafi/My_Study/4_1/AI_Lab/aclImdb/train/neg/" + filename
    	with open(s, 'r') as infile2:
        	for line in infile2:
            	doc += line
        	corpus.append(doc)
        	labels.append(0)
    	    doc = ""
	print(len(corpus))
	print(len(labels))

	filenames = []
	with open('F:/Rafi/My_Study/4_1/AI_Lab/aclImdb/test/mixed/posnfull.txt', 'r') as infile:
    	for line in infile:
    	    filenames.append(line.rstrip())

	filenamesneg = []
	with open('F:/Rafi/My_Study/4_1/AI_Lab/aclImdb/test/mixed/negnfull.txt', 'r') as infile:
    	for line in infile:
    	    filenamesneg.append(line.rstrip())


	from sklearn.feature_extraction.text import TfidfVectorizer
	tf = TfidfVectorizer(analyzer='word', ngram_range=(1,2), min_df = 0, max_features = 100000, stop_words = 'english')
	tfidf_matrix = tf.fit_transform(corpus)
	print(tfidf_matrix.shape)
	training = tfidf_matrix[:3400]
	l_train = labels[:3400]

	from sklearn import svm
	from sklearn.metrics import accuracy_score

	model = svm.SVC(kernel='rbf', C=2, gamma=1) 
	model.fit(training, l_train)

	test = []
# for filename in filenames:
	s = enterstr.get()
# print(filename+"\n")
	with open(s, 'r') as infile2:
    	for line in infile2:
        	print(line+"\n")
        # new_doc_tfidf_matrix = tf.transform(line)
    	    test.append(line)
        # predicted = model.predict(new_doc_tfidf_matrix)

	new_doc_tfidf_matrix = tf.transform(test)
	predicted = model.predict(new_doc_tfidf_matrix)
	print(predicted)
	counter = 0
	for i in range(len(predicted)):
    	print(predicted[i])
    	if predicted[i] == 1:
    	    counter+=1
	print('percentage of positive comments ')
	print(counter)
	print(len(predicted))
	print((float(counter)/float(len(predicted)))*100.0)

w = Tk()
w.geometry('300x400+700+150')
enterstr = StringVar()
txttable = Entry(w,textvariable = enterstr).grid(row=5,column=6)
btn1 =  Button(w,text='click here',command=printit).grid(row=7,column=6)
w.mainloop()



# for filename in filenames:
# s = "F:/Rafi/My_Study/4_1/AI_Lab/POSTESTDATA/gravity.txt"
# print(filename+"\n")



