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
model = joblib.load('comment_trainer.pkl') 
import pickle
from sklearn.externals import joblib
filenames = [
'aebastofyuccaflats.txt',
'afoxtale2.2.txt',
'Astoryaboutlove.txt',
'benarthur.txt',
'birdemi21.8.txt',
'birdemic1.8.txt',
'Car54WhereAreYou.txt',
'crossover.txt',
'DanesWithoutaClue.txt',
'danywizard1.9.txt',
'DevilFish.txt',
'disastermovie1.8.txt',
'DragonballEvolution.txt',
'dream2.1.txt',
'epicmovie.txt',
'glitter.txt',
'goingboard.txt',
'hobgoblins.txt',
'houseofdead.txt',
'humshakals.txt',
'IAccuseMyParents.txt',
'indianflames.txt',
'ItsPat.txt',
'jurassicpark.txt',
'keloglanvsbi.txt',
'Laserblast.txt',
'manos1.9.txt',
'MeettheSpartans.txt',
'Mitchell.txt',
'mostaragogo.txt',
'nighttraintomundofine.txt',
'photosalad.txt',
'pledge1.9.txt',
'SantaClaus.txt',
'sonofthemask.txt',
'spacemutiny.txt',
'surfschool.txt',
'SurvivalIsland.txt',
'TheAztecMummyAgainsttheHumanoidRobot.txt',
'TheBatPeople.txt',
'thebeastofyuccaflats.txt',
'TheodoreRex.txt',
'TimeChasers.txt',
'turkinspace.txt',
'whoscaddy.txt',
'yessir.txt',
]
tf = joblib.load('vectorizer.pkl') 
counter2 = 0
for filename in filenames:
	test = []
	s = "F:/Rafi/My_Study/4_1/AI_Lab/AIData/Test/"+filename
	with open(s, 'r') as infile2:
		for line in infile2:
			test.append(line)
	new_doc_tfidf_matrix = tf.transform(test)
	predicted = model.predict(new_doc_tfidf_matrix)
	counter = 0
	for i in range(len(predicted)):
   		if predicted[i] == 1:
   			counter+=1
	print('percentage of positive comments ')
	print((float(counter)/float(len(predicted)))*100.0)
	print('\n')
	counter2+=1
	print(counter2)
