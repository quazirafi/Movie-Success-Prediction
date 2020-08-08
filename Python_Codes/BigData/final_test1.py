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
'12AngryMen8.9.txt',
'2001 A Space Odyssey.txt',
'2012.txt',
'3 Idiots.txt',
'A Clockwork Orange.txt',
'Alien.txt',
'Aliens.txt',
'AlvinandtheChipmunks.txt',
'AlvinandtheChipmunksThe Squeakquel.txt',
'American Beauty.txt',
'American History X8.5.txt',
'Amlie.txt',
'Apocalypse Now8.5.txt',
'Back to the Future8.5.txt',
'Bicycle Thieves.txt',
'Braveheart.txt',
'Casablanca8.5.txt',
'Cinema Paradiso.txt',
'Citizen Kane.txt',
'City Lights8.6.txt',
'City of God8.7.txt',
'clasoftitans.txt',
'Dangal.txt',
'Das Boot.txt',
'Django Unchained.txt',
'Double Indemnity.txt',
'Dr. Strangelove or How I Learned to Stop Worrying and Love the Bomb.txt',
'Eternal Sunshine of the Spotless Mind.txt',
'FightClub8.8.txt',
'ForrestGump8.7.txt',
'Full Metal Jacket.txt',
'Gladiator8.5.txt',
'Goodfellas8.7.txt',
'Grave of the Fireflies.txt',
'HowtheGrinchStoleChristmas.txt',
'Inception8.8.txt',
'Inglourious Basterds.txt',
'Intersteller8.5.txt',
'Its a Wonderful Life8.6.txt',
'Lawrence of Arabia.txt',
'Life Is Beautiful8.6.txt',
'Like Stars on Earth.txt',
'Lon- The Professional8.6.txt',
'M.txt',
'Memento8.5.txt',
'Modern Times8.5.txt',
'Monty Python and the Holy Grail.txt',
'newmoon.txt',
'North by Northwest.txt',
'Oldboy.txt',
'Once Upon a Time in America.txt',
'Once Upon a Time in the West8.6.txt',
'One Flew Over the Cuckoos Nest8.7.txt',
'PhoenixForgotten.txt',
'Princess Mononoke.txt',
'Psycho8.5.txt',
'PulpFiction8.9.txt',
'Raiders of the Lost Ark8.5.txt',
'Rear Window8.5.txt',
'Requiem for a Dream.txt',
'Reservoir Dogs.txt',
'Saving Private Ryan8.6.txt',
'SchindlersList8.9.txt',
'Se7en8.6.txt',
'SevenSamurai8.6.txt',
'ShawshankRedemption9.3.txt',
'Singin in the Rain.txt',
'Snatch.txt',
'Spirited Away8.5.txt',
'Star Wars Episode VI - Return of the Jedi.txt',
'Star Wars- Episode IV - A New Hope8.7.txt',
'Star Wars- Episode V - The Empire Strikes Back8.8.txt',
'Sunset Blvd..txt',
'Taxi Driver.txt',
'Terminator 2- Judgment Day8.5.txt',
'The Dark Knight Rises.txt',
'The Departed8.5.txt',
'The Good, the Bad and the Ugly8.9.txt',
'The Great Dictator.txt',
'The Green Mile8.5.txt',
'The Intouchables8.6.txt',
'The Kid.txt',
'The Lion King8.5.txt',
'The Lives of Others.txt',
'The Lord of the Rings- The Fellowship of the Ring8.8.txt',
'The Lord of the Rings- The Return of the King8.9.txt',
'The Lord of the Rings- The Two Towers8.7.txt',
'The Matrix8.7.txt',
'The Pianist8.5.txt',
'The Prestige8.5.txt',
'The Shining.txt',
'The Silence of the Lambs8.6.txt',
'The Sting.txt',
'The TwilightSagaBreakingDawn - Part.txt',
'The TwilightSagaBreakingDawn -Part2.txt',
'The Usual Suspects8.6.txt',
'TheDarkKnight9.txt',
'TheGodfather9.2.txt',
'TheGodfatherPartTwo9.0.txt',
'threemenandababy.txt',
'To Kill a Mockingbird.txt',
'top_graph_data.txt',
'Toy Story 3.txt',
'Toy Story.txt',
'TransformersAgeofExtinction.txt',
'TransformersRevengeoftheFallen.txt',
'Vertigo.txt',
'WALLE.txt',
'Whiplash8.5.txt',
'wildogs.txt',
'Witness for the Prosecution.txt',
'Your Name.txt',
'10,000BC.txt',
'A Story About Love.txt',
'aebastofyuccaflats.txt',
'afoxtale2.2.txt',
'AmericasSweethearts.txt',
'Anne B. Real.txt',
'Astoryaboutlove.txt',
'Baby Geniuses.txt',
'Battlefield Earth.txt',
'benarthur.txt',
'birdemi21.8.txt',
'birdemic1.8.txt',
'Body in the Web.txt',
'BodyintheWeb.txt',
'Boggy Creek II And the Legend Continues.txt',
'Breaking Wind.txt',
'Car 54, Where Are You.txt',
'Car54WhereAreYou.txt',
'Cats&Dogs.txt',
'Chairman of the Board.txt',
'crossover.txt',
'Danes Without a Clue.txt',
'DanesWithoutaClue.txt',
'danywizard1.9.txt',
'Devil Fish.txt',
'DevilFish.txt',
'disastermovie1.8.txt',
'Dragonball Evolution.txt',
'DragonballEvolution.txt',
'dream2.1.txt',
'Ed.txt',
'epicmovie.txt',
'Fat Slags.txt',
'Fatslags.txt',
'Flubber.txt',
'Ghosts Cant Do It.txt',
'GhostsCantDoIt.txt',
'Gigli.txt',
'Girl in Gold Boots.txt',
'glitter.txt',
'goingboard.txt',
'greenhornet.txt',
'Heaven Is for Real.txt',
'hobgoblins.txt',
'houseofdead.txt',
'humshakals.txt',
'I Accuse My Parents.txt',
'IAccuseMyParents.txt',
'indianflames.txt',
'inspectorgadget.txt',
'Invasion of the Neptune Men.txt',
'Its Pat The Movie.txt',
'ItsPat.txt',
'jurassicpark.txt',
'keloglanvsbi.txt',
'KindergartenCop.txt',
'Kyaa Kool Hain Hum 3.txt',
'Laserblast.txt',
'Lawnmower Man 2 Beyond Cyberspace.txt',
'MaidinManhattan.txt',
'manos1.9.txt',
'Meet the Spartans.txt',
'MeettheSpartans.txt',
'michael.txt',
'Mitchell.txt',
'mostaragogo.txt',
'nighttraintomundofine.txt',
'objective.txt',
'photosalad.txt',
'pledge1.9.txt',
'Prince of Space.txt',
'Ride Along 2.txt',
'Santa Claus Conquers the Martians.txt',
'Santa Claus.txt',
'Santa with Muscles.txt',
'SantaClaus.txt',
'ScaryMovie4.txt',
'sexandcity.txt',
'Simon Sez.txt',
'Smolensk.txt',
'sonofthemask.txt',
'Soultaker.txt',
'spacemutiny.txt',
'surfschool.txt',
'Survival Island.txt',
'SurvivalIsland.txt',
'Tees Maar Khan.txt',
'The Aztec Mummy Against the Humanoid Robot.txt',
'The Bat People.txt',
'The Blade Master.txt',
'The Incredibly Strange Creatures Who Stopped Living and Became Mixed-Up Zombies!!.txt',
'The Maize The Movie.txt',
'The Oogieloves in the Big Balloon Adventure.txt',
'The Pumaman.txt',
'The Skydivers.txt',
'The Touch of Satan.txt',
'TheAztecMummyAgainsttheHumanoidRobot.txt',
'TheBatPeople.txt',
'thebeastofyuccaflats.txt',
'TheHaunting.txt',
'Theodore Rex.txt',
'TheodoreRex.txt',
'ThePrincessDiaries2RoyalEngagement.txt',
'Time Chasers.txt',
'TimeChasers.txt',
'Too Beautiful.txt',
'Track of the Moon Beast.txt',
'turkinspace.txt',
'Twilight.txt',
'Warrior of the Lost World.txt',
'whoscaddy.txt',
'yessir.txt',
'Zaat.txt',
'Zombie Nightmare.txt',
]
tf = joblib.load('tfidf_matrix.pkl') 
counter2 = 0
import codecs
graph_points = []
index = 0
target = []
for i in range(112):
	target.append(1)
for i in range(114):
	target.append(0)
for threshold in range(50,101,5):
	correct_prediction = []
	for filename in filenames:
		test = []
		s = "F:\\Rafi\\My_Study\\4_1\\AI_Lab\\AIData\\Graph_Plot\\mixed\\"+filename
		with codecs.open(s, "r",encoding='utf-8', errors='ignore') as infile2:
			for line in infile2:
				test.append(line)
		new_doc_tfidf_matrix = tf.transform(test)
		predicted = []
		predicted = model.predict(new_doc_tfidf_matrix)
		counter = 0
		for i in range(len(predicted)):
   			if predicted[i] == 1:
   				counter+=1
		percentage_pos = (float(counter)/float(len(predicted)))*100.0
		if percentage_pos>=float(threshold):
			correct_prediction.append(1)
		else:
			correct_prediction.append(0)
	counter3 = 0
	print(len(correct_prediction))
	print(len(target))
	for i in range(226):
		if correct_prediction[i] == target[i]:
			counter3+=1
	graph_points.append((threshold,float((float(counter3)/float(226))*100.0)))
	index+=1
import sys
import matplotlib.pyplot as plt
ths = []
pres = []
sys.stdout = open("F:\\Rafi\\My_Study\\4_1\\AI_Lab\\AIData\\Graph_Plot\\graph_data.txt", "w")
for th,pre in graph_points:
	sys.stdout.write("%d %f\n"%(th,pre))
# plt.xlim([0,100])
# plt.ylim([0,100])
# import numpy as np
# import matplotlib.pyplot as plt

# N = 51


# ind = np.arange(N)  # the x locations for the groups
# width = 0.03       # the width of the bars

# fig, ax = plt.subplots()
# rects1 = ax.bar(ind, pres, width, color='r')


# # add some text for labels, title and axes ticks
# ax.set_ylabel('Scores')
# ax.set_title('Scores by group and gender')
# ax.set_xticks(ind + width / 2)
# ax.set_xticklabels(ind)
# ax.legend((rects1), ('Correctly Classified'))
# plt.show()
# plt.plot(ths,pres)	
# plt.show()