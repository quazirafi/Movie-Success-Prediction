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
fileNames = [
'aebastofyuccaflats.txt',
'afoxtale2.2.txt',
'Astoryaboutlove.txt',
'benarthur.txt',
'birdemi21.8.txt',
'birdemic1.8.txt',
'BodyintheWeb.txt',
'Car54WhereAreYou.txt',
'crossover.txt',
'DanesWithoutaClue.txt',
'danywizard1.9.txt',
'DevilFish.txt',
'disastermovie1.8.txt',
'DragonballEvolution.txt',
'dream2.1.txt',
'epicmovie.txt',
'Fatslags.txt',
'GhostsCantDoIt.txt',
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
import re
totalPolarity = 0.0
for fileName in fileNames:
	s = ""
	s = "F:\\Rafi\\My_Study\\4_1\\AI_Lab\\AIData\\Test\\"
	with open(s+fileName) as f:
		polarityScore = 0.0
		for line in f:
			for word in re.findall(r'\w+', line):
				values = episodes[word.lower()]
				for val in values:
					polarityScore += float(val)
		print("%s = %f"%(fileName,polarityScore))
		totalPolarity+=polarityScore
print("avg ploraity = %f"%(totalPolarity/49.0))