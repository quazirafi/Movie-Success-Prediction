from os import listdir
from os.path import isfile, join
mypath="F:\\Rafi\\My_Study\\4_1\\AI_Lab\\Python_Codes\\RuleBasedClassifier"
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
print(onlyfiles)