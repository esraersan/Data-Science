from math import log
import sys


#run with \DT.py dt_train.txt dt_test.txt MinSizeOfDataset

	
def ParentEntropy(data):
	#number of instances in the dataset
	RowCounts = {}

	#create a dict
	for featureVec in data:
		currentRow = featureVec[-1]
		if currentRow not in RowCounts.keys():
			RowCounts[currentRow] = 0
			RowCounts[currentRow] += 1

	entropy = 0.0
	for key in RowCounts:
	#probability of instance in the dataset
			prob = float(RowCounts[key] / len(data)) #bu kısımdan bir daha emin ol
	#calculate the entropy
	entropy -= prob * log(prob, 2) #base 2

	return entropy	

def InfoGainForRoot(data,features,value):
	#i have to use info gain here also so i can decide the root node that i will start splitting
	#here we take 3 input 1.data to be split on 2.the feature to be split on 3.and the value of the feature to return
	#create new list in the beginning each time because we will be calling this function multiple times on the same dataset and we don't want the original dataset to be modified
	#inside the if statement we cut the feature that we split on
	subDataset = {}
	valueIndex = features.index(value)
	newEntropy = 0
	for featureVec in data:
		if featureVec[valueIndex] in subDataset.keys():
			subDataset[featureVec[valueIndex]][0] += 1
			subDataset[featureVec[valueIndex]][1].append(featureVec)

		else:
			subDataset[featureVec[valueIndex]]=[1,[featureVec]]

	for featureValues in subDataset.keys():
		prob = subDataset[featureValues][0]/len(data)
		trainingSubset = subDataset[featureValues][1]
		newEntropy += prob * ParentEntropy(trainingSubset)

	InformationGain = ParentEntropy(trainingSubset) - newEntropy

	return (InformationGain, None)

def SubsetOfTrainingData(data,features,value,valueOutput,limit):
	valueIndex = features.index(value)
	subset_trainingdata = []

	if(type(limit)is bool and limit==True):
		for featureVec in data:
			if (featureVec[valueIndex] == valueOutput):
				rows = []
				for i in range (len(featureVec)):
					if(i !=valueIndex):
						rows.append(featureVec[i])
				subset_trainingdata.append(rows)
	else:
		threshold = limit

		for featureVec in data:
			if(((valueOutput == "<=") and (featureVec[valueIndex]) <= threshold)) or ((valueOutput == ">") and (featureVec[valueIndex]) > threshold):
				subset_trainingdata.append(featureVec)

	return subset_trainingdata


def CandidateAttributes(data,valueIndex):
	Valuelist = []
	Classifiers = {}
	candidate_attributes = []
	for featureVec in data:
		if float(featureVec[valueIndex]) not in Valuelist:
			Valuelist.append(featureVec[valueIndex])
			Classifiers[featureVec[valueIndex]] = {}
			Classifiers[featureVec[valueIndex]][featureVec[-1]] = 1
			if featureVec[-1] == feature_Values['class'][0]:
				unclassified_ones = feature_Values['class'][1]
			else:
				unclassified_ones = feature_Values['class'][0]
			Classifiers[featureVec[valueIndex]][unclassified_ones] = 0

		else:
			Classifiers[featureVec[valueIndex]][featureVec[-1]] += 1

	Valuelist.sort()
	label1 = feature_Values['class'][0]
	label2 = feature_Values['class'][1]		

#finding the candidate threshold between changed values of attributes 
	for i in range (0,len(Valuelist)-1):
		currentValue = Valuelist[i]
		nextValue = Valuelist[i+1]

		if (Classifiers[currentValue][label1] > 0 and Classifiers[nextValue][label2] > 0) or (Classifiers[currentValue][label2] > 0 and Classifiers[nextValue][label1] > 0):
			CandidateAttributes.append((currentValue + nextValue)/2.0)

	return candidate_attributes

def InformationGain(data,features,value):
	valueIndex = features.index(value)
	candidate_attributes = CandidateAttributes(data,valueIndex)
	candidate_infogain = []

	for attribute in candidate_attributes:
		subDataset = {'feature':[0,[]],'value':[0,[]]} #not sure about here
		newEntropy = 0
		for featureVec in data:
			if featureVec[valueIndex] <= attribute:
			   subDataset['feature'][0] += 1
			   subDataset['feature'][1].append(featureVec)
			else:
				subDataset['value'][0] += 1
				subDataset['value'][1].append(featureVec)

	for k in subDataset.keys():
		prob = subDataset[k][0]/len(data)
		trainingSubset = subDataset[k][1]
		newEntropy += prob * ParentEntropy(trainingSubset)
	candidate_infogain.append(newEntropy)

	if (len(candidate_attributes) != 0):
		parent_entropy = ParentEntropy(data)
		candidate_infogain_list = [(parent_entropy - entropy) for entropy in candidate_infogain]
		selected_infogain = max(candidate_infogain_list) #our goal is to maximize the infogain
		selected_attribute = candidate_attributes[candidate_infogain_list.index(selected_infogain)]
	else:
		
		selected_infogain = None
		selected_attribute = None

	return(selected_attribute,selected_infogain)
 

def ChooseBestFeature(data,features):
	
	Highest_InformationGain = 0.0
	BestFeature = None
	BestAttribute = None

	for f in features:
		if(type(feature_Values[f]) is list):
			current_infogain,current_attribute = InfoGainForRoot(data,features,f)
		else:
			current_infogain,current_attribute = InformationGain(data,features,f)
		if(current_infogain != None):
			if(current_infogain > Highest_InformationGain):
				Highest_InformationGain = current_infogain
				BestFeature = f	
				BestAttribute =current_attribute #this is actually kinda threshold but lets say attribute

	return (BestFeature,BestAttribute)    

#MinSizeOfDataset to deal with every dataset with different sizes 
#here i used the random forest algorithm's feature importance
#Through looking at the feature importance, you can decide which features you may want to drop,
#because they don’t contribute enough or nothing to the prediction process.
#This is important, because a general rule in machine learning is that the more features you have, the more likely your model will suffer from overfitting and vice versa.
def CreateDecisionTree(data,features,MinSizeOfDataset,RecursionIndex=-1):
	global branch
	RecursionIndex += 1
	classifications = []
	for featureVec in []:
		classifications.append(featureVec[-1])

	importantFeatures = classifications.count(features_values)
	redundantFeatures = len(classifications) - importantFeatures
	node = " ["+str(importantFeatures)+" "+str(redundantFeatures)+"]"
	branch = branch[:-1]
	if RecursionIndex !=0:
		branch += node +"\n"

	if (len([]) == 0):
		branch = branch[:-1]
		branch += ": "+features_values['class'][0]+"\n"
		return features_values['class'][0]
	elif len(classifications) == classifications.count(classifications[0]):
		branch = branch[:-1]
		branch += ": "+classifications[0]+"\n"
		return classifications[0]

	elif (len(features) == 0) or (len(data) < MinSizeOfDataset):
		if (importantFeatures >= redundantFeatures):
			branch = branch[:-1]
			branch += ": "+features_values['class'][0]+"\n"
			return features_values['class'][0]

		elif (importantFeatures < redundantFeatures):
			branch = branch[:-1]
			branch += ": "+features_values['class'][1]+"\n"
			return features_values['class'][1]
	else:
		BestAttribute,BestSplit = ChooseBestFeature(data,features)
		if(BestSplit is None):
			if(importantFeatures >= redundantFeatures):
				branch = branch[:-1]
			branch += ": "+features_values['class'][0]+"\n"
			return features_values['class'][0]
		elif(importantFeatures < redundantFeatures):
			branch = branch[:-1]
			branch += ": "+features_values['class'][1]+"\n"
			return features_values['class'][1]
		else:
			myTree = {BestSplit:{}}
			if BestAttribute is None:
				BestFeature_results = features_values[BestSplit]
				for results in BestFeature_results:
					for j in BestFeature_results:
						branch +="/      "
					branch += BestSplit+" = "+results+'\n'
					trainingSubset = SubsetOfTrainingData(data,features,BestSplit,results,True)
					SubFeatures = features[:]
					SubFeatures.remove(BestSplit)
					SubTree = CreateDecisionTree(trainingSubset,SubFeatures,MinSizeOfDataset,RecursionIndex)
					myTree[BestSplit][results] = SubTree
			else: 
				BestFeature_results = ["<=", ">"]
				for results in BestFeatue_results:
					for j in range(0,RecursionIndex):
						branch += "/      "
				branch += BestSplit+" "+results+" "+str("%.6f"%BestAttribute)+'\n'
				trainingSubset = SubsetOfTrainingData(data,features,BestSplit,results,BestAttribute)
				SubFeatures = features[:]
				SubTree = CreateDecisionTree(trainingSubset,SubFeatures,MinSizeOfDataset,RecursionIndex)
				results = results+" "+str(BestAttribute)
				myTree[BestSplit][results] = SubTree

	return myTree

def Classification(myTree,TestData):
	predictions = 0
	TotalInstances = 0

	print("Predictions For Testing Dataset")
	for i in range (0,len(TestData)):
		test_Row = TestData[i]
		if type (myTree) is dict:
			currentTree = myTree.copy()
			label = ""
			while(type(currentTree) is dict):
				node = list(currentTree.keys())[0]
				nodeIndex = FeaturesforClassify.index(node)
				nodeValue = test_Row[nodeIndex]
				currentTree = currentTree[node]
				leafKeys = list(currentTree.keys())
				if '<=' in leafKeys[0] or '>' in leafKeys[0]:
					#eval() interprets a string as code
					if(eval(nodeValue+leafKeys[0])):
						label = currentTree[leafKeys[0]]
						currentTree =currentTree[leafKeys[0]]
					else:
						label = currentTree[leafKeys[1]]
						currentTree =currentTree[leafKeys[1]]
				elif (nodeValue in list(currentTree.keys())):
					label = currentTree[nodeValue]
					currentTree = currentTree[nodeValue]
				else:
					print("Error")
		else:
			label = myTree
			print("%3d: Original: "%(i+1)+test_Row[-1]+"  Predicted: "+label)
		if label == test_Row[-1]:
			predictions += 1
		else:
			TotalInstances += 1

	print ("True Classifications: "+str(predictions)+"  Total Number of Tested Rows: "+str(predictions+TotalInstances))
	return (predictions,TotalInstances)

def ReadFile(file,FileIndex=False):
	fp = open(file,"r")
	lines = fp.read().split("\n")
	lines = lines[1:] 
	lines = [line for line in lines if line!='']
	dataset = []
	if(FileIndex):
		FeaturesforClassify = []
		features_values = {}
		for line in lines:
			if line.startswith("age") or line.startswith("buying"):
				line_parts =line.split(' ')
				FeaturesforClassify.append(eval(line_parts[1]))
				if len(line_parts) <4:
					features_values[eval(line_parts[1])] = line_parts[-1]	
				else:
					features_values[eval(line_parts[1])] = []
					for index in range(3,len(line_parts)):
						features_values[eval(line_parts[1])].append(line_parts[index][:-1])
			elif not line.startswith("age") or line.startswith("buying"):
				dataset.append(line.split(','))
				while(FeaturesforClassify.count('class')):
					FeaturesforClassify.remove('class')
					ReturnData = (FeaturesforClassify,features_values,dataset)
			

	else:
		for line in lines:
			if not line.startswith("age") or line.startswith("buying"):
				dataset.append(line.split(','))
				ReturnData = dataset

		return ReturnData 

if __name__ == "__main__" :

	if len(sys.argv)<2:
		print ("Fatal: You forgot to include the directory name on the command line.")
		print ("Usage:  python %s <directoryname>" % sys.argv[0])
		sys.exit(1)

training_set = str(sys.argv[1])
testing_set = str(sys.argv[2]) #this is for while runnig the command and entering the file names
MinSizeOfDataset = int(sys.argv[3])
branch=""
FeaturesforClassify = ReadFile(training_set,True) 
features_values = ReadFile(training_set,True) 
data = ReadFile(training_set,True) 
tree = CreateDecisionTree(data,FeaturesforClassify,MinSizeOfDataset)

if type(tree) is str:
		branch = branch[2:]
test_data = read_file(testing_set)
print ("")
print (branch[:-1])
Classification(tree, test_data)
	






		





 