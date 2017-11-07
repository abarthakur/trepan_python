import numpy as np
import pandas


from scipy import stats

import queue as Q

###########################################



###########################################
def get_data(filename): # Satimage dataset
    data = pandas.read_csv(filename, sep=r"\s+", header=None)
    data = data.values

    dataX = np.array(data[:,range(data.shape[1]-1)])
    dataY = np.array(data[np.arange(data.shape[0]),data.shape[1]-1])

    # convert dataY to one-hot, 6 classes
    num_classes = 6
    dataY = np.array([x-2 if x==7 else x-1 for x in dataY]) # re-named class 7 to 6(5)
    dataY_onehot = np.zeros([dataY.shape[0], num_classes])
    dataY_onehot[np.arange(dataY_onehot.shape[0]), dataY] = 1

    return dataX, dataY_onehot

def createModel (trainX,trainY,num_classes):
	model = Sequential()
	model.add(Dense(16, input_dim=trainX.shape[1], activation="sigmoid"))
	model.add(Dense(16, activation="sigmoid"))
	model.add(Dense(num_classes, activation="softmax"))
	model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=['accuracy'])
	model.fit(trainX, trainY, epochs=20, batch_size=10) # epochs=150
	return model

###########################################

class Oracle:
	def __init__(self,network,num_classes):
		self.network=network
		self.num_classes=num_classes

	def setDistributions(self,X):
		self.distributions =[]
		#only consider continuous distributions
		self.dimension= X.shape[1]
		for i in range(0,self.dimension):
			values = X[:,i].reshape(X.shape[0])
			kernel = stats.gaussian_kde(values,bw_method='silverman')
			# print(kernel)
			# print(kernel.resample(1))
			self.distributions.append(kernel)

	def oracle_example(self,example):
		#print(example.shape)
		onehot =self.network.predict(np.array([example])).reshape(self.num_classes)
		# print(onehot)
		return np.argmax(onehot)

	def oracle_constraints(self,constraints,n):
		#read each constraint
		X_examples= np.zeros((n,self.dimension))
		lab_examples = np.zeros(n)
		num_valid=0
		print(n)
		while(num_valid<n):
			sample=self.genSample()
			if (self.validSample(sample,constraints)):
				label=self.oracle_example(sample)
				X_examples[num_valid,:]=sample
				lab_examples[num_valid]=label
				num_valid+=1
				print(num_valid)
											
		return examples

	#can be more efficient
	def genSample(self):
		sample = np.zeros(self.dimension)
		#assuming features have independent distributions
		for i in range(0,self.dimension):
			# print(i)
			sample[i]=self.distributions[i].resample(1)[0]
		return sample

	def validSample(self,sample,constraints):
		for cons in constraints:
			(satisfied,splitrule)=cons
			if satisfied!=splitrule.satisfied(sample):
				# print("REJECTED")
				return False
		print("ACCEPTED")
		return True

###########################################

class SplitRule:
	#<= is left , > is right
	def __init__(self,splits):
		self.splits=splits
	def satisfied(self,sample):
		(attr,val)=self.splits[0]
		# print(attr,val)
		# print(sample[attr])
		if sample[attr] <= val:
			return True
		else:
			return False

###########################################

class Node:
	def __init__(self,examples,total_size):
		print("NEW NODE!")
		self.leaf=True
		self.left=None
		self.right=None
		self.splitrule=None
		self.num_examples= examples[0].shape[0]

		if self.num_examples==0:
			self.priority=0
			return

		self.dominant = self.getDominantClass(examples)
		self.misclassified=self.getMisclassified(examples)
		self.fidelity = 1 - (float(self.misclassified)/self.num_examples)
		self.reach = float(self.num_examples)/total_size
		self.priority = self.reach* (1 - self.fidelity)
		print(self.fidelity,self.reach,self.num_examples)

	def getDominantClass(self,examples):
		(trainX,labels) = examples
		counts={}
		for label in labels:
			if label not in counts:
				counts[label]=1
				continue
			counts[label]+=1
		max_count=0
		max_class=0
		for label in counts:
			if counts[label]>max_count:
				max_class=label
				max_count=counts[label]

		return max_class

	def getMisclassified(self,examples):
		(trainX,labels)=examples
		misCount=0
		for label in labels:
			if label != self.dominant:
				misCount+=1
		return misCount

	def classify(self,sample):
		if self.splitrule.satisfied(sample):
			return self.left.classify(sample)
		else:
			return self.right.classify(sample)


###########################################

def entropy(counts,n):
	res=0
	for key in counts:
		c = counts[key]
		if (c==0):
			continue
		p = float(c)/n
		res-=p*np.log2(p)
	return res


def mutual_information(X,y):
	gains = np.zeros(X.shape)
	n = X.shape[0]
	ind_array=np.argsort(X)
	labels, counts = np.unique(y, return_counts=True)
	lcounts={}
	rcounts={}
	for i in range(0,labels.shape[0]):
		rcounts[labels[i]]=counts[i]
		lcounts[labels[i]]=0
	# print("PARENT : ")
	# print(rcounts)
	e_parent = entropy(rcounts,n)
	temp = np.zeros((n,1))
	j=0
	for i in ind_array[:-1]:
		lab = y[i]
		lcounts[lab]+=1
		rcounts[lab]-=1
		# print(lcounts)
		# print(rcounts)
		f_l=(float((j+1))/n)
		f_r=1-f_l
		e_l =f_l* entropy(lcounts,n) #weighted entropies
		e_r=f_r *entropy(rcounts,n)
		gains[i]= e_parent-(e_l+e_r)
		temp[i]=j
		j+=1
		# print(lcounts)
		# print(return_counts)
		# print(i, gains[i])
	gains[ind_array[-1]]=-np.inf
	# print("WINNER")
	# w=np.argmax(gains)
	# print(w,gains[w],temp[w])

	return gains

#usual c4.5 split only for now
def bestMofNSplit(examples):
	(X,labels)=examples
	n=X.shape[0]
	d=X.shape[1]
	gains = np.zeros((n,d))
	for i in range(0,d):
		gains[:,i]=mutual_information(X[:,i],labels)
	split_point = np.unravel_index(np.argmax(gains),gains.shape)
	# print(split_point)
	# print(gains[split_point])
	srule= SplitRule([(split_point[1],X[split_point])])
	return srule

def partition(examples,srule):
	(X,Y) = examples
	n = X.shape[0]
	# print(X[1:5,:])
	el=[]
	er=[]
	for i in range(0,n):
		if srule.satisfied(X[i,:]):
			el.append(i)
		else:
			er.append(i)
	print(len(el))
	print(len(er))
	examples_l = (X[el,:],Y[el])
	examples_r = (X[er,:],Y[er])
	return examples_l,examples_r

###########################################



trainX, trainY = get_data("data/sat.trn")
testX, testY = get_data("data/sat.tst")

# from sklearn import datasets
# iris = datasets.load_iris()
# trainX = iris.data[:, :2]  # we only take the first two features.
# Y = iris.target
# # print(trainX.shape)
# # print(Y.shape)
# idx=list(range(0,30))+list(range(50,80))+list(range(100,130))
# trainX=trainX[idx,:]
# Y=Y[idx]
# # print(trainX.shape)
# # print(Y.shape)
# trainY=np.zeros((Y.shape[0],3))
# for i in range(0,Y.shape[0]):
# 	trainY[i,Y[i]]=1


num_classes = trainY.shape[1]
total_size = trainX.shape[0]
print(num_classes,total_size)

from keras.models import Sequential
from keras.layers import Dense
model = createModel(trainX,trainY,num_classes)

##PARAMS
Smin = 10
MAX_NODES=20

oracle = Oracle(model,num_classes)
oracle.setDistributions(trainX)

labels = np.zeros((trainX.shape[0]))
for i in range(0,trainX.shape[0]):
	labels[i]=oracle.oracle_example(trainX[i,:])
	# print(labels[i])
trainingSet=(trainX,labels)
# print(labels[0:10])


sortedQueue = Q.PriorityQueue()
root = Node(trainingSet,total_size)
root.leaf=False
sortedQueue.put((root.priority,(0,root,trainingSet,[])))

# quit()

num_nodes=1
while not sortedQueue.empty():
	(p, (t,node, examples,constraints))=sortedQueue.get()
	num_ex=examples[0].shape[0]
	print("############PROCESSING "+str(num_ex)+" #############")

	if num_ex<Smin:
		print("NEED EXTRA")
		(trainX,labels)= examples
		n_extra = Smin - num_ex
		(tX_or,lab_or) = oracle.oracle_constraints(constraints,n_extra)
		tX_aug = np.concatenate([trainX,tX_or],axis=0)
		lab_aug = np.concatenate([labels,lab_or],axis=0)
		examples_aug=(tX_aug,lab_aug)		
	else :
		print("ALL OK")
		examples_aug = examples

	srule = bestMofNSplit(examples_aug)
	# print(srule.splits)
	examples_l,examples_r = partition(examples,srule)
	lnode= Node(examples_l,total_size)
	rnode= Node(examples_r,total_size)
	
	node.left = lnode
	node.right = rnode
	node.splitrule=srule

	if (num_nodes<MAX_NODES): # or oracle says stop
		cons2 = constraints+[(True,srule)]
		p = lnode.priority
		sortedQueue.put((p,(num_nodes,lnode,examples_l,cons2)))
		num_nodes+=1
	if (num_nodes<MAX_NODES): # or oracle says stop
		cons2 = constraints+[(False,srule)]
		p = rnode.priority
		sortedQueue.put((p,(num_nodes,rnode,examples_r,cons2)))
		num_nodes+=1
	





