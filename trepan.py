import numpy as np
import pandas


from scipy import stats

import queue as Q

import pdb

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
			sample=self.genSample(constraints)
			label=self.oracle_example(sample)
			X_examples[num_valid,:]=sample
			lab_examples[num_valid]=label
			num_valid+=1
			# print(num_valid)
											
		return (X_examples,lab_examples)

	#can be more efficient
	def genSample(self,constraints):
		sample = np.zeros(self.dimension)
		#assuming features have independent distributions
		for i in range(0,self.dimension):
			# print(i)
			done=False
			while not done :
				# print("THIS IS I :" +str(i))
				min_val = constraints.min_val(i)
				max_val = constraints.max_val(i)
				sample[i]=self.distributions[i].resample(1)[0]
				if sample[i] > min_val and sample[i] < max_val :
					done=True
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
	#m of n split

	def __init__(self,splits,m,n):
		self.splits=splits
		self.m=m
		self.n=n
		self.op_dict= {"gte":self.gte,"lte":self.lte}
		self.processSplits()


	def processSplits(self):
		self.max_dict={}
		self.min_dict={}
		for (attr,op_string,val) in self.splits:
			if op_string in ["lte" ,"lt"]:
				if attr not in self.max_dict:
					self.max_dict[attr]=val
				self.max_dict[attr] = max(self.max_dict[attr],val)
			elif op_string in ["gte","gt"]:
				if attr not in self.min_dict:
					self.min_dict[attr]=val
				self.min_dict[attr] = min(self.min_dict[attr],val)

	#for building constraints
	def invert(self):
		splits2= []
		inverse = {"gte":"lt","gt":"lte","lte":"gt","lt":"gte"}
		for (attr,op_string,val) in self.splits:
			op_string=inverse[op_string]
			splits2.append((attr,op_string,val))
		s2 = SplitRule(splits2,self.m,self.n)
		return s2


	def gte(self,arg1, arg2):
		return arg1 >= arg2
	def lte(self,arg1, arg2):
		return arg1 <= arg2
	def lt(self,arg1,arg2):
		return arg1 < arg2
	def gt(self,arg1,arg2):
		return arg1 > arg2

	def satisfied(self,sample):
		sat=0

		for split in self.splits:
			(attr,op_string,val)=split
			op = self.op_dict[op_string]
			if op(sample[attr],val):
				sat+=1
			# print(attr,val)
			# print(sample[attr])
		if sat < self.m:
			return False
		else:
			return True

	def max_val(self,dim):
		if dim in self.max_dict :
			return self.max_dict[dim]
		else :
			return np.inf
	def min_val(self,dim):
		if dim in self.min_dict:
			return self.min_dict[dim]
		else :
			return -np.inf


###########################################

class Node:
	def __init__(self,examples,total_size):
		self.leaf=True
		self.left=None
		self.right=None
		self.splitrule=None
		self.num_examples= examples[0].shape[0]

		if self.num_examples==0:
			self.priority=0
			print("NEW NODE! with priority = "+ str(self.priority))
			return

		self.dominant = self.getDominantClass(examples)
		self.misclassified=self.getMisclassified(examples)
		self.fidelity = 1 - (float(self.misclassified)/self.num_examples)
		self.reach = float(self.num_examples)/total_size
		self.priority = (-1)*self.reach* (1 - self.fidelity)
		# print(self.fidelity,self.reach,self.num_examples)
		print("NEW NODE! with priority = "+ str(self.priority))

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
		if self.leaf :
			return self.dominant
		if self.splitrule.satisfied(sample):
			return self.left.classify(sample)
		else:
			return self.right.classify(sample)


###########################################

class Constraints :

	def __init__(self,num_dim):
		# self.cons_list=[]
		self.num_dim=num_dim
		self.max_list = np.zeros(num_dim)
		self.min_list = np.zeros(num_dim)

	def addRule(self,split):
		for i in range(0,self.num_dim):
			self.max_list[i]=max(self.max_list[i],split.max_val(i))
			self.min_list[i]=min(self.min_list[i],split.min_val(i))

	def max_val(self,dim):
		return self.max_list[dim]

	def min_val(self,dim):
		return self.min_list[dim]

	def copy(self):
		c = Constraints(self.num_dim)
		c.max_list=np.copy(self.max_list)
		c.min_list=np.copy(self.min_list)
		return c


###########################################		

def entropy(counts,n):
	res=0
	for key in counts:
		c = counts[key]
		if (c==0):
			continue
		p = float(c)/n
		# print(p)
		res-=p*np.log2(p)

	# print(res)
	return res


def mutual_information(X,y):
	gains = np.zeros(X.shape)
	n = X.shape[0]
	ind_array=np.argsort(X)
	labels, counts = np.unique(y, return_counts=True)
	lcounts={}
	rcounts={}
	if (X.shape[0]!=y.shape[0]):
		print("ERROR ")

	for i in range(0,labels.shape[0]):
		lcounts[labels[i]]=counts[i]
		rcounts[labels[i]]=0

	global debugging,curr_attr,glob_attr
	printnow = debugging and (curr_attr==glob_attr)
	if printnow:
		print("PARENT : ")
		print(entropy(lcounts,n))
		pdb.set_trace()

	e_parent = entropy(lcounts,n)
	temp = np.zeros((n,1))
	j=0
	prev=-1
	#process in reverse, to deal with identical values
	for i in reversed(ind_array):
		lab = y[i]
		# print(lcounts)
		# print(rcounts)
		#fixed error in iterative loading, didn't consider the case that many 
		#indices can lead to same split
		if (prev >=0 and X[prev]==X[i]):
			gains[i]=gains[prev]
			j+=1
			rcounts[lab]+=1
			lcounts[lab]-=1	
			continue
		prev=i

		f_r=(float(j)/n)
		f_l=1-f_r
		e_r=f_r *entropy(rcounts,n)
		e_l =f_l* entropy(lcounts,n) #weighted entropies
		gains[i]= e_parent-(e_l+e_r)
		temp[i]=j
		j+=1

		rcounts[lab]+=1
		lcounts[lab]-=1		


		if printnow  and j==n:
			print (str(i) + " : LEFT "+ str(f_l*entropy(lcounts,n))+" RIGHT "+str(f_r*entropy(rcounts,n)))
			pdb.set_trace()
			entropy(lcounts,n)
			print( "PROBS : "+str(f_l)+" : "+str(f_r))
			print("GAIN : "+str(gains[i]))
		# print(lcounts)
		# print(return_counts)
		# print(i, gains[i])
	# gains[ind_array[-1]]=-np.inf
	# print("WINNER")
	# w=np.argmax(gains)
	# print(w,gains[w],temp[w])
	# print("END GAIN : "+str(gains[ind_array[-1]]))
	if printnow:
		pdb.set_trace()
	return gains

#usual c4.5 split only for now
def bestMofNSplit(examples):
	(X,labels)=examples
	n=X.shape[0]
	d=X.shape[1]
	print("SPLITTING "+str(n)+" EXAMPLES")
	gains = np.zeros((n,d))
	for i in range(0,d):
		gains[:,i]=mutual_information(X[:,i],labels)
	split_point = np.unravel_index(np.argmax(gains),gains.shape)
	if (np.max(gains)<1e-6):
		return None
	# print(split_point)
	# print(gains[split_point])
	srule= SplitRule([(split_point[1],"lte",X[split_point])],1,1)
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
	# pdb.set_trace()

	if len(er)==0:
		print("OOOOOOOOPS")
		print(srule.splits)
		n=X.shape[0]
		d=X.shape[1]

		global debugging
		debugging=True
		# pdb.set_trace()
		global glob_attr,curr_attr
		glob_attr=srule.splits[0][0]

		print("SPLITTING "+str(n)+" EXAMPLES")
		gains = np.zeros((n,d))
		for i in range(0,d):
			curr_attr=i
			gains[:,i]=mutual_information(X[:,i],Y)
		split_point = np.unravel_index(np.argmax(gains),gains.shape)
		# quit()

	return examples_l,examples_r

###########################################
debugging=False
glob_attr=-1
curr_attr=-1
np.random.seed(200)
from tensorflow import set_random_seed
set_random_seed(2)

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
MAX_NODES=100

oracle = Oracle(model,num_classes)
oracle.setDistributions(trainX)

labels = np.zeros((trainX.shape[0]))
for i in range(0,trainX.shape[0]):
	labels[i]=oracle.oracle_example(trainX[i,:])
	# print(labels[i])
trainingSet=(trainX,labels)
# print(labels[0:10])



num_dim=trainX.shape[1]

sortedQueue = Q.PriorityQueue()
root = Node(trainingSet,total_size)
root.leaf=False
sortedQueue.put((root.priority,(0,root,trainingSet,Constraints(num_dim))))

# quit()

num_nodes=1
while not sortedQueue.empty():
	(p, (t,node, examples,constraints))=sortedQueue.get()
	num_ex=examples[0].shape[0]
	print("############PROCESSING "+str(num_ex)+" #############")

	##need to check generate examples for bugs, and see if it passes all the constraints.
	if num_ex<Smin:
		print("NEED EXTRA")
		(trainX,labels)= examples
		n_extra = Smin - num_ex
		(tX_or,lab_or) = oracle.oracle_constraints(constraints,n_extra)
		# print(tX_or[:,1:5])
		tX_aug = np.concatenate([trainX,tX_or],axis=0)
		lab_aug = np.concatenate([labels,lab_or],axis=0)
		# print(tX_aug[:,1:5])
		examples_aug=(tX_aug,lab_aug)		
	else :
		print("ALL OK")
		examples_aug = examples

	srule = bestMofNSplit(examples_aug)
	if not srule:
		#skip this node, its already pretty pure
		#leave as leaf
		continue
	# quit()
	# print(srule.splits)
	examples_l,examples_r = partition(examples,srule)
	lnode= Node(examples_l,total_size)
	rnode= Node(examples_r,total_size)
	
	node.left = lnode
	node.right = rnode
	node.splitrule=srule

	if (num_nodes<MAX_NODES): # or oracle says stop
		cons2 = constraints.copy()
		cons2.addRule(srule)
		p = lnode.priority
		sortedQueue.put((p,(num_nodes,lnode,examples_l,cons2)))
		num_nodes+=1
	if (num_nodes<MAX_NODES): # or oracle says stop
		cons2 = constraints.copy()
		cons2.addRule(srule.invert())
		p = rnode.priority
		sortedQueue.put((p,(num_nodes,rnode,examples_r,cons2)))
		num_nodes+=1
	





