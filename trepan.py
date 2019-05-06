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
	'''
	Wrapper object for the ANN which we wish to imitate. Also contains logic to generate examples from the 
	constrained distribution of training examples.
	'''

	def __init__(self,network,num_classes,trainX):
		self.network=network
		self.num_classes=num_classes
		self.dimension=trainX.shape[1]
		self.feature_distributions=self.generate_feature_distributions(trainX)

	def generate_feature_distributions(self,trainX):
		'''
		Returns a list of objects modeling the probability distributions of each feature.
		For continuous features we use Gaussian Kernel Density Estimation from scipy.stats
		'''

		feature_distributions =[]
		#only consider continuous distributions
		for i in range(0,self.dimension):
			feature_values = trainX[:,i].reshape(trainX.shape[0])
			kernel = stats.gaussian_kde(feature_values,bw_method='silverman')
			feature_distributions.append(kernel)
		return feature_distributions

	def get_oracle_label(self,example):
		'''
		Returns the label predicated by the oracle network for example
		'''

		onehot =self.network.predict(np.array([example])).reshape(self.num_classes)
		return np.argmax(onehot)

	def generate_constrained_examples_with_labels(self,constraints,num_examples):
		'''
		Returns a tuple of examples,oracle_labels , where examples are drawn from the distribution of the
		training examples, after constraints have been applied to it.
		'''

		examples= np.zeros((num_examples,self.dimension))
		oracle_labels = np.zeros(num_examples)
		i=0
		print(num_examples)
		for i in range(0,num_examples):
			example=self.generate_constrained_example(constraints)
			label=self.get_oracle_label(example)
			examples[i,:]=example
			oracle_labels[i]=label
											
		return (examples,oracle_labels)

	#can be more efficient
	def generate_constrained_example(self,constraints):
		'''
		Returns an example drawn from the distribution of the training examples, after constraints have been applied to it.
		'''

		example = np.zeros(self.dimension)
		#assuming features have independent distributions, sample each feature separately
		for i in range(0,self.dimension):
			min_val = constraints.min_val(i)
			max_val = constraints.max_val(i)
			done=False
			#generate the ith feature by rejection sampling
			while not done :
				#sample i^th feature from its distribution
				example[i]=self.feature_distributions[i].resample(1)[0]
				if example[i] > min_val and example[i] < max_val :
					done=True
		return example

	def is_valid_example(self,example,constraints):
		'''
		Returns True if sample satisfies given constraints, else False
		'''
		
		for constraint in constraints:
			(satisfied,splitrule)=constraint
			if satisfied!=splitrule.satisfied(example):
				return False
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
	'''
	Object represents a single node in the decision tree. It's important fields are
	leaf : bool, True if node is a leaf node, False if it is an internal node
	left_child,right_child : type:Node , children of internal node
	splitrule : type:SplitRule object used to route an arriving example to either the left or right node.
	The splitrule is chosen to be the "best" according to 
	'''

	def __init__(self,labeled_examples,total_num_examples):
		self.leaf=True
		self.left_child=None
		self.right_child=None
		self.splitrule=None
		self.num_examples= labeled_examples["trainX"].shape[0]

		if self.num_examples==0:#when does this happen?
			self.priority=0
		else:
			self.dominant = self.get_dominant_class(labeled_examples)
			self.misclassified=self.get_misclassified_count(labeled_examples)
			self.fidelity = 1 - (float(self.misclassified)/self.num_examples)
			self.reach = float(self.num_examples)/total_num_examples
			self.priority = (-1)*self.reach* (1 - self.fidelity)
		
		print("NEW NODE! with priority = "+ str(self.priority))

	def get_dominant_class(self,labeled_examples):
		'''
		This function returns the "dominant" class of this node, i.e., the class with the highest count of the examples in this node.
		The dominant class
		'''

		trainX = labeled_examples["trainX"]
		labels = labeled_examples["labels"]
		class_counts={}
		#get count for all labels
		for label in labels:
			if label not in class_counts:
				#insert in counter
				class_counts[label]=0
			class_counts[label]+=1

		#get the class with the max count
		max_count=0
		max_class=0
		for label in class_counts:
			if class_counts[label]>max_count:
				max_class=label
				max_count=class_counts[label]
		return max_class

	def get_misclassified_count(self,labeled_examples):
		'''
		Get the number of training examples misclassified by this node, if it were a leaf.
		This is nothing but the number of examples not belonging to the dominant class.
		This value is used to calculate the "priority" of a node, which determines when to explore/split a node.
		'''

		labels = labeled_examples["labels"]
		num_misclassified=0
		for label in labels:
			if label != self.dominant:
				num_misclassified+=1
		return num_misclassified

	def classify(self,example):
		'''
		Returns the predicted class for a given example.
		For an internal node, the example is recursively routed to either the left or right child according to the **splitrule**,
		till it descends to a leaf.
		For a leaf, the dominant (max count) class is returned as the label
		'''
		
		if self.leaf :
			return self.dominant
		if self.splitrule.satisfied(example):
			return self.left_child.classify(example)
		else:
			return self.right_child.classify(example)


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
Smin = 30
MAX_NODES=200

oracle = Oracle(model,num_classes,trainX)

labels = np.zeros((trainX.shape[0]))
for i in range(0,trainX.shape[0]):
	labels[i]=oracle.get_oracle_label(trainX[i,:])
	# print(labels[i])
training_set = {"trainX":trainX,"labels":labels}
# print(labels[0:10])



num_dim=trainX.shape[1]

sortedQueue = Q.PriorityQueue()
root = Node(training_set,total_size)
root.leaf=False
sortedQueue.put((root.priority,(0,root,(trainX,labels),Constraints(num_dim))))

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
		(tX_or,lab_or) = oracle.generate_constrained_examples_with_labels(constraints,n_extra)
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

	if examples_r[0].shape[0]==0 or examples_l[0].shape[0]==0:
		el2,er2=partition(examples_aug,srule)
		if el2[0].shape[0]==0 or er2[0].shape[0]==0:
			print("An empty split? Shouldn't be possible")
			print("Split Rule : "+str(srule.splits))
			(xtemp,ytemp)=examples_aug
			n=xtemp.shape[0]
			d=xtemp.shape[1]
			# global debugging
			debugging=True
			print("Entering debugging mode")
			# pdb.set_trace()
			# global glob_attr,curr_attr
			glob_attr=srule.splits[0][0]
			print("SPLITTING "+str(n)+" EXAMPLES")
			gains = np.zeros((n,d))
			for i in range(0,d):
				curr_attr=i
				gains[:,i]=mutual_information(xtemp[:,i],ytemp)
			split_point = np.unravel_index(np.argmax(gains),gains.shape)

	examples_l_dict={"trainX":examples_l[0],"labels":examples_l[1]}
	examples_r_dict={"trainX":examples_r[0],"labels":examples_r[1]}
	lnode= Node(examples_l_dict,total_size)
	rnode= Node(examples_r_dict,total_size)
	
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
	

fidelity=0
n_test= testX.shape[0]
for i in range(0,n_test):
	lab = oracle.get_oracle_label(testX[i,:])
	lab2 = root.classify(testX[i,:])
	fidelity += (lab==lab2)

fidelity=float(fidelity)/n_test

print("Fidelity of the model is : "+str(fidelity))
