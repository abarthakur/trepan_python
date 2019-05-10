import numpy as np
from scipy import stats
import queue



class Trepan:
	'''
	Wrapper class for tree building algorithm TREPAN as described in 
	"Extracting tree-structured representations of trained networks" : Craven,Shavlik 1993

	Differences/Unimplemented/Implemented differently:
	1. Not using m-of-n splits in decision tree nodes, instead a simple C4.5 split (Quinlan,1993) is used.
	2. Input features must be numeric.
	3. Stopping criterion is only num_nodes < MAX_NODES . Other criterion described in the paper [TODO:Description?] is unimplemented.
	'''

	@staticmethod
	def build_tree(MIN_EXAMPLES_PER_NODE,MAX_NODES,trainX,oracle):
		'''			
			Parameters
			----------
			MIN_EXAMPLES_PER_NODE : int 
									corresponds to S_min in the original paper.
			MAX_NODES 	: int
			trainX		: numpy array 
						: training examples of dimension (num_examples,num_dimensions) 
			oracle 		: 	Oracle object, used to generate samples given constraints of linear inequalities on the input space,
							It also wraps the NN model to imitate, which it uses to label the instances. 

			Returns
			--------
			root : the root node of the built tree. Call root.classify(single_example) to get the prediction of the imitating tree. 
				single_example must have dimension (num_examples,num_dimensions)
		'''

		total_num_examples = trainX.shape[0]
		num_dimensions = trainX.shape[1]
		#generate labels from oracle
		labels = np.zeros((trainX.shape[0]))
		for i in range(0,trainX.shape[0]):
			labels[i]=oracle.get_oracle_label(trainX[i,:])

		all_examples=(trainX,labels)
		all_examples_dict={"trainX":trainX,"labels":labels}

		#initialize queue with root
		sortedQueue = queue.PriorityQueue()
		root = Node(all_examples_dict,total_num_examples)
		sortedQueue.put((root.priority,0,root,all_examples,Constraints(num_dimensions)))

		num_nodes=1
		while not sortedQueue.empty():
			(p, tiebreaker, node, examples, constraints)=sortedQueue.get()
			num_examples=examples[0].shape[0]
			assert(node.leaf)
			assert(num_examples>0)

			print("############PROCESSING "+str(num_examples)+" #############")

			if num_examples<MIN_EXAMPLES_PER_NODE:
				print("NEED EXTRA")
				(trainX,labels)= examples
				num_required = MIN_EXAMPLES_PER_NODE - num_examples
				(trainX_oracle,labels_oracle) = oracle.generate_constrained_examples_with_labels(constraints,num_required)
				trainX_aug = np.concatenate([trainX,trainX_oracle],axis=0)
				labels_aug = np.concatenate([labels,labels_oracle],axis=0)
				examples_aug=(trainX_aug,labels_aug)		
			else :
				print("ALL OK")
				examples_aug = examples

			srule = SplitFinder.find_best_m_of_n_split(examples_aug)
			#a good split was not found, so keep as leaf
			if not srule:
				continue
			examples_l,examples_r = partition(examples,srule)

			#even though the trivial splits are avoided with examples_aug, 
			#the splitrule may still split the examples trivially
			#trivial split, so keep as leaf
			if len(examples_l[0])==0 or len(examples_r[0])==0:
				continue

			#TODO: Add stop criterion thresholding the proportion of dominant class, similar to 
			
			#number of nodes will exceed MAX_NODES if we make the split, so keep as leaf
			if (MAX_NODES - num_nodes<2):
				continue

			#split the node, and make as internal
			examples_l_dict={"trainX":examples_l[0],"labels":examples_l[1]}
			examples_r_dict={"trainX":examples_r[0],"labels":examples_r[1]}
			left_child= Node(examples_l_dict,total_num_examples)
			right_child= Node(examples_r_dict,total_num_examples)
			node.left_child = left_child
			node.right_child = right_child
			node.splitrule=srule
			node.leaf=False

			#add child nodes
			constraints_left = constraints.copy()
			constraints_left.addRule(srule)
			priority = left_child.priority
			sortedQueue.put((priority,num_nodes,left_child,examples_l,constraints_left))
			num_nodes+=1
			
			constraints_right = constraints.copy()
			constraints_right .addRule(srule.invert())
			priority = right_child.priority
			sortedQueue.put((priority,num_nodes,right_child,examples_r,constraints_right))
			num_nodes+=1
		
		return root


###########################################


class SplitFinder:
	'''
	Wrapper class containing algorithm to find splits, and utility functions to calculate certain mathematical formulae.

	'''


	@staticmethod
	def entropy(class_frequencies,num_examples):
		'''
		Entropy of a data point is the "surprisal" value of it. 
		Less probable points have a higher value of entropy.

		Mathematically, entropy defined for a set of categorical data points is,
		H(x)= - \sum_{i=1}^{n} ( P(X=class_i) \log P(X=class_i) )

		Overall for a set of data points, entropy is more for a more uncertain distribution.
		So, for a homogenous set of points, the entropy is the lowest.
		Entropy is always positive.
		'''
		entropy=0
		for class_i in class_frequencies:
			frequency = class_frequencies[class_i]
			#lim(x->0) xlogx = 0
			if (frequency==0):	
				continue
			prob = float(frequency)/num_examples
			entropy+= (-1) * prob*np.log2(prob)
		return entropy

	@staticmethod
	def mutual_information(X,y):
		'''
		Information gain is the difference in entropy between the original state & the new state.
		In this case it is 
		IG(X)	= H(X) - fraction(X<=split) * H(X|X<=split) - fraction(X>split) * H(X|X>split)
				= H_parent - frac_l * H_l - frac_r * H_r

		Assuming a "less than equal to" split rule.

		Parameters
		---------
		X : np array of shape (num_examples,)
		y : np array of shape (num_examples,) 
			the category/class labels

		Returns:
		gains : np array of shape (num_examples,1)
				ith value corresponds to taking the ith example as the split point

		'''

		num_examples=X.shape[0]
		assert(y.shape[0]==num_examples)
		assert(len(X.shape)==1 and len(y.shape)==1)

		#get unique classes and their frequencies
		classes, class_frequencies = np.unique(y, return_counts=True)
		num_classes=classes.shape[0]

		#get sorted indices for X, sorting along axis=0 (only axis here)
		sorted_indices=np.argsort(X,axis=0)
		
		#intialize gains array
		gains = np.zeros(num_examples)

		#initialize splits with all points in the left partition
		left_frequencies={}
		right_frequencies={}
		for i in range(0,num_classes):
			left_frequencies[classes[i]]=class_frequencies[i]
			right_frequencies[classes[i]]=0

		entropy_parent = SplitFinder.entropy(left_frequencies,num_examples)
		prev_idx=None
		shifted=0
		'''
		Note: Since this function calculates the IG values iteratively, we need to consider the case of identical values,
		and skip them, since they provide the same split.

		Start with the highest value. (This is the trivial split)
		While calculating the gain, the split point is always in the left set since "<=" rule is use.
		At the end, we shift an example from the left child to the right child (except for case of identical values).

		'''
		for idx in reversed(sorted_indices):
			label = y[idx]

			#case : identical value
			if (prev_idx and X[prev_idx]==X[idx]):
				gains[idx]=gains[prev_idx]
			else:
				frac_right=(float(shifted)/num_examples)
				frac_left=1-frac_right
				gains[idx]=entropy_parent
				gains[idx]-= frac_left* SplitFinder.entropy(left_frequencies,num_examples)
				gains[idx]-= frac_right* SplitFinder.entropy(right_frequencies,num_examples)
			shifted+=1
			right_frequencies[label]+=1
			left_frequencies[label]-=1		
			prev_idx=idx
		return gains

	@staticmethod
	def find_best_binary_split(examples):
		'''
		Find the best binary split along a single axis, according to maximum information gain.
		This is the same split as used in the C4.5 algorithm, Quinlan 1993

		Returns
		-------
		srule : SplitRule object 
		'''
		(X,labels)=examples
		num_examples=X.shape[0]
		num_dimensions=X.shape[1]
		print("SPLITTING "+str(num_examples)+" EXAMPLES")

		#initialize gains 
		gains = np.zeros((num_examples,num_dimensions))
		#calculate gains considering each feature
		for i in range(0,num_dimensions):
			gains[:,i]= SplitFinder.mutual_information(np.reshape(X[:,i],num_examples),
														np.reshape(labels,num_examples))

		#low gains = parent purity is not increased by much - so don't split
		if (np.max(gains)<1e-6):
			return None

		#get split point (sample_idx,feature_idx), i.e. the point with max gains
		split_point = np.unravel_index(np.argmax(gains),gains.shape)

		#build split rule object
		feature_to_split=split_point[1]
		split_value=X[split_point]

		#avoid making a trivial split
		if (X[:,feature_to_split] <= split_value).all() or (X[:,feature_to_split] >= split_value).all():
			return None

		splits=[(feature_to_split,"lte",split_value)]
		#create a simple split rule, i.e. 1-of-1
		srule= SplitRule(splits,1,1)
		return srule

	@staticmethod
	def find_best_m_of_n_split(examples):
		'''
		Find the best m-of-n split. An m-of-n split, is a splitting function composed of n boolean expressions.
		An m-of-n split is satisifed if at least m-of-n expressions is satisfied
		TODO: Right now just returns a binary split
		'''

		seed= SplitFinder.find_best_binary_split(examples)
		##TODO find best m-of-n split with hill climbing method, with C4.5 split as 
		srule=seed
		return srule


def partition(examples,srule):
	'''
	Utility function to partition an example set by filtering with a SplitRule object.

	Returns
	-------
	examples_l,examples_r : np arrays of shape (*,num_dimensions)
	'''
	(X,y) = examples
	num_examples = X.shape[0]
	left_partition,right_partition=[[],[]]

	for idx in range(0,num_examples):
		if srule.satisfied(X[idx,:]):
			left_partition.append(idx)
		else:
			right_partition.append(idx)

	examples_l = (X[left_partition,:],y[left_partition])
	examples_r = (X[right_partition,:],y[right_partition])
	return examples_l,examples_r


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


class Node:
	'''
	Object represents a single node in the decision tree. It's important fields are
	leaf : bool, True if node is a leaf node, False if it is an internal node
	left_child,right_child : type:Node , children of internal node
	splitrule : type:SplitRule object used to route an arriving example to either the left or right node.
	The splitrule is chosen to be the "best" according to SplitFinder.find_best_m_of_n_split.

	priority is calculated as
	priority = reach (1- fidelity)
	reach 	= fraction of instances that reach n
			= num_examples/total_num_examples
	fidelity= classification rate (wrt to ANN labels, not ground truth)
			= 1 - (misclassified/num_examples)

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


class SplitRule:
	'''
	Class representing a splitting/partitioning function.
	The function is particularly an m-of-n expression which is composed of n boolean value expressions,
	and which is satisfied by an example if at least m out of n expressions is satisfied.
	The boolean expressions are linear inequalities/equalities.

	Keeps a list of upper and lower bounds on each feature. Used by constraints object.

	Parameters
	---------
	splits	: List[(feature_to_split,operator,split_value)]
	m		: int (minimum number of constraints to satisfy for SplitRule to be satisfied)
	n		: int , number of constraints this splitrule is composed of
	'''

	def __init__(self,splits,m,n):
		self.splits=splits
		self.m=m
		self.n=n
		self.op_dict= {"gte":self.gte,"lte":self.lte}
		self.process_splits()

	def process_splits(self):
		self.max_dict={}
		self.min_dict={}
		for (feature_to_split,operator,split_value) in self.splits:
			if operator in ["lte" ,"lt"]:
				if feature_to_split not in self.max_dict:
					self.max_dict[feature_to_split]=split_value
				self.max_dict[feature_to_split] = max(self.max_dict[feature_to_split],split_value)
			elif operator in ["gte","gt"]:
				if feature_to_split not in self.min_dict:
					self.min_dict[feature_to_split]=split_value
				self.min_dict[feature_to_split] = min(self.min_dict[feature_to_split],split_value)

	#for building constraints
	def invert(self):
		'''
		Returns the "inverse" of this SplitRule object. 
		Does this by inverting each of the individual constraints.
		While at each level, only one (left child's) split is evaluated, 
		We need to pass the inverse to the right child to add to its list of constraints.
		Note: What is the inverse of an m-of-n split? 
		'''
		inverted_splits= []
		inverse_map = {"gte":"lt","gt":"lte","lte":"gt","lt":"gte"}
		for (feature_to_split,operator,val) in self.splits:
			inverse_operator=inverse_map[operator]
			inverted_splits.append((feature_to_split,inverse_operator,val))
		invsplit = SplitRule(inverted_splits,self.m,self.n)
		return invsplit

	def gte(self,arg1, arg2):
		return arg1 >= arg2

	def lte(self,arg1, arg2):
		return arg1 <= arg2

	def lt(self,arg1,arg2):
		return arg1 < arg2

	def gt(self,arg1,arg2):
		return arg1 > arg2

	def satisfied(self,sample):
		'''
		Evaluates the splitrule given a sample.
		
		Parameters
		----------
		sample : np array of shape (num_features)

		Returns
		---------
		True if at least m constraints are satisfied
		False otherwise
		'''
		num_satisfied=0
		for split in self.splits:
			(feature_idx,op_string,val)=split
			op = self.op_dict[op_string]
			if op(sample[feature_idx],val):
				num_satisfied+=1
		if num_satisfied < self.m:
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



