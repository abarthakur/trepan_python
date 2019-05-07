from keras.models import Sequential
from keras.layers import Dense
import pandas
import numpy as np

from trepan import Trepan,Oracle

###########################################

def load_landsat_data(filename):
	'''
	Utility function to load Landsat dataset.
	Landsat dataset : https://archive.ics.uci.edu/ml/datasets/Statlog+(Landsat+Satellite)
	num_classes= 7, but 6th is empty. 
	This functions
	- Reads the data
	- Renames the class 7 to 6
	- Generates one-hot vector labels
	'''

	data = pandas.read_csv(filename, sep=r"\s+", header=None)
	data = data.values

	dataX = np.array(data[:,range(data.shape[1]-1)])
	dataY = np.array(data[np.arange(data.shape[0]),data.shape[1]-1])

	# convert dataY to one-hot, 6 classes
	num_classes = 6
	dataY = np.array([x-2 if x==7 else x-1 for x in dataY]) # re-named class 7 to 6 as class 6 is empty
	dataY_onehot = np.zeros([dataY.shape[0], num_classes])
	dataY_onehot[np.arange(dataY_onehot.shape[0]), dataY] = 1

	return dataX, dataY_onehot

def create_model (trainX,trainY,num_classes):
	model = Sequential()
	model.add(Dense(16, input_dim=trainX.shape[1], activation="sigmoid"))
	model.add(Dense(16, activation="sigmoid"))
	model.add(Dense(num_classes, activation="softmax"))
	model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=['accuracy'])
	model.fit(trainX, trainY, epochs=5, batch_size=10) # epochs=150
	return model

###########################################

# np.random.seed(200)
# from tensorflow import set_random_seed
# set_random_seed(2)

#load training data
trainX, trainY = load_landsat_data("data/sat.trn")
testX, testY = load_landsat_data("data/sat.tst")
num_classes = trainY.shape[1]
total_num_examples = trainX.shape[0]
print(num_classes,total_num_examples)


#build oracle
model = create_model(trainX,trainY,num_classes)
oracle = Oracle(model,num_classes,trainX)

#build tree with TREPAN
MIN_EXAMPLES_PER_NODE = 30
MAX_NODES=200
root=Trepan.build_tree(MIN_EXAMPLES_PER_NODE,MAX_NODES,trainX,oracle)

#calculate fidelity
num_test_examples= testX.shape[0]
correct=0
for i in range(0,num_test_examples):
	ann_prediction = oracle.get_oracle_label(testX[i,:])
	tree_prediction = root.classify(testX[i,:])
	correct += (ann_prediction==tree_prediction)

fidelity=float(correct)/num_test_examples

print("Fidelity of the model is : "+str(fidelity))

