# trepan_python

An implementation for tree building algorithm TREPAN as described in "Extracting tree-structured representations of trained networks" : Craven,Shavlik 1993.

TREPAN extracts a decision tree from a neural network using a sampling method.

TODO:
1. Support discrete (including categorical) features.
2. Support m-of-n splits at nodes.
	* Implement hill climbing algorithm to find best m-of-n constraint given a single C4.5 constraint as seed.
	* Currently uses a C4.5 constraint. (Quinlan,1993)
3. Stopping criterion is only num_nodes < MAX_NODES . Other criterion described in the paper (TODO:Description) is unimplemented.
4. Replicate/get close to results on at least one dataset

To run, first install the required Python 3 modules

```
pip install -r requirements.txt
```

Then, run `run.py` 
```
python run.py
```

`run.py` does the following things
1. Trains a simple 2-hidden-layer neural network on the Landsat dataset
2. Builds a TREPAN tree to imitiate mentioned NN
3. Calculates the 'fidelity' of the TREPAN tree compared to the NN, on the test data set.