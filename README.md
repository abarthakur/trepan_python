# trepan_python
An implementation of the TREPAN algorithm in python. TREPAN extracts a decision tree from an ANN using a sampling method.

Currently still testing. Currently implements-
1. Continuous attributes input but not discrete attributes
2. Single attribute split rules, not M of N rules.
3. Stops tree growth when number of internal nodes exceeds threshold. Need to add condition specified in the original paper.
