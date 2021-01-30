# Metacontroller Documentation

This folder contains the implementation of the updated hierarchical strategy discovery algorithm. 

## How to run BMPS

The easiest way to run BMPS on a new environment is to use the ```optimize``` function from ```metacontroller.py```, which optimizes the VOC weights with Bayesian Optimization. An example of how to do this can be found in ```comparison_meta_hier.ipynb``` in which the goal-switching variant as well as a lessened, purely hierarchical variant of our algorithm are trained and evaluated. 

## Extensions to the mouselab environment
The original mouselab environment has been extended in ``` utils.mouselab_VAR ``` to include the following adjustments: 
1. Computational speedup through tree contraction.
2. An optional adjusted cost function returning the number of nodes neeeded to compute VPI and VPI_action features. 
3. Updated functions to compute paths to a node while taking the possibility of multiple paths leading to the same node into account. 
4. The option to define the environment using a Normal distribution instead of a Categorical distribution. Behind the scenes the Normal distribution will be binned and treated as a Categorical distribution.

Additional adjustments to mouselab related to the hierarchical decomposition are made in ```utils.mouselab_metacontroller ```:
1. Goal MDP creation: Goal nodes are extracted from the whole MDP and returned as a new MDP. The goal distribution is updated by the best expected reward received along any path to that goal. If the goal state has been computed already, the goal node is initialized revealed.
2. Sub MDP creation: Given a goal node, a low level planning MDP is created containing all the nodes connecting the root node to the selected goal. Revealed nodes will be initialized as revealed in the new MDP.
3. Dummy node: To enable goal switching the sub goal creation can optionally include adding a dummy node to the low level MDP. This dummy node is initialized as a revealed goal state with a reward value representing the expected termination reward of the best alternative goal. 

