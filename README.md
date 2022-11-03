# Leveraging Reinforcement Learning to Discover Algorithms for Computationally Efficient Hierarchical Planning

This repository is the official implementation of [Leveraging Reinforcement Learning to Discover Algorithms for Computationally Efficient Hierarchical Planning](Paper/Paper.pdf). 

## Structure

### Simulation experiments

The results of the simulation experiments can be found in the top level folders "2_36", "3_54", "4_72", "5_90". The foldernames correspond to the used environment structure specified by the number of goals and number of total nodes. 

### Human experiments

The results of the experiments can be found in the Human-Experiments folder. The cleaned data and analysis code are in the sub-directories named after the corresponding experiment. 

The experiments were hosted on Heroku and run on Cloudresearch and Prolific. The experiment code can be found in ```Human-experiments/ssd-discovery-eu```.

### Metacontroller

The metacontroller folder contains the main source code for the hierarchical BMPS algorithm. 

The main files of the implementation are: 

The environment implementation:
```Metacontroller/utils/mouselab_metacontroller.py```

The main algorithm:
```Metacontroller/metacontroller.py```

The original BMPS implementation:
```Metacontroller/vanilla_BMPS.py```

### How to run BMPS

The easiest way to run BMPS on a new environment is to use the ```optimize``` function from ```metacontroller.py```, which optimizes the VOC weights with Bayesian Optimization. An example of how to do this can be found in ```comparison_meta_hier.ipynb``` in which the goal-switching variant as well as a lessened, purely hierarchical variant of our algorithm are trained and evaluated. 

### Extensions to the mouselab environment
The original mouselab environment has been extended in ``` utils.mouselab_VAR ``` to include the following adjustments: 
1. Computational speedup through tree contraction.
2. An optional adjusted cost function returning the number of nodes neeeded to compute VPI and VPI_action features. 
3. Updated functions to compute paths to a node while taking the possibility of multiple paths leading to the same node into account. 
4. The option to define the environment using a Normal distribution instead of a Categorical distribution. Behind the scenes the Normal distribution will be binned and treated as a Categorical distribution.

Additional adjustments to mouselab related to the hierarchical decomposition are made in ```utils.mouselab_metacontroller ```:
1. Goal MDP creation: Goal nodes are extracted from the whole MDP and returned as a new MDP. The goal distribution is updated by the best expected reward received along any path to that goal. If the goal state has been computed already, the goal node is initialized revealed.
2. Sub MDP creation: Given a goal node, a low level planning MDP is created containing all the nodes connecting the root node to the selected goal. Revealed nodes will be initialized as revealed in the new MDP.
3. Dummy node: To enable goal switching the sub goal creation can optionally include adding a dummy node to the low level MDP. This dummy node is initialized as a revealed goal state with a reward value representing the expected termination reward of the best alternative goal.

## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```
OR
```setup
conda env create -f environment.yml
```

> You can use either pip or conda to install all required dependencies

## Training

To train the model(s) in the paper, run this command:

```train
python BMPS_flat.py <number of goals> 1
```

```train
python BMPS_hierarchical.py <number of goals> 1
```
```train
python breadth.py <number of goals> 1
```

```train
python depth.py <number of goals> 1
```

```train
python backward.py <number of goals> 1
```

```train
python bidirectional.py <number of goals> 1
```

```train
python adaptive_metareasoning.py <number of goals> 1 1000 0
```

## Evaluation

To evaluate, run:

```eval
python random_test.py <number of goals>
```

```eval
python myopic_voc_flat_test.py <number of goals> 
```

```eval
python myopic_voc_hierarchical_test.py <number of goals>
```

```eval
python BMPS_flat.py <number of goals> 0
```

```eval
python BMPS_hierarchical.py <number of goals> 0
```

```eval
python breadth.py <number of goals> 0
```
```eval
python depth.py <number of goals> 0
```

```eval
python backward.py <number of goals> 0
```

```eval
python bidirectional.py <number of goals> 0
```

```eval
python adaptive_metareasoning.py <number of goals> 0 1000 0
```





