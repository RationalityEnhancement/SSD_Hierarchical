# Leveraging Reinforcement Learning to Discover Algorithms for Computationally Efficient Hierarchical Planning

This repository is the official implementation of [Leveraging Reinforcement Learning to Discover Algorithms for Computationally Efficient Hierarchical Planning](XXX). 


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





