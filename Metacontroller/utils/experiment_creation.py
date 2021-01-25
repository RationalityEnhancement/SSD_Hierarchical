import numpy as np

# Helper functions for generating json files to be used in the mouselab-MDP online experiment
# Author: Lovis Heindrich

def rewards_json(start_seed, end_seed, TREE, INIT, Mouselab):
    """Creates rewards as used in the mouselab experiment.

    Args:
        start_seed (int): start seed (inclusive)
        end_seed (int): stop seed (exclusive)
        TREE (list): Mouselab tree structure
        INIT (list): Mouselab init structure
        Mouselab (Mouselab): Mouselab environment class
    """
    truths = []
    rewards = "["
    for i in range(start_seed, end_seed):
        np.random.seed(i)
        env = Mouselab(TREE, INIT)
        ground_truth = list(env.ground_truth)
        rewards += f"\n{{\"trial_id\": {i}, \"stateRewards\": {str(ground_truth)}}},"
        truths.append(ground_truth)
    rewards = rewards[:-1] + "\n]"
    return rewards, truths

def getGraph(TREE):
    """Creates a mouselab graph as used in structure.json in mouselab experiments

    Args:
        TREE (list): Mouselab tree structure

    Returns:
        [type]: [description]
    """
    keys = ["left", "up", "right", "farright"]
    graph = {}
    for i in range(len(TREE)):
        node_dict = {}
        c_counter = 0
        for c in TREE[i]:
            node_dict[keys[c_counter]] = [0, c]
            c_counter += 1
        graph[i] = node_dict
    return graph

def actions_to_graph_actions(path, TREE):
    """Transforms a path of nodes to the corresponding actions in the graph. 

    Args:
        path (list): List of nodes along the takeb path
        TREE (list): Mouselab tree

    Returns:
        List: Actions that correspond to the taken path in string form
    """

    graph = getGraph(TREE)
    graph_actions = []
    for i in range(1, len(path)):
        previous = path[i-1]
        current = path[i]
        # Find action from previous to current
        possible_actions = graph[previous]
        for k, v in possible_actions.items():
            if v[1] == current:
                str(graph_actions.append(k))
                break
        
    return graph_actions

def get_demonstrations(start_seed, end_seed, W, TREE, INIT, disable_meta, trace, HIGH_COST=1,LOW_COST=1, cost_function="Basic"):
    """Creates a demonstration json for the environments for the given seeds

    Args:
        start_seed (int): Start seed (included)
        end_seed (int): End seed (not included)
        W (array): Weights for the metacontroller
        TREE (array): Mouselab tree
        INIT (array): Mouselab init
        disable_meta (bool): Enables/disables goal switching
        trace (func): tracing function giving the actions of BMPS for the given tree
        COST (int, optional): Cost of a click. Defaults to 1.

    Returns:
        (string, array, array, array): Json string of the demonstration trials and individual elements as arrays
    """
    json = ""
    click_list = []
    reward_list = []
    action_list = []
    for i in range(start_seed, end_seed):
        SEED = i
        clicks, rewards, actions = trace(W, TREE, INIT, HIGH_COST=HIGH_COST, LOW_COST=LOW_COST, SWITCH_COST=0, SEED=SEED, term_belief=False, disable_meta=disable_meta, cost_function=cost_function)
        click_list.append(clicks)
        reward_list.append(rewards)
        action_list.append(actions)
        json += "\n{\"pid\":1,\"actions\":"

        json += str(actions_to_graph_actions(actions, TREE)).replace("\'", "\"")

        json += ",\"clicks\":"

        json += str(clicks)

        json += ",\"stateRewards\":"

        json += str(list(rewards))

        json += "},"

    return json[:-1], click_list, reward_list, action_list