from utils.mouselab_VAR import MouselabVar
from utils.distributions import Normal, Categorical, expectation, cross
from itertools import compress
import numpy as np
import kmeans1d

class MouselabMeta(MouselabVar):
    """ 
    Mouselab wrapper that adds hierarchical structure to the MDP. 
    Takes a list of goal nodes as an additional input which are then used to create a goal MDP containing only the goal nodes as well as sub MDPs for each goal containing all nodes along the paths that connect the root node to the goal node.
    """

    def __init__(self, tree, init, goals=None, ground_truth=None, term_belief=True, sample_term_reward=False, depth_weight = 0, cost_weight = 1, high_cost = 1, low_cost = 1, seed = None, simple_cost=True):
        if seed is not None:
            np.random.seed(seed)

        super().__init__(tree, init, ground_truth=ground_truth, cost=0, term_belief=term_belief, sample_term_reward=sample_term_reward, depth_weight=depth_weight, cost_weight=cost_weight, simple_cost=simple_cost)

        # Init leaf nodes as goals if not set
        if goals is not None:
            self.goals = goals
        else: 
            goals = []
            for i,el in enumerate(tree):
                if len(el) == 0:
                    goals.append(i)
            self.goals = goals
        
        # Precompute paths
        self._all_paths = [p for _, p in enumerate(self.all_paths())]
        self.goal_paths = {}
        for goal in self.goals:
            goal_paths = [p for p in self._all_paths if goal in p]
            self.goal_paths[goal] = goal_paths

        self.HIGH_COST = high_cost
        self.LOW_COST = low_cost


    def get_max_path_reward(self, goal,f=max):
        """
        Get the maximum expected reward on a path to the given goal.
        """

        path_rewards = []
        for path in self.goal_paths[goal]:
            path_reward = 0
            for node in path:
                # Only consider nodes up to the goal
                if node == goal:
                    break
                else:
                    reward = self._state[node]
                    if hasattr(reward, "sample"):
                        path_reward += reward.expectation()
                    else:
                        path_reward += reward
            path_rewards.append(path_reward)
        return f(path_rewards)

    def get_max_path(self, goal, include=False):
        """
        Get the maximum expected reward on a path to the given goal.
        """
        path_rewards = self.get_max_path_reward(goal, np.array)
        if not include:
            reward = self._state[goal]
            if hasattr(reward, "sample"):
                path_rewards -= reward.expectation()
            else:
                path_rewards -= reward
        best_path = np.argmax(path_rewards)
        return self.goal_paths[goal][best_path]

    def get_goal_reward(self, goal):
        """
        Returns the reward distribution for a goal state taking the best path to the goal into account.
        """

        max_path_reward = self.get_max_path_reward(goal)
        # Update the distribution by the value along the path
        goal_state = self._state[goal]
        if hasattr(goal_state, "sample"):
            if hasattr(goal_state, "mu") and hasattr(goal_state, "sigma"):
                return Normal(goal_state.mu + max_path_reward, goal_state.sigma)
            elif hasattr(goal_state, "vals") and hasattr(goal_state, "probs"):
                vals = tuple([value + max_path_reward for value in goal_state.vals])
                return Categorical(vals, goal_state.probs)
            else:
                print(f"Type {type(goal_state)} not supported.")
                raise NotImplementedError()
        else:
            return goal_state + max_path_reward

    def get_goal_distribution(self, goal, n=4):
        """
        Returns the reward distribution for a goal state taking the best path to the goal into account.
        """

        #max_path_reward = self.get_max_path_reward(goal)
        max_path = self.get_max_path(goal, include=True)
        path_reward = [self._state[node] if hasattr(self._state[node], "sample") else Categorical([self._state[node]]) for node in max_path[1:]]
        reward = Categorical([0])
        for r in path_reward:
            reward += r
        return shrink_categorical(reward, n=n)

    def variance(self, node):
        value = self._state[node]
        if hasattr(value, "sample"):
            return value.var()
        else:
            return 0

    def get_goal_myopic_distribution(self, goal, n=4):
        """Finds the best click of the goal subtree through variance heuristic, then calculates the goal distribution based on the distribution of the best click and the expected reward of other nodes on the best path.

        Args:
            goal (int): Node index of the goal node
        """
        max_path = self.get_max_path(goal, include=True)
        max_variance_node = max(max_path, key=self.variance)
        reward = Categorical([0])
        for node in max_path: 
            value = self._state[node]
            if node == max_variance_node:
                reward += value
            elif hasattr(value, "sample"):
                reward += Categorical([value.expectation()])
            else:
                reward += Categorical([value])
        return shrink_categorical(reward, n=n)

    def create_sub_nodes(self, original_nodes):
        """ 
        Creates a new node list out of a given set of nodes and creates a two way mapping between the sets.
        """
        nodes = range(len(original_nodes))

        real_to_sub = {}
        sub_to_real = {}
        for node in original_nodes:
            index = len(real_to_sub)
            real_to_sub[node] = index
            sub_to_real[index] = node
        
        return nodes, sub_to_real, real_to_sub

    def get_goal_MDP(self, high_actions=[], n=4):
        """
        Create a high level MDP containing only the goal states. Rewards are adjusted to incorporate the expected reward along the best path. 

        :returns: tuple(MouselabEnv, dict1, dict2)
            Where:
            MouselabEnv is the created high level MDP
            dict1 is a state mapping dictionary from the goal MDP to the meta MDP
            dict2 is a state mapping dictionary from the meta MDP to the goal MDP
        """

        subtree = [0] + self.goals
        goal_nodes, goal_to_real, real_to_goal = self.create_sub_nodes(subtree)

        goal_tree = [[] for g in goal_nodes]

        # Initialize the state by taking the rewards along the way into account
        goal_ground_truth = [0] + [self.ground_truth[goal_to_real[node]] + self.get_max_path_reward(goal_to_real[node]) for node in goal_nodes[1:]]
        #goal_state = [0] + [self.get_goal_myopic_distribution(goal_to_real[node]) for node in goal_nodes[1:]]
        goal_state = [0] + [self.get_goal_reward(goal_to_real[node]) for node in goal_nodes[1:]]
        #goal_state = [0] + [self.get_goal_distribution(goal_to_real[node]) for node in goal_nodes[1:]]
        
        only_goal_paths = [list(filter(lambda x: x in self.goals, path)) for path in self._all_paths]
        unique_goal_paths = []
        for path in only_goal_paths:
            if path not in unique_goal_paths:
                unique_goal_paths.append(path)

        for path in unique_goal_paths:
            previous = 0
            for value in path:
                value = real_to_goal[value]
                if value not in goal_tree[previous]:
                    goal_tree[previous].append(value)
                    previous = value

        # Contraction method
        # state = [node if hasattr(node, "sample") else Categorical([node]) for node in self._state]
        # goal_dict = {g:g for g in self.goals}
        # operations, goal_tree, real_to_goal = reduce_rec(self.tree, goal_dict)
        # goal_to_real = {v:k for k,v in real_to_goal.items()}

        # estimated_state = compute_goal_state(state, operations)
        # estimated_state = shrink_categorical(estimated_state, n=n)
        # real_state = [0] + [self.get_goal_reward(goal_to_real[node]) for node in range(1, len(goal_tree))]

        # goal_state = [real_state[i] if not hasattr(real_state[i], "sample") else estimated_state[i] for i in range(len(real_state))]

        # goal_ground_truth = [0] + [self.ground_truth[goal_to_real[node]] + self.get_max_path_reward(goal_to_real[node]) for node in range(1, len(goal_state))]

        goal_mdp = MouselabHigh(goal_tree, goal_state, ground_truth=goal_ground_truth, cost=self.HIGH_COST, simple_cost=self.simple_cost, high_actions=high_actions)
        return goal_mdp, goal_to_real, real_to_goal

    def get_low_level_MDP(self, goal, alternative=None):
        """
        Create a new MDP containing the subtree of the given goal. An additional dummy node representing the alternative best reward is inserted.
        
        :returns: tuple(MouselabEnv, dict1, dict2)
            Where:
            MouselabEnv is the created high level MDP
            dict1 is a state mapping dictionary from the low level MDP to the meta MDP
            dict2 is a state mapping dictionary from the meta MDP to the low level MDP
        """

        # Find relevant nodes for subtree
        relevant_nodes = []
        for path in self.goal_paths[goal]:
            for node in path:
                if node not in relevant_nodes:
                    relevant_nodes.append(node)

        sub_nodes, sub_to_real, real_to_sub = self.create_sub_nodes(relevant_nodes)

        sub_ground_truth = [self.ground_truth[sub_to_real[node]] for node in sub_nodes]
        sub_state = [self._state[sub_to_real[node]] for node in sub_nodes]
        sub_tree = [[] for n in sub_nodes]

        # Copy tree structure excluding nodes that aren't part of the subtree
        for node in sub_nodes:
            real_node = sub_to_real[node]
            for child_node in self.tree[real_node]:
                if child_node in relevant_nodes:
                    sub_tree[node].append(real_to_sub[child_node])
        
        meta_disabled = alternative is None
        # Add dummy node
        if not meta_disabled:
            new_node = len(sub_tree)
            sub_tree.append([])
            sub_tree[0].append(new_node)
            sub_ground_truth.append(alternative)#.expectation())
            sub_state.append(alternative)
            sub_to_real[new_node] = None

        sub_mdp = MouselabVar(sub_tree, sub_state, ground_truth=sub_ground_truth, cost=self.LOW_COST, simple_cost=self.simple_cost)
        return sub_mdp, sub_to_real, real_to_sub

class MouselabHigh(MouselabVar):
    """ Wrapper for goal MDPs that optimizes action features by setting VPI action to 0 instead of computing it.
    """

    def __init__(self, TREE, INIT, ground_truth, cost, simple_cost, high_actions):
        super().__init__(TREE, INIT, ground_truth=ground_truth, cost=cost, simple_cost=simple_cost)
        self.high_actions = high_actions
    
    # Discretize before calculating features
    def action_features(self, action, bins=4, state=None):
        """Returns the low action features used for BMPS excluding vpi action

        Arguments:
            action: low level action for computation
            option: option for which computation
            bins: number of bins for discretization
            state: low state for computation
        """
        state = state if state is not None else self._state
        state_disc = self.discretize(state, bins)

        assert state is not None

        if action == self.term_action:
            if self.simple_cost:
                return np.array([
                    0,
                    0,
                    0,
                    0,
                    self.expected_term_reward(state)
                ])
            else:
                return np.array([
                    [0, 0, 0],
                    0,
                    0,
                    0,
                    self.expected_term_reward(state)
                ])

        if self.simple_cost:
            return np.array([
                self.cost(action),
                self.myopic_voc(action, state_disc),
                0,
                self.vpi(state_disc),
                self.expected_term_reward(state)
            ])

        else:
            return np.array([
                self.action_cost(action),
                self.myopic_voc(action, state_disc),
                0,
                self.vpi(state_disc),
                self.expected_term_reward(state)
            ])

def reduce_rec(tree, goal_mapping={}):
    """ Recursively reduces the tree by applying one of two operations: Adding parent and child or multiplying two children together.

    Args:
        tree ([[int]]): Tree structure.
    
    Returns: 
        operations ([(str, int, int)]): Applied operations
        tree([[int]]): Reduced tree.
    """
    def find_add_pair(tree, goal_states):
        # Conditions: 
        # Parent can not have other children
        # Child can not have other parents
        parent_counter = {0: 0}
        potential_parents = []
        for i in range(1, len(tree)): # Excludes root
            node = tree[i]
            for child in node: 
                if child in parent_counter.keys():
                    parent_counter[child] += 1
                else:
                    parent_counter[child] = 1
            if len(node) == 1:
                potential_parents.append(i)
        for parent in potential_parents:
            child = tree[parent][0]
            if parent_counter[child] == 1: 
                if (child not in goal_states) or (parent not in goal_states): # Don't add 2 goal states
                    return True, parent, child
        return False, None, None

    def find_reduce_pair(tree, goal_states):
        # Conditions:
        # Pair has a single, identical child
        # Pair has a single, identical parent
        parent_mapping = {i:[] for i in range(len(tree))}
        potential_pairs = []
        # Find pair candidates with identical child
        for i in range(1, len(tree)): #Excludes root
            node = tree[i]
            # Store parents of each node for later comparison
            for child in node:
                parent_mapping[child] = parent_mapping[child] + [node]
            # Check all previous nodes for a potential pair with the current node
            # Condition: Both have the same single child
            if len(node) == 1:
                for j in range(i):
                    if len(tree[j]) == 1 and (tree[j] == node):
                        potential_pairs.append((j, i))
        for pair in potential_pairs:
            a, b = pair
            parents_a = parent_mapping[a]
            parents_b = parent_mapping[b]
            # Check if both nodes in the pair have the same single parent
            if (len(parents_a) == len(parents_b) == 1) and (parents_a == parents_b):
                if (a not in goal_states) or (b not in goal_states): # Don't combine two goal states
                    return True, a, b
        # Alternative: Find pair with same parent and no children
        for i in range(len(tree)):
            if len(tree[i]) == 0:
                for j in range(i+1, len(tree)):
                    if len(tree[j]) == 0:
                        parents_a = parent_mapping[i]
                        parents_b = parent_mapping[j]
                        if (len(parents_a) == len(parents_b) == 1) and (parents_a == parents_b):
                            if (i not in goal_states) or (j not in goal_states): # Don't combine two goal states
                                return True, i, j
        return False, None, None

    def add_pair(tree, a, b, goal_mapping):
        # Always remove the child
        assert b in tree[a]
        # Copy tree and remove node b
        new_tree = [tree[i] for i in range(len(tree))]
        # Replace a's children with b's children
        new_tree[a] = tree[b]
        new_tree.pop(b)
        # All states greater than a/b are now one index lower - adjust edges
        node_value = max(a, b)
        for i in range(len(new_tree)):
            node = new_tree[i]
            new_tree[i] = [child if child < node_value else child - 1 for child in node]
        new_goal_map = {g:(v if v < node_value else v-1) for g, v in goal_mapping.items()}
        return new_tree, new_goal_map

    def reduce_pair(tree, a, b, goal_mapping):
        # Always remove the higher index
        if a>b:
            a, b = b, a
        # Copy tree and remove node b
        new_tree = [tree[i] for i in range(len(tree)) if i != b]
        # Edit edges
        for i in range(len(new_tree)):
            # Remove b from parent's children
            if b in new_tree[i]:
                new_tree[i] = [j for j in new_tree[i] if j!=b]
            # All states greater than b are now one index lower - adjust edges
            new_tree[i] = [child if child < b else child - 1 for child in new_tree[i]]
            # Remove duplicates
            new_tree[i] = list(set(new_tree[i]))
        new_goal_map = {g:(v if v < b else v-1) for g, v in goal_mapping.items()}
        return new_tree, new_goal_map

    new_tree = [x for x in tree]
    #ew_state = [node if hasattr(node, "sample") else Categorical([node]) for node in state]

    operations = []

    done = False
    while not done:
        num_reductions = 0
        # Priority 1: Merge direct parent child connections through add
        done_add = False
        while not done_add:
            found, a, b = find_add_pair(new_tree, goal_mapping.values())
            if found: 
                new_tree, goal_mapping = add_pair(new_tree, a, b, goal_mapping)
                operations.append(tuple(("add", a, b)))
                num_reductions += 1
            else: 
                done_add = True
        # Priority 2: Merge neighbors with the same parent and child
        done_mul = False
        while not done_mul:
            found, a, b = find_reduce_pair(new_tree, goal_mapping.values())
            if found: 
                new_tree, goal_mapping = reduce_pair(new_tree, a, b, goal_mapping)
                operations.append(tuple(("mul", a, b)))
                num_reductions += 1
            else: 
                done_mul = True
        # If no add or mul pair is found the tree is minimal
        if num_reductions == 0:
            done = True
    return operations, new_tree, goal_mapping

def compute_goal_state(state, operations):
    def reduce_add(state, i, j):
        new_state = [state[i] for i in range(len(state)) if i is not j]
        new_state[i] = new_state[i] + state[j]
        return new_state

    def reduce_mul(state, i, j):
        new_state = [state[i] for i in range(len(state)) if i is not j]
        new_state[i] = cross([state[i], state[j]], max)
        return new_state
    
    for i in range(len(operations)):
        op, a, b = operations[i]
        if op == "add":
            state = reduce_add(state, a, b)
        elif op == "mul":
            state = reduce_mul(state, a, b)
        else:
            assert False
    return state

def shrink_categorical(cat, n=4):
    '''
    Reduces the categorical distribution to distribution of size n using k-means clustering
    :param cat: categorical distribution
    :param n: number of bins/clusters to be reduced to
    :return:
    '''
    if (not hasattr(cat, "sample")) or (len(cat.vals) < n):
        return cat
    clusters, centroids = kmeans1d.cluster(cat.vals, n)
    probs = [0 for _ in range(n)]
    for cluster, prob in zip(clusters, cat.probs):
        probs[cluster] += prob
    return Categorical(centroids, probs=probs)