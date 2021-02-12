from utils.mouselab import MouselabEnv
from utils.distributions import Categorical, PointMass, expectation, cross
import numpy as np
from itertools import compress
from contracts import contract

NO_CACHE = False
if NO_CACHE:
    lru_cache = lambda _: (lambda f: f)
else:
    from functools import lru_cache

CACHE_SIZE = int(2**20)
SMALL_CACHE_SIZE = int(2**14)
ZERO = PointMass(0)

class MouselabVar(MouselabEnv):
    def __init__(self, tree, init, ground_truth=None, cost=0, term_belief=True, sample_term_reward=False, depth_weight = 0, cost_weight = 1, simple_cost=True):
        super().__init__(tree, init, ground_truth=ground_truth, cost=cost, term_belief=term_belief, sample_term_reward=sample_term_reward, depth_weight=depth_weight, cost_weight=cost_weight)
        self.all_paths_ = tuple(tuple(n) for n in self.all_paths())
        _, self.operations = compute_operations(tree, [])
        self.simple_cost = simple_cost
        self._state = self.discretize(self._state, 4)

    def ground_truth_reward(self):
        """Returns the ground truth value of the best possible path.

        Returns:
            float: Reward
        """
        path_rewards = []
        for path in self.all_paths_:
            path_reward = 0
            for n in path:
                path_reward += self.ground_truth[n]
            path_rewards.append(path_reward)
        return np.max(path_rewards)


    # Adapted to take all paths into account
    def node_value_to(self, node, state=None):
        """A distribution over rewards up to and including the given node."""
        state = state if state is not None else self._state
        all_paths = self.path_to(node)
        values = []
        path_rewards = []
        for path in all_paths:
            path_reward = ZERO
            for n in path:
                if hasattr(n, "sample"):
                    path_reward += state[n]
                else:
                    path_reward += state[n]
            path_rewards.append(path_reward)
            values.append(path_reward.expectation())
        best_path = np.argmax(values)
        return path_rewards[best_path]

    # Rewritten to take multiple paths to a node into account
    def path_to(self, node):
        """Returns all paths leading to a given node.

        Args:
            node (int): Target node paths to are searched for

        Returns:
            list of list of int: All paths to the node in a nested list 
        """
        all_paths = self.all_paths()
        node_paths = [p for p in all_paths if node in p]
        # Cut of remaining path after target node
        up_to_node = [p[:p.index(node)+1] for p in node_paths]
        return up_to_node

    # Rewritten to take multiple paths to a node into account
    @lru_cache(CACHE_SIZE)
    @contract
    def vpi_action(self, action, state) -> 'float, >= -0.001':
        # Unique nodes in all paths leading to the action
        paths = self.path_to(action)
        flat_paths = [node for path in paths for node in path]
        obs = (*self.subtree[action], *flat_paths)
        obs = tuple(np.unique(obs))
        return (self.node_value_after_observe(obs, 0, state).expectation()
                - self.expected_term_reward(state)
                )

    # Discretize before calculating features
    def action_features(self, action, bins=4, state=None):
        """Returns the low action features used for BMPS

        Arguments:
            action: low level action for computation
            bins: number of bins for discretization
            state: low state for computation
        """
        state = state if state is not None else self._state
        #state_disc = self.discretize(state, bins)

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
                self.myopic_voc(action, state),
                self.vpi_action(action, state),
                self.vpi(state),
                self.expected_term_reward(state)
            ])

        else:
            return np.array([
                self.action_cost(action),
                self.myopic_voc(action, state), #state_disc),
                self.vpi_action(action, state), #state_disc),
                self.vpi(state),#_disc),
                self.expected_term_reward(state)
            ])

    def discretize(self, state, bins):
        """Discretizes the state space

        Arguments:
            state: state to see which actions possible
            bins: number of bins for discretization
        """
        state_disc = []
        for s in state:
            if hasattr(s, 'sample') and hasattr(s, 'mu'):
                dist = s.to_discrete(n=bins, max_sigma=4)
                dist.vals = tuple([(round(val, 3)) for val in dist.vals])
                dist.probs = tuple([(round(p, 3)) for p in dist.probs])
                state_disc.append(dist)
            else:
                state_disc.append(s)
        return tuple(state_disc)

    # Remove duplicates
    def _get_subtree(self):
        def gen(n):
            yield n
            for n1 in self.tree[n]:
                yield from gen(n1)
        subtree = [tuple(gen(n)) for n in range(len(self.tree))]
        for i, sub in enumerate(subtree):
            subtree[i] = tuple(np.unique(sub))
        return subtree


    def action_cost(self, action):
        """Returns the estimated number of clicks needed for action features computation in BMPS

        Arguments:
            action: low level action for computation
        """
        
        paths = self.path_to(action)
        flat_paths = [node for path in paths for node in path]
        obs = (*self.subtree[action], *flat_paths)
        obs = np.unique(obs)
        vpi_action_nodes = (len(obs)-1) * self.cost(action)
        vpi_nodes = (len(self.subtree[0])-1) * self.cost(action)
        myopic_nodes = self.cost(action)
        return [myopic_nodes, vpi_action_nodes, vpi_nodes]

    def to_obs_tree(self, state, node, obs=()):
        """Updated obs tree computation for tree contraction method.
        """
        state = [state[n] if n in obs else expectation(state[n]) for n in range(len(state))]
        state = [node if hasattr(node, "sample") else Categorical([node]) for node in state]
        return tuple(state)

    def node_value_after_observe(self, obs, node, state):
        """A distribution over the expected value of node, after making an observation.
        
        obs can be a single node, a list of nodes, or 'all'
        """
        obs_tree = self.to_obs_tree(state, node, obs)
        if self.exact:
            return self.compute_node_value(obs_tree)
        else:
            assert False

    @lru_cache(16384)
    def compute_node_value(self, obs_tree):
        """ Wrapper for node compute function independent of precomputed operations to ensure proper caching.

        """
        return exact_node_value_after_observe(obs_tree, self.operations)

####### TREE REDUCE

def exact_node_value_after_observe(state, operations):
    """ Computes the categorical node value of the tree by applying the passed operations.

    Args:
        state ([Categorical]): Node value distribution based on the observation.
        operations ([(str, int, int)]): Operations to be applied.

    Returns:
        result (Categorical): Categorical node value of the final tree node. 
    """
    def reduce_add(state, i, j):
        new_state = [state[i] for i in range(len(state)) if i is not j]
        new_state[i] = new_state[i] + state[j]
        return new_state

    def reduce_mul(state, i, j):
        new_state = [state[i] for i in range(len(state)) if i is not j]
        new_state[i] = cross([state[i], state[j]], max)
        return new_state

    def split_obs(obs, x, num_parents):
        node_var: Categorical = obs[x]
        obs_values = []
        for v in zip(node_var.vals):
            obs_copy = [o for o in obs]
            obs_copy[x] = Categorical(v)
            # Copy obs to new nodes
            for i in range(num_parents - 1):
                obs_copy.append(Categorical(v))
            obs_values.append(obs_copy)
        probs = node_var.probs
        return probs, obs_values
    
    for i in range(len(operations)):
        op, a, b = operations[i]
        if op == "add":
            state = reduce_add(state, a, b)
        elif op == "mul":
            state = reduce_mul(state, a, b)
        elif op == "split":
            # a = split, b = num parents
            probs, obs_vals = split_obs(state, a, b)
            states = []
            # Solve partial trees recursively
            for val in obs_vals:
                states.append(exact_node_value_after_observe(val, operations[i+1:]))
            # Combine partial responses
            total_p = []
            total_v = []
            for var_p, cat in zip(probs, states):
                assert len(cat) == 1
                p, v = cat[0].probs, cat[0].vals
                p = [x*var_p for x in p]
                total_p += p
                total_v += v
            state = [Categorical(total_v, probs=total_p)]
            break
        else:
            assert False
    assert len(state) == 1
    return state[0]

def reduce_rec(tree):
    """ Recursively reduces the tree by applying one of two operations: Adding parent and child or multiplying two children together.

    Args:
        tree ([[int]]): Tree structure.
    
    Returns: 
        operations ([(str, int, int)]): Applied operations
        tree([[int]]): Reduced tree.
    """
    def find_add_pair(tree):
        # Conditions: 
        # Parent can not have other children
        # Child can not have other parents
        parent_counter = {0: 0}
        potential_parents = []
        for i in range(len(tree)):
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
                return True, parent, child
        return False, None, None

    def find_reduce_pair(tree):
        # Conditions:
        # Pair has a single, identical child
        # Pair has a single, identical parent
        parent_mapping = {i:[] for i in range(len(tree))}
        potential_pairs = []
        # Find pair candidates with identical child
        for i in range(len(tree)):
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
                return True, a, b
        # Alternative: Find pair with same parent and no children
        for i in range(len(tree)):
            if len(tree[i]) == 0:
                for j in range(i+1, len(tree)):
                    if len(tree[j]) == 0:
                        parents_a = parent_mapping[i]
                        parents_b = parent_mapping[j]
                        if (len(parents_a) == len(parents_b) == 1) and (parents_a == parents_b):
                            return True, i, j
        return False, None, None

    def add_pair(tree, a, b):
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
        return new_tree

    def reduce_pair(tree, a, b):
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
        return new_tree

    new_tree = [x for x in tree]
    #ew_state = [node if hasattr(node, "sample") else Categorical([node]) for node in state]

    operations = []

    done = False
    while not done:
        num_reductions = 0
        # Priority 1: Merge direct parent child connections through add
        done_add = False
        while not done_add:
            found, a, b = find_add_pair(new_tree)
            if found: 
                new_tree = add_pair(new_tree, a, b)
                operations.append(tuple(("add", a, b)))
                num_reductions += 1
            else: 
                done_add = True
        # Priority 2: Merge neighbors with the same parent and child
        done_mul = False
        while not done_mul:
            found, a, b = find_reduce_pair(new_tree)
            if found: 
                new_tree = reduce_pair(new_tree, a, b)
                operations.append(tuple(("mul", a, b)))
                num_reductions += 1
            else: 
                done_mul = True
        # If no add or mul pair is found the tree is minimal
        if num_reductions == 0:
            done = True
    return operations, new_tree

def compute_operations(tree, operations = []):
    """ Computes the operations needed to reduce the tree down to a single node.

    Args:
        tree ([[int]]): Tree structure as passed when initializing the MDP.
        operations (list, optional): Previous calculated operations for recursive calls. The first call should pass an empty list.

    Returns:
        tree: Partially reduced tree used in recursive calls. 
        operations ([tuple]): List of tuples containing the operations used and two additional parameters specifying the affected nodes.
    """
    def find_split_node(tree):
        leafs = [i for i in range(len(tree)) if tree[i] == []]
        for leaf in leafs:
            num_parents = 0
            for node in tree: 
                if leaf in node: 
                    num_parents += 1
            if num_parents > 1: 
                return leaf
        return None

    def split_tree(tree, x):
        tree = [node for node in tree]
        assert tree[x] == [] # Only leaf nodes work
        num_parents = 0 
        for i in range(len(tree)):
            if (i != x) and (x in tree[i]):
                # If more than one parent - add new node as child for other parent
                if num_parents != 0: 
                    tree[i] = [node for node in tree[i] if x != node]
                    tree[i].append(len(tree) - 1 + num_parents)
                num_parents += 1
        # New children for additional parent nodes
        for i in range(num_parents - 1):
            tree.append([])
        return tree, num_parents

    # Find mul - add operations
    op, tree = reduce_rec(tree)
    operations += op
    # Check if done
    if len(tree) == 1:
        return tree, operations
    # Otherwise, find a node to split into two trees
    split = find_split_node(tree)
    assert split != None # Some trees might not work yet
    tree, num_parents = split_tree(tree, split)
    operations += [("split", split, num_parents)]
    # Recursively find next operations after split
    return compute_operations(tree, operations)

######## SLOW BRUTE FORCE METHOD
#     def to_obs_tree(self, state, node, obs=()):
#         state = [state[n] if n in obs else expectation(state[n]) for n in range(len(state))]
#         obs_tree = tuple((node.vals) if hasattr(node, "sample") else (node,) for node in state)
#         probs = tuple((node.probs) if hasattr(node, "sample") else (1,) for node in state)
#         del state
#         return obs_tree, probs

#     def node_value_after_observe(self, obs, node, state):
#         """A distribution over the expected value of node, after making an observation.
        
#         obs can be a single node, a list of nodes, or 'all'
#         """
#         obs_tree, probs = self.to_obs_tree(state, node, obs)
#         if self.exact:
#             return exact_node_value_after_observe(obs_tree, probs, self.all_paths_)
#         else:
#             assert False

# #@lru_cache(4096)
# def exact_node_value_after_observe(obs_tree, probs, paths):
#     value_combinations = np.array(np.meshgrid(*obs_tree)).T.reshape(-1, len(obs_tree))
#     prob_combinations = np.prod(np.array(np.meshgrid(*probs)).T.reshape(-1, len(probs)), axis=1)
    
#     res = [np.sum(np.take(value_combinations, path, axis=1), axis=1) for path in paths]

#     path_combinations = np.array(res).T
#     max_per_combination = np.max(path_combinations, axis=1)
#     res = np.sum(max_per_combination * prob_combinations)
#     del value_combinations, prob_combinations, path_combinations, obs_tree, probs
#     return Categorical([res])