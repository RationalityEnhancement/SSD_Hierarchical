from collections import namedtuple, defaultdict, deque, Counter
import numpy as np
import gym
from gym import spaces
from utils.distributions import smax, cmax, cross_1, sample, expectation, Normal, PointMass, Categorical
from toolz import memoize, get
import random
from contracts import contract
import math
import itertools as it
from toolz import reduce
from itertools import compress

NO_CACHE = False
if NO_CACHE:
    lru_cache = lambda _: (lambda f: f)
else:
    from functools import lru_cache

CACHE_SIZE = int(2 ** 16)
SMALL_CACHE_SIZE = int(2 ** 14)
ZERO = PointMass(0)


class MouselabEnv(gym.Env):
    """MetaMDP for a tree with a discrete unobserved reward function."""
    metadata = {'render.modes': ['human', 'array']}
    term_state = '__term_state__'

    def __init__(self, tree, init, no_options, goals, option_set, env_type , ground_truth=None, branch_cost=0, switch_cost=0, tau=20, sample_term_reward=False, repeat_cost=1):

        self.init = (0, *init[1:])
        if ground_truth is not None:
            if len(ground_truth) != len(init):
                raise ValueError('len(ground_truth) != len(init)')
            self.ground_truth = np.array(ground_truth)
        else:
            self.ground_truth = np.array(list(map(sample, init)))
        self.ground_truth[0] = 0.
        self.cost = - abs(branch_cost)
        self.switch_cost = - abs(switch_cost)
        self.sample_term_reward = sample_term_reward
        self.high_term_action = len(init) + 2 * no_options  # low level action + high level actions (-1 + 1)
        low_term_actions = []
        low_action_space = []
        for i in range(no_options):
            length = len(option_set[i]) + 1
            low_term_actions.append(i + len(self.init))
            low_action_space.append(spaces.Discrete(length))

        self.low_term_actions = tuple(low_term_actions)

        # Required for gym.Env API.
        self.high_action_space = spaces.Discrete(no_options + 1)
        self.low_action_space = tuple(low_action_space)

        self.initial_states = None  # TODO
        self.exact = True  # TODO
        self.tree = tree
        self.subtree = self._get_subtree()
        self.tau = tau
        self.no_options = no_options
        self.goals = goals
        self.option_set = option_set
        self.env_type = env_type
        self.in_option = False
        self.selected_option = 0

        self._reset()

    def _get_subtree(self):
        """Generate subtrees for each node
        """
        def gen(n):
            yield n
            for n1 in self.tree[n]:
                yield from gen(n1)
        return [tuple(gen(n)) for n in range(len(self.tree))]

    def _reset(self):
        """Reset the environment
        """
        if self.initial_states:
            self.init = random.choice(self.initial_states)
        self.paths = self._paths()
        self.low_state = self.init
        self.past_utility = 0
        self.in_option = False
        high_state = {}

        self.high_state = high_state  # Temporary

        # print("0th self.high_state = {} high_state = {}".format(self.high_state, high_state))
        for op in range(1, self.no_options+1):
            high_state[op] = self.high_belief_update(self.init, op)
            # print("self.high_state = {} high_state = {}".format(self.high_state, high_state))
        self.high_state = high_state

        obs_list = []
        for i in range(len(self.init)):
            obs_list.append([])
        self.obs_list = obs_list
        self.tree_no_goals = self.no_goal_tree()
        return self.low_state, self.high_state

    def no_goal_tree(self):
        """Generate tree removing goal nodes
        """
        def which_goal(node, goals):
            for i, g in enumerate(goals):
                if g in node:
                    return goals[i]
            return -1  # Error

        def common_member(a, b):
            a_set = set(a)
            b_set = set(b)
            if len(a_set.intersection(b_set)) > 0:
                return True
            return False

        goals = []
        for g in self.goals:
            goals.append(g[0])

        tree_no_goals = []
        for i, n in enumerate(self.tree):
            if not (common_member(goals, n)):
                tree_no_goals.append(n)
            else:
                goal = which_goal(n, goals)
                new_n = []
                for x in n:
                    if x != goal:
                        new_n.append(x)
                tree_no_goals.append(new_n)
        return tree_no_goals

    def _paths(self, start=0):
        """Generate paths

        Arguments:
            start: to define starting node
        """
        alp = [p for _, p in enumerate(self.all_paths())]
        paths = []
        for node in range(len(self.init)):
            path = [start]
            if node == start or node in self.tree[0]:
                paths.append(path)
            else:
                in_path = [node in a for _, a in enumerate(alp)]
                path_to_check = list(compress(alp, in_path))
                for _, p in enumerate(path_to_check):
                    path.extend(p[1:p.index(node)])
                paths.append(path)

        goal_paths = []
        for _, a in enumerate(alp):
            goal_paths.append(a[-1])
        self.goal_paths = tuple(goal_paths)
        self.allpaths = tuple(alp)
        return tuple(paths)

    def all_paths(self, start=0):
        """All paths from start node

        Arguments:
            start: to define starting node
        """
        def rec(path):
            children = self.tree[path[-1]]
            if children:
                for child in children:
                    yield from rec(path + [child])
            else:
                yield path

        return rec([start])

    def low_step(self, action):
        if (self.in_option == False):
            assert 0, 'Cant take low level step: First select Option'
        check_list = self.option_set[self.selected_option - 1] + [self.low_term_actions[self.selected_option - 1]]
        # print(check_list)
        if (action not in check_list):
            assert 0, 'Cant take low level step: action not in option set'
        if self.low_state is self.term_state:
            assert 0, 'state is terminal'
        if action == self.low_term_actions[self.selected_option - 1]:
            self.in_option = False  # Exiting option
            option = action - len(self.init) + 1
            reward,_ = self.low_term_reward(option, self.low_state)
            done = True
            obs = False
        else:  # observe a node
            self.in_option = True
            self.low_state = self._observe(action)
            option = self.low_term_actions[self.selected_option - 1] - len(self.init) + 1
            self.high_state[option] = self.high_belief_update(self.low_state, option)  # Update high level belief state
            reward = self.cost
            done = False
            obs = False
        return self.low_state, reward, done, obs


    def low_step_test(self, action):
            if (self.in_option == False):
                assert 0, 'Cant take low level step: First select Option'
            check_list = self.option_set[self.selected_option - 1] + [self.low_term_actions[self.selected_option - 1]]
            # print(check_list)
            if (action not in check_list):
                assert 0, 'Cant take low level step: action not in option set'
            if self.low_state is self.term_state:
                assert 0, 'state is terminal'
            if action == self.low_term_actions[self.selected_option - 1]:
                self.in_option = False  # Exiting option
                option = action - len(self.init) + 1
                reward = self.low_term_reward_actual(option, self.low_state)
                done = True
                obs = False
            else:  # observe a node
                self.in_option = True
                self.low_state = self._observe(action)
                option = self.low_term_actions[self.selected_option - 1] - len(self.init) + 1
                self.high_state[option] = self.high_belief_update(self.low_state, option)  # Update high level belief state
                reward = self.cost
                done = False
                obs = False
            return self.low_state, reward, done, obs

    def high_term_reward_psuedo(self, option):
        """Psuedo high level termination reward

        Arguments:
            option: option selected
        """
        return - math.sqrt(self.high_state[option][1]) / 1000

    def high_step(self, action):
        """Perform high level step

        Arguments:
            action: action selected
        """
        if self.in_option != False:
            assert 0, 'Cant take high level step'
        if self.high_state is self.term_state:
            assert 0, 'state is terminal'
        if action < len(self.init) + self.no_options:
            assert 0, 'Action selected is for low_step'
        if action == self.high_term_action:  # Terminate High
            self.in_option = True  # Enter option
            high_mean = []
            for i in self.high_state:
                high_mean.append(self.high_state[i][0])
            option_index = np.array(high_mean).argmax()
            self.selected_option = option_index + 1
            reward = self.high_term_reward_psuedo(option_index + 1)
            done = True
            obs = False
        else:  # observe goal state
            option = action - (len(self.init) + self.no_options - 1)
            goal_to_observe = self.goals[option - 1][0]
            self.low_state = self._observe(goal_to_observe)
            self.high_state[option] = self.high_belief_update(self.low_state, option)
            reward = self.switch_cost
            done = False
            obs = True
        return self.high_state, reward, done, obs

    def _observe(self, action):
        """Observe node value for action specified

        Arguments:
            action: action for observation
        """
        if self.ground_truth is not None:
            result = self.ground_truth[action]
        else:
            result = self.low_state[action].sample()
        s = list(self.low_state)
        s[action] = result
        return tuple(s)

    def low_term_reward(self, option, low_state): #Compare in value in beginning
        """Termination reward according to current belief obtained following current belief

        Arguments:
            option: option for computation
            low_state: low state for computation
        """
        goal = self.goals[option-1][0]
        path_to_check = [goal in p for p in self.allpaths]
        paths = compress(self.allpaths, path_to_check)
        rewards = []
        vars_list = []
        for path in paths:
            reward = 0
            var = 0
            for x in path:
                if(hasattr(low_state[x], 'sample')):
                    reward += low_state[x].expectation()
                    var += low_state[x].sigma ** 2 # Assuming Normal
                else:
                    reward += low_state[x]
            rewards.append(reward)
            vars_list.append(var)
        max_reward = max(rewards)
        m = max(rewards)
        index = [i for i, j in enumerate(rewards) if j == m]
        min_var = min([vars_list[i] for i in index])
        return max_reward, min_var

    def low_term_reward_actual(self, option, low_state):
        """Actual termination reward obtained following current belief

        Arguments:
            option: option for computation
            low_state: low state for computation
        """
        goal = self.goals[option - 1][0]
        path_to_check = [goal in p for p in self.allpaths]
        paths = compress(self.allpaths, path_to_check)

        actual_rewards = []
        believed_rewards = []
        for path in paths:
            believed_reward = 0
            for p in path:
                if hasattr(low_state[p], 'sample'):  # From distribution
                    believed_reward += low_state[p].expectation()
                else:
                    believed_reward += low_state[p]
            believed_rewards.append(believed_reward)
            actual_rewards.append(sum([self.ground_truth[p] for p in path]))
        index = np.argmax(believed_rewards)
        # print("Believed Max Reward = {}".format(believed_rewards[index]))
        return actual_rewards[index]

    def high_term_reward(self):
        """Termination reward of high policy
        """
        actual_rewards = []
        believed_rewards = []
        for path in self.allpaths:
            believed_reward = 0
            for p in path:
                if hasattr(self.low_state[p], 'sample'):  # From distribution
                    believed_reward += self.low_state[p].expectation()
                else:
                    believed_reward += self.low_state[p]
            believed_rewards.append(believed_reward)
            actual_rewards.append(sum([self.ground_truth[p] for p in path]))
        index = np.argmax(believed_rewards)
        # print("Believed Max Reward = {}".format(believed_rewards[index]))
        return actual_rewards[index]

    def high_belief_update(self, low_state, option):
        """Update the high belief state

        Arguments:
            low_state: low state
            option: option to update
        """
        val, var = self.low_term_reward(option, low_state)
        return [val, var]

    def path_to(self, node):
        """Returns a path to specified node

        Arguments:
            node: node to get path to
        """
        return self.paths[node]

    @classmethod
    def new(cls, no_options, tree, reward, seed=None, env_type="manual", option_set=None, initval=None, **kwargs):
        """Returns a MouselabEnv with a symmetric structure.

        Arguments:
            no_options: number of options
            tree: tree structure for the environment
            reward: reward function to set reward distribution of nodes
            seed: seed value, useful when sampling the node value
            env_type: environment type
            option_set: to define option set manually
            initval: to set the reward for each node manually
        """

        if seed is not None:
            np.random.seed(seed)
        if not callable(reward):
            r = reward
            reward = lambda depth: r

        init = []
        for i in range(len(tree)):
            init.append(reward(i))

        if initval is not None:
            init = initval
            
        # Goals are the leaf nodes
        goals = []
        for i,el in enumerate(tree):
            if len(el) == 0:
                goals.append([i])

        # Recursively gather the parents of a given node in the tree
        def get_parents(tree, state):
            parents = []
            if state != 0:
                for i, el in enumerate(tree):
                    if state in el:
                        # Index = node
                        parents.append(i)
                rec_parents = []
                for p in parents:
                    rec_parents += get_parents(tree, p)
                return parents + rec_parents
            return []

        if option_set is None:
            opt = []
            for i, g in enumerate(goals):
                parents = get_parents(tree, g[0])
                # Unique parents excluding the root node
                option = np.unique(np.array(parents))[1:]
                opt.append(option.tolist())
        else:
            opt = option_set
        return cls(tuple(tree), init, no_options, tuple(goals), tuple(opt), env_type, **kwargs)

    def _render(self, mode='notebook', close=False):
        """
        Renders the environment structute
        """
        if close:
            return
        from graphviz import Digraph

        def color(val):

            if val > 0:
                return '#8EBF87'
            else:
                return '#F7BDC4'

        dot = Digraph()
        for x, ys in enumerate(self.tree):
            r = self.low_state[x]
            observed = not hasattr(self.low_state[x], 'sample')
            c = color(r) if observed else 'grey'
            label = str(round(r, 2)) if observed else str(x)
            dot.node(str(x), label=label, style='filled', color=c)
            for y in ys:
                dot.edge(str(x), str(y))
        return dot

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

    def low_action(self, state):
        """Generates low actions possible

        Arguments:
            state: state to see which actions possible
        """
        possible = self.option_set[self.selected_option - 1]
        possible_actions = [x for x in possible if hasattr(state[x], 'sample')]
        possible_actions = possible_actions + [self.low_term_actions[self.selected_option - 1]]
        for i in possible_actions:
            yield i

    def cost_no_nodes(self, option, action):
        """Returns the estimated number of clicks needed for action features computation in BMPS

        Arguments:
            option: option for which computation
            action: low level action for computation
        """
        obs = (*self.subtree[action][0:], *self.path_to(action)[1:])
        obs = list(set(obs))
        vpi_action_nodes = len(obs) * self.cost
        action = self.goals[option - 1][0]
        obs = (*self.subtree[action][0:], *self.path_to(action)[1:])
        obs = list(set(obs))
        vpi_nodes = len(obs) * self.cost
        myopic_nodes = 1 * self.cost
        return [myopic_nodes, vpi_action_nodes, vpi_nodes]
        

    def low_action_features(self, action, option, bins=257, state=None):
        """Returns the low action features used for BMPS

        Arguments:
            action: low level action for computation
            option: option for which computation
            bins: number of bins for discretization
            state: low state for computation
        """
        state = state if state is not None else self.low_state
        state_disc = self.discretize(state, bins)

        assert state is not None

        if action == self.low_term_actions[option-1]:
            return np.array([
                [0, 0, 0],
                0,
                0,
                0,
                self.expected_low_term_reward(option, state)
            ])

        return np.array([
            self.cost_no_nodes(option, action),
            self.low_myopic_voc(action, option, state_disc),
            self.low_vpi_action(action, option, state_disc),
            self.low_vpi(option, state_disc),
            self.expected_low_term_reward(option, state)
        ])

    def expected_high_term_reward(self, high_state):
        """Returns the expected high termination reward

        Arguments:
            high_state: high state for computation
        """
        option_index = np.array(high_state).argmax()
        expected_high_term_reward = high_state[option_index]
        return expected_high_term_reward

    def high_myopic_voc(self, state, action, bins=4):
        """Returns the high level myopic VOC

        Arguments:
            state: high state for computation
            action: high level action for computation
            bins: number of bins to discretize continuous distribution
        """
        option_to_explore = action - (len(self.init) + self.no_options - 1)
        goal_clicked = self.goals[option_to_explore - 1][0]
        node = self.low_state[goal_clicked]
        if hasattr(node, 'sample'):
            if hasattr(node, 'mu'):
                dist = node.to_discrete(n=bins, max_sigma=4)
                dist.vals = tuple([(round(val, 3)) for val in dist.vals])
                dist.probs = tuple([(round(p, 3)) for p in dist.probs])
            else:
                dist = node
        else:
            dist = Categorical(vals=[node], probs=[1])

        r, p = zip(*dist)
        expected_return = 0
        high_state = []
        for op in range(1, self.no_options + 1):
            val, _ = self.low_term_reward(op, self.low_state)
            high_state.append(val)

        for k in range(len(p)):  # Find best option to explore for each possible goal value
            state2 = list(self.low_state)
            state2[goal_clicked] = r[k]
            high_state[option_to_explore - 1], _ = self.high_belief_update(state2, option_to_explore)
            expected_return += p[k] * max(high_state)
        return expected_return - self.expected_high_term_reward(state)

    def shrink(self, option_dist):
        if len(option_dist) == 2:
            return option_dist
        else:
            #         print("Continuing")
            two_dist = [option_dist[0], option_dist[1]]
            #         print(two_dist)
            new_dist = [cmax(two_dist, default=ZERO)] + option_dist[2:]
            return self.shrink(new_dist)

    @lru_cache(CACHE_SIZE)
    def high_vpi(self, state, bins=4):
        """Returns the high level VPI

        Arguments:
            state: high state for computation
            bins: number of bins to discretize continuous distribution
        """
        dists = []
        for option in range(1, self.no_options + 1):  # To get the node distributions
            goal_clicked = self.goals[option - 1][0]
            node = self.low_state[goal_clicked]
            if hasattr(node, 'sample'):
                if hasattr(node, 'mu'):
                    dist = node.to_discrete(n=bins, max_sigma=4)
                    dist.vals = tuple([(round(val, 3)) for val in dist.vals])
                    dist.probs = tuple([(round(p, 3)) for p in dist.probs])
                else:
                    dist = node
            else:
                dist = Categorical(vals=[node], probs=[1])
            dists.append(dist)
        net_dist = self.shrink(dists)
        expected_return = cmax(net_dist).expectation()
        return expected_return - self.expected_high_term_reward(state)

    def high_action_features(self, action, state=None):
        """Returns the high action features used for BMPS

        Arguments:
            action: high level action for computation
            state: high state for computation
        """
        state = state if state is not None else self.high_state
        state1 = []
        for i in state:
            state1.append(state[i][0])
        state1 = tuple(state1)
        if action == self.high_term_action:
            return np.array([
                0,  # Cost
                0,  # Myopic VOC
                0,  # VPI
                self.expected_high_term_reward(state1)
            ])

        return np.array([
            self.switch_cost,
            self.high_myopic_voc(state1, action),
            self.high_vpi(state1),
            self.expected_high_term_reward(state1)
            ])

    def results(self, state, action, bins):
        """Returns a list of possible results of taking action in state.
        Each outcome is (probability, next_state, reward).

        Arguments:
            state: state for computation
            action: action for computation
            bins: number of bins for discretization
        """
        print("Action = {}".format(action))
        state_des = self.discretize(state, bins=bins)
        if action == self.low_term_actions[self.selected_option-1]:

            yield (1, self.term_state, self.expected_low_term_reward_disc(self.selected_option, state_des))
        else:
            for r, p in state_des[action]:
                s1 = list(state_des)
                s1[action] = r
                yield (p, tuple(s1), self.cost)

    @lru_cache(CACHE_SIZE)
    def expected_low_term_reward(self, option, low_state):
        """The expected termination reward for an option

       Arguments:
            option: option for computation
            low_state: state for computation
       """
        val, _ = self.low_term_reward(option, low_state)
        if abs(val) < 0.001:
            val = 0.0
        return val

    @lru_cache(CACHE_SIZE)
    def expected_low_term_reward_disc(self, option, state):
        """The expected termination reward for an option computing using a discretized state

       Arguments:
            option: option for computation
            state: state for computation
       """
        # print(self.term_reward(option, state))
        return self.term_reward(option, state).expectation()

    def term_reward(self, option, state):
        """A distribution over the return gained by acting given a belief state.

        Arguments:
             option: option for computation
             state: state for computation
        """
        return self.node_value(0, option, state)

    def node_value(self, node, option, state):
        """A distribution over total rewards after the given node.

        Arguments:
             node: node for computation
             option: option for computation
             state: state for computation
        """
        if node == 0:
            nodes = []
            for c in self.tree[node]:
                if c in self.option_set[option-1]:
                    nodes.append(c)
        else:
            nodes = self.tree[node]
        return max((self.node_value(n1, option, state) + state[n1]
                    for n1 in nodes),
                   default=ZERO, key=expectation)

    @contract
    def low_myopic_voc(self, action, option, state) -> 'float, >= -0.001':
        """
         Calculates myopic voc for a given option. Using literal definition of myopic voc. Take the expectation of
         best possible action after observing the node

         Arguments:
             action: action for computation
             option: option for computation
             state: state for computation
         """
        # print("Action = {}, state = {}".format(action, state))
        if hasattr(state[action], 'sample'):
            possible = list(state[action])
            r, p = zip(*possible)
            expected_reward = 0
            for k in range(len(p)):
                state2 = list(state)
                state2[action] = r[k]
                expected_reward += p[k] * self.expected_low_term_reward_disc(option, tuple(state2))
            result = float(expected_reward - self.expected_low_term_reward_disc(option, state))
            if abs(result) < 0.001:
                result = 0.0
            return result
        else:
            return 0.0

    @contract
    def low_vpi_action(self, action, option, state) -> 'float, >= -0.001':
        """
         Calculates vpi_action for a given option.  Nodes of importance are those who are either
         parents or children of the node selected

         Arguments:
             action: action for computation
             option: option for computation
             state: state for computation
         """
        obs = (*self.subtree[action][0:], *self.path_to(action)[1:])
        obs = list(set(obs))
        op_dist = self.node_value_after_observe_option(option, 0, state, obs)
        node_idx = self.goals[option - 1][0]
        if not hasattr(state[node_idx], 'sample'):
            goal_dist = Categorical(vals=[state[node_idx]], probs=[1])
        else:
            goal_dist = state[node_idx]
        dists = [op_dist, goal_dist]
        nvao = float((cross_1(dists, sum)).expectation())
        result = nvao - self.expected_low_term_reward_disc(option, state)

        if abs(result) < 0.001:
            result = 0.0

        return result

    @lru_cache(CACHE_SIZE)
    @contract
    def low_vpi(self, option, state) -> 'float, >= -0.001':
        """
        Calculates vpi for a given option. All nodes of branch in option set are important.
        Basically calculating vpi_action with goal node selected

        Arguments:
            option: option for computation
            state: state for computation
        """
        action = self.goals[option - 1][0]
        obs = (*self.subtree[action][0:], *self.path_to(action)[1:])
        obs = list(set(obs))
        op_dist = self.node_value_after_observe_option(option, 0, state, obs)
        node_idx = self.goals[option-1][0]
        if not hasattr(state[node_idx], 'sample'):
            goal_dist = Categorical(vals=[state[node_idx]], probs=[1])
        else:
            goal_dist = state[node_idx]
        dists = [op_dist, goal_dist]
        nvao = float((cross_1(dists, sum)).expectation())
        result = nvao - self.expected_low_term_reward_disc(option, state)

        if abs(result) < 0.001:
            result = 0.0
        return result

    def to_obs_tree(self, state, node, option, obs=(), sort=True):
        """
        Takes the number of nodes that can be observed and creates multi-nested tuple for each possible path
        Nodes which aren't observable are set to its expected value (normally 0)

        Arguments:
            state: state to generate observation tree from
            node: node to generate obs_tree from
            option: option being explored
            obs: obs can be a single node, a list of nodes, or 'all'
            sort: flag to sort the observation tree generated
        """
        # print("IN TO OBS TREE")
        maybe_sort = sorted if sort else lambda x: x

        def rec(n):
            # print("In rec({})".format(n))
            subjective_reward = state[n] if n in obs else expectation(state[n])
            # print("state[n] = {}".format(state[n]))
            # print("Subjective Reward = {}, n in obs: {} state[n] = {}, expectation(state[n]) = {}"
            # .format(subjective_reward, n in obs, state[n], expectation(state[n])))
            # print("n = {} self.tree[n] = {}".format(n, self.tree[n]))
            if n == 0:
                child_nodes = []
                for ch in self.tree_no_goals[n]:
                    if ch in self.option_set[option-1]:
                        child_nodes.append(ch)
            else:
                child_nodes = self.tree_no_goals[n]
            children = tuple(maybe_sort(rec(c) for c in child_nodes))
            # print("Children = {}".format(children))
            # print("Returning for n={} RETURNED = {}".format(n,(subjective_reward, children)))
            return subjective_reward, children
        return rec(node)

    def node_value_after_observe_option(self, option, node, state, obs):
        """A distribution over the expected value of node considering only an option, after making an observation.

        Arguments:
            option: option being explored
            node: node to generate obs_tree from
            state: state to compute value from
            obs: obs can be a single node, a list of nodes, or 'all'
        """
        obs_tree = self.to_obs_tree(state, node, option, obs)
        # print("Option: {}, OBS: {}, OBS_TREE: {}".format(option,obs, obs_tree))
        if self.exact:
            return exact_node_value_after_observe(obs_tree)
        else:
            return node_value_after_observe(obs_tree)

    def option_actual_path(self, option, low_state):
        """Actual path followed by agent in an option according to belief state

        Arguments:
            option: option to find path in
            low_state: Low level belief state
        """
        goal = self.goals[option - 1][0]
        path_to_check = [goal in p for p in self.allpaths]
        paths = compress(self.allpaths, path_to_check)
        path_list = []
        actual_rewards = []
        believed_rewards = []
        for path in paths:
            path_list.append(path)
            believed_reward = 0
            for p in path:
                if hasattr(low_state[p], 'sample'):  # From distribution
                    believed_reward += low_state[p].expectation()
                else:
                    believed_reward += low_state[p]
            believed_rewards.append(believed_reward)
            actual_rewards.append(sum([self.ground_truth[p] for p in path]))
            index = np.argmax(believed_rewards)
        # print("Believed Max Reward = {}".format(believed_rewards[index]))
        return path_list[index]

    def actual_path(self, low_state):
        """Actual path followed by agent according to belief state

        Arguments:
            low_state: Low level belief state
        """
        path_list = []
        actual_rewards = []
        believed_rewards = []
        for path in self.allpaths:
            #  print("Path = {}".format(path))
            path_list.append(path)
            believed_reward = 0
            for p in path:
                if hasattr(low_state[p], 'sample'):  # From distribution
                    believed_reward += low_state[p].expectation()
                else:
                    believed_reward += low_state[p]
            believed_rewards.append(believed_reward)
            #  print("Believed Reward = {}".format(believed_reward))
            actual_rewards.append(sum([self.ground_truth[p] for p in path]))
            index = np.argmax(believed_rewards)
        # print("Believed Max Reward = {}".format(believed_rewards[index]))
        return path_list[index]

@lru_cache(SMALL_CACHE_SIZE)
def node_value_after_observe(obs_tree):
    """A distribution over the expected value of node, after making an observation.

    Arguments:
            obs_tree: the tree indicating the nodes used for observation
    """
    children = tuple(node_value_after_observe(c) + c[0] for c in obs_tree[1])
    return smax(children, default=ZERO)


@lru_cache(None)
def exact_node_value_after_observe(obs_tree):
    """A distribution over the expected value of node, after making an observation.

    Arguments:
            obs_tree: the tree indicating the nodes used for observation
    """
    children = tuple(exact_node_value_after_observe(c) + c[0]
                     for c in obs_tree[1])
    return cmax(children, default=ZERO)
