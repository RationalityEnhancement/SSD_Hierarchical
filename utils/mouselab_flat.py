from collections import namedtuple, defaultdict, deque, Counter
import numpy as np
import gym
from gym import spaces
from utils.distributions import smax, cmax, cross_1, sample, expectation, Normal, PointMass, Categorical
from toolz import memoize, get
import random
from contracts import contract
from functools import reduce
import math
import itertools as it
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

    def __init__(self, tree, init, no_options, goals, option_set, env_type, ground_truth=None, branch_cost=0,
                 switch_cost=0, tau=20, sample_term_reward=False, repeat_cost=1):

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
        self.term_action = len(init)
        self.n_obs = len(self.init) - 1
        # Required for gym.Env API.
        self.action_space = spaces.Discrete(len(self.init) + 1)
        self.n_actions = self.action_space.n - 1
        self.initial_states = None  # TODO
        self.exact = True  # TODO
        self.tree = tree
        self.subtree = self._get_subtree()
        self.tau = tau
        # May remove later. Check if req by any function in non hierarchical manner
        self.no_options = no_options
        self.goals = goals
        self.option_set = option_set

        self.env_type = env_type

        self._reset()

    def _get_subtree(self):
        def gen(n):
            yield n
            for n1 in self.tree[n]:
                yield from gen(n1)
        return [tuple(gen(n)) for n in range(len(self.tree))]

    def _reset(self):
        if self.initial_states:
            self.init = random.choice(self.initial_states)
        self.paths = self._paths()
        self._state = self.init

        obs_list = []
        for i in range(len(self.init)):
            obs_list.append([])
        self.obs_list = obs_list
        self.tree_no_goals = self.no_goal_tree()
        return self._state

    def no_goal_tree(self):
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
        def rec(path):
            children = self.tree[path[-1]]
            if children:
                for child in children:
                    yield from rec(path + [child])
            else:
                yield path

        return rec([start])

    def _step(self, action):
        if self._state is self.term_state:
            assert 0, 'state is terminal'
        if action == self.term_action:
            reward, _ = self.term_reward(self._state)
            done = True
            obs = False
        else:  # observe a node
            self._state = self._observe(action)
            reward = self.cost
            done = False
            obs = False
        return self._state, reward, done, obs

    def _step_actual(self, action):
        if self._state is self.term_state:
            assert 0, 'state is terminal'
        if action == self.term_action:
            reward = self.term_reward_actual(self._state)
            done = True
            obs = False
        else:  # observe a node
            self._state = self._observe(action)
            reward = self.cost
            done = False
            obs = False
        return self._state, reward, done, obs

    def _observe(self, action):
        if self.ground_truth is not None:
            result = self.ground_truth[action]
        else:
            result = self._state[action].sample()
        s = list(self._state)
        s[action] = result
        return tuple(s)

    def term_reward(self, state):
        rewards = []
        vars_list = []
        for path in self.allpaths:
            reward = 0
            var = 0
            for x in path:
                if hasattr(state[x], 'sample'):
                    reward += state[x].expectation()
                    var += state[x].sigma ** 2  # Assuming Normal
                else:
                    reward += state[x]
            rewards.append(reward)
            vars_list.append(var)
        max_reward = max(rewards)
        m = max(rewards)
        index = [i for i, j in enumerate(rewards) if j == m]
        min_var = min([vars_list[i] for i in index])
        return max_reward, min_var

    def term_reward_actual(self, state):
        actual_rewards = []
        believed_rewards = []
        for path in self.allpaths:
            believed_reward = 0
            for p in path:
                if hasattr(state[p], 'sample'):  # From distribution
                    believed_reward += state[p].expectation()
                else:
                    believed_reward += state[p]
            believed_rewards.append(believed_reward)
            actual_rewards.append(sum([self.ground_truth[p] for p in path]))
        index = np.argmax(believed_rewards)
        # print("Believed Max Reward = {}".format(believed_rewards[index]))
        return actual_rewards[index]

    def path_to(self, node):
        return self.paths[node]

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

    @classmethod
    def new(cls, no_options, tree, reward, seed=None, env_type="manual", option_set = None, initval=None, **kwargs):
        """Returns a MouselabEnv with a symmetric structure.

        Arguments:
            no_options: Number of options in the environment
            tree: Tree structure of environment
            reward: a function that returns the reward distribution at a given depth.
            seed: Seed
            env_type: Default value is manual
            option_set: Option Set
            initval: Initialization of nodes

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
        # Get number of goals
        goals = []
        c = 0
        for _, n in enumerate(tree):
            if n != []:
                prev = n
            if n == []:
                if len(prev) != 1:
                    goals.append([prev[c]])
                    c += 1
                else:
                    goals.append(prev)
                    c = 0
        # print("Goals = {}".format(goals))

        def option_add(opt, opt_temp):
            option = []
            for _, node in enumerate(opt_temp):
                if type(node) == int:
                    option.append(node)
                else:
                    option.extend(node)
            woduplicates = list(set(option))
            opt.append(woduplicates)
            return opt

        if option_set is None:
            g_c = 0
            opt_no = 0
            c = 1
            goal = goals[g_c]
            opt_temp = []
            opt = []
            for j, t in enumerate(tree[0]):
                # For a particular goal
                lst = tree[c:]
                try:
                    max_index = len(lst) - lst[::-1].index(goal) - 1
                except:
                    max_index = -1  # No more index left
                if max_index == 0:  # Final nodes belong to last option
                    opt_temp.append(lst[0])
                    c += 1
                    lst = tree[c:]
                if max_index == -1 or max_index == 0:  # Update the option list
                    g_c += 1
                    if opt_temp != []:
                        opt = option_add(opt, opt_temp)
                        opt_no += 1
                    opt_temp = []
                    if g_c >= no_options:
                        g_c = no_options - 1
                    goal = goals[g_c]

                opt_temp.append(t)

                for i, tt in enumerate(lst):
                    opt_temp.append(tt)
                    if tt == goal:
                        c += i + 1
                        break
                if opt_no == no_options - 1:  # Nodes of last goal only
                    if j != (len(tree[0]) - 1):
                        add_nodes = []
                        for _, rt in enumerate(tree[0][j:]):
                            add_nodes.append(rt)
                    else:
                        add_nodes = [t]
                    lst = add_nodes + lst
                    opt = option_add(opt, lst)
                    break
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
            r = self._state[x]
            observed = not hasattr(self._state[x], 'sample')
            c = color(r) if observed else 'grey'
            node_val = str(round(r, 2)) if observed else str(x)
            dot.node(str(x), label=node_val, style='filled', color=c)
            for y in ys:
                dot.edge(str(x), str(y))
        return dot

    def discretize(self, state, bins=4):
        state_disc = []
        for s in state:
            if hasattr(s, 'sample') and hasattr(s, 'mu'):
                # print("Normal detected")
                dist = s.to_discrete(n=bins, max_sigma=4)
                dist.vals = tuple([(round(val, 3)) for val in dist.vals])
                dist.probs = tuple([(round(p, 3)) for p in dist.probs])
                state_disc.append(dist)
            else:
                state_disc.append(s)
        return tuple(state_disc)

    def actions(self, state):
        if state is self.term_state:
            return
        for i, v in enumerate(state):
            if hasattr(v, 'sample'):
                yield i
        yield self.term_action



    def results(self, state, action, bins=4):
        """Returns a list of possible results of taking action in state.

        Each outcome is (probability, next_state, reward).
        """
        # print("Action = {}".format(action))
        state_des = self.discretize(state, bins=bins)
        if action == self.term_action:
            yield (1, self.term_state, self.expected_term_reward_disc(state_des))
        else:
            for r, p in state_des[action]:
                s1 = list(state_des)
                s1[action] = r
                yield (p, tuple(s1), self.cost)

    @lru_cache(CACHE_SIZE)
    def expected_term_reward(self, state):
        val, _ = self.term_reward(state)
        if abs(val) < 0.001:
            val = 0.0
        return val

    @lru_cache(CACHE_SIZE)
    def expected_term_reward_disc(self, state):
        # print(self.term_reward(option, state))
        return self.term_reward_disc(state).expectation()

    def term_reward_disc(self, state):
        """A distribution over the return gained by acting given a belief state."""
        return self.node_value(0, state)

    def node_value(self, node, state):
        """A distribution over total rewards after the given node."""
        return max((self.node_value(n1, state) + state[n1]
                    for n1 in self.tree[node]),
                   default=ZERO, key=expectation)

        # @lru_cache(CACHE_SIZE)

    def action_features(self, action, bins=4, state=None):
        state = state if state is not None else self._state
        state_disc = self.discretize(state, bins)

        assert state is not None

        if action == self.term_action:
            return np.array([
                0,
                0,
                0,
                0,
                self.expected_term_reward(state)
            ])

        return np.array([
            self.cost,
            self.myopic_voc(action, state_disc),
            self.vpi_action(action, state_disc),
            self.vpi(state_disc),
            self.expected_term_reward(state)
            ])

    def shrink(self, option_dist):
        if len(option_dist) == 2:
            return option_dist
        else:
            #         print("Continuing")
            two_dist = [option_dist[0], option_dist[1]]
            #         print(two_dist)
            new_dist = [cmax(two_dist, default=ZERO)] + option_dist[2:]
            return self.shrink(new_dist)

    @contract
    def myopic_voc(self, action, state) -> 'float, >= -0.001':
        """
        Calculates myopic voc


        Using literal definition of myopic voc. Take the expectation of best possible action after observing the node
        """
        # print("Action = {}, state = {}".format(action, state))
        if hasattr(state[action], 'sample'):
            possible = list(state[action])
            r, p = zip(*possible)
            expected_reward = 0
            for k in range(len(p)):
                state2 = list(state)
                state2[action] = r[k]
                expected_reward += p[k] * self.expected_term_reward_disc(tuple(state2))
            result = float(expected_reward - self.expected_term_reward_disc(state))
            if abs(result) < 0.001:
                result = 0.0
            return result
        else:
            return 0.0

    @contract
    def vpi_action(self, action, state) -> 'float, >= -0.001':
        """
        Calculates vpi action. Nodes of importance are those who are either parents or children of the node selected
        """

        # print("Ground Truth = {}".format(self.ground_truth))
        # print("State = {}".format(state))
        # print("Action = {}".format(action))
        option_dist = []
        obs = (*self.subtree[action][0:], *self.path_to(action)[1:])
        obs = list(set(obs))
        for option in range(1, self.no_options + 1):
            op_dist = self.node_value_after_observe_option(option, state, obs)
            node_idx = self.goals[option - 1][0]
            if not hasattr(state[node_idx], 'sample'):
                goal_dist = Categorical(vals=[state[node_idx]], probs=[1])
            else:
                goal_dist = state[node_idx]
            dists = [op_dist, goal_dist]
            option_dist.append(cross_1(dists, sum))

        net_dist = self.shrink(option_dist)
        nvao = float(cmax(net_dist, default=ZERO).expectation())

        # print(obs)
        # print("Env.state = {}".format(state))
        # for _,i in enumerate(state):
        #     print(i)
        # print("Expected Term Reward = {}".format(self.expected_term_reward(state)))
        # print("Observe Node Expected = {}".format(self.node_value_after_observe(obs, 0, state,verbose).expectation()))
        result = nvao - self.expected_term_reward_disc(state)
        if abs(result) < 0.001:
            result = 0.0

        return result

    @lru_cache(CACHE_SIZE)
    @contract
    def vpi(self, state) -> 'float, >= -0.001':
        """
        Calculates vpi. All nodes of branch are important. Basically calculating vpi_action with goal node selected
        """
        option_dist = []
        for option in range(1, self.no_options+1):
            action = self.goals[option - 1][0]
            obs = (*self.subtree[action][0:], *self.path_to(action)[1:])
            obs = list(set(obs))
            op_dist = self.node_value_after_observe_option(option, state, obs)
            node_idx = self.goals[option-1][0]
            if not hasattr(state[node_idx], 'sample'):
                goal_dist = Categorical(vals=[state[node_idx]], probs= [1])
            else:
                goal_dist = state[node_idx]
            dists = [op_dist, goal_dist]

            option_dist.append(cross_1(dists, sum))

        net_dist = self.shrink(option_dist)
        nvao = float(cmax(net_dist, default=ZERO).expectation())
        # print("VPI Node observe value = {}".format(nvao))
        result = nvao - self.expected_term_reward_disc(state)
        if abs(result) < 0.001:
            result = 0.0
        return result

    def to_obs_tree(self, state, node, obs=(), sort=True):
        """
        Takes the number of nodes that can be observed and creates multi-nested tuple for each possible path
        Nodes which aren't observable are set to its expected value (normally 0)
        """

        # print("IN TO OBS TREE")
        maybe_sort = sorted if sort else lambda x: x

        def rec(n):
            # print("In rec({})".format(n))
            subjective_reward = state[n] if n in obs else expectation(state[n])
            # print("state[n] = {}".format(state[n]))
            # print("Subjective Reward = {}, n in obs: {} state[n] = {},
            # expectation(state[n]) = {}".format(subjective_reward, n in obs, state[n], expectation(state[n])))
            # print("n = {} self.tree[n] = {}".format(n, self.tree[n]))
            children = tuple(maybe_sort(rec(c) for c in self.tree[n]))
            # print("Children = {}".format(children))
            # print("Returning for n={} RETURNED = {}".format(n,(subjective_reward, children)))
            return subjective_reward, children
        return rec(node)

    def node_value_after_observe(self, obs, node, state, option):
        """A distribution over the expected value of node, after making an observation.

        obs can be a single node, a list of nodes, or 'all'
        """
        obs_tree = self.to_obs_tree_option(state, node, option, obs)
        # print("Option: {}, OBS: {}, OBS_TREE: {}".format(option,obs, obs_tree))
        if self.exact:
            return exact_node_value_after_observe(obs_tree)
        else:
            return node_value_after_observe(obs_tree)

    def to_obs_tree_option(self, state, node, option, obs=(), sort=True):
        """
        Takes the number of nodes that can be observed and creates multi-nested tuple for each possible path
        Nodes which aren't observable are set to its expected value (normally 0)
        """

        maybe_sort = sorted if sort else lambda x: x

        def rec(n):
            # print("In rec({})".format(n))
            subjective_reward = state[n] if n in obs else expectation(state[n])
            # print("state[n] = {}".format(state[staten]))
            # print("Subjective Reward = {}, n in obs: {} state[n] = {}, expectation(state[n]) = {}".format(subjective_reward, n in obs, state[n], expectation(state[n])))
            if n == 0:
                child_nodes = []
                for ch in self.tree_no_goals[n]:
                    if ch in self.option_set[option - 1]:
                        child_nodes.append(ch)
            else:
                child_nodes = self.tree_no_goals[n]
            children = tuple(maybe_sort(rec(c) for c in child_nodes))
            return subjective_reward, children

        return rec(node)

    def node_value_after_observe_option(self, option, state, obs):
        """
        Calculates vpi. All nodes of option branch are important. Basically calculating vpi_action with goal node selected
        """
        return self.node_value_after_observe(obs, 0, state, option)

    @lru_cache(CACHE_SIZE)
    def expected_option_term_reward_disc(self, option, state):
        # print(self.term_reward(option, state))
        return self.option_term_reward(option, state).expectation()

    def option_term_reward(self, option, state):
        """A distribution over the return gained by acting given a belief state."""
        return self.option_node_value(0, option, state)

    def option_node_value(self, node, option, state):
        """A distribution over total rewards after the given node."""
        if node == 0:
            nodes = []
            for c in self.tree_no_goals[node]:
                if c in self.option_set[option-1]:
                    nodes.append(c)
        else:
            nodes = self.tree_no_goals[node]
        return max((self.option_node_value(n1, option, state) + state[n1]
                    for n1 in nodes),
                   default=ZERO, key=expectation)


@lru_cache(SMALL_CACHE_SIZE)
def node_value_after_observe(obs_tree):
    """A distribution over the expected value of node, after making an observation.

    `obs` can be a single node, a list of nodes, or 'all'
    """
    children = tuple(node_value_after_observe(c) + c[0] for c in obs_tree[1])
    return smax(children, default=ZERO)

@lru_cache(None)
def exact_node_value_after_observe(obs_tree):
    """A distribution over the expected value of node, after making an observation.

    `obs` can be a single node, a list of nodes, or 'all'
    """
    children = tuple(exact_node_value_after_observe(c) + c[0]
                     for c in obs_tree[1])
    return cmax(children, default=ZERO)
