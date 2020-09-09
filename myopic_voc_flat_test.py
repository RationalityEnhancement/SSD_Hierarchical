import os
from utils.mouselab_flat import MouselabEnv
from utils.distributions import Normal, Categorical
import random
import math
import time
import pandas as pd
from itertools import compress
import numpy as np
import argparse

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('no_goals', type=str)
    args = parser.parse_args()

    NO_OPTION = int(args.no_goals)
    cwd = os.getcwd()
    cwd += '/' + str(NO_OPTION) + '_' + str(NO_OPTION * 18)

    TREE_1 = np.load(cwd + '/tree.npy')
    DIST = np.load(cwd + '/dist.npy')
    TREE = []
    for t in TREE_1:
        TREE.append(t)
    OPTION_SET = np.load(cwd + '/option_set.npy')
    BRANCH_COST = 1
    SWITCH_COST = 1
    SEED = 0
    TAU = 20
    NO_BINS = 4

    NO_OPTION = 2
    BRANCH_COST = 1
    SWITCH_COST = 1
    SEED = 0
    TAU = 20
    
    node_types = []
    for tpe in DIST:
        node_types.append(tpe)

    def reward(i):
        global node_types
        sigma_val = {'V1': 5, 'V2': 10, 'V3': 20, 'V4': 40, 'G1': 100, 'G2': 120, 'G3': 140, 'G4': 160, 'G5': 180}
        return Normal(mu=0, sigma=sigma_val[node_types[i]])

    
    try:
        os.makedirs(cwd + '/Myopic_Results')
    except:
        pass

    def myopic_voc(x):
            if x == env.term_action:
                return 0
            state_disc = env.discretize(env._state, NO_BINS)
            return env.myopic_voc(x, state_disc) + env.cost # env.cost already takes sign into account

    num_episodes = 100
    # env._state = env.discretize(env._state, 4)
    # for i,s in enumerate(env._state):
    #     print(i,s)
    cum_reward = 0
    tic = time.time()
    df = pd.DataFrame(columns=['i', 'return','actions','Actual Path','Time', 'ground_truth'])
    for i in range(num_episodes):
        print(i)
        env = MouselabEnv.new(NO_OPTION, TREE, reward=reward, branch_cost=BRANCH_COST, switch_cost=SWITCH_COST, tau=TAU,
                      seed=SEED+i)
        env_tic = time.time()
        exp_reward = 0
        actions = []
        while True:
    #         print("Env State: {}".format(env._state))
            action_possible = list(env.actions(env._state))
            action = max(action_possible, key = myopic_voc)
            actions.append(action)
    #         print("Action Taken: {}".format(action))
            _, rew, done, _=env._step_actual(action)
            exp_reward += rew
            if done:
                break

        env_toc = time.time()
        df.loc[i] = [i, exp_reward, actions, env.actual_path(env._state), env_toc-env_tic, env.ground_truth]
        cum_reward += exp_reward
        df.to_csv(cwd + '/Myopic_Results/random_'+ str(NO_BINS)+ '.csv')
        np.save(cwd + '/Myopic_Results/CumResult_' + str(NO_BINS), cum_reward / num_episodes)
    print(cum_reward / num_episodes)
    toc = time.time()
    np.save(cwd + '/Myopic_Results/Eval_Time_' + str(NO_BINS), toc - tic)