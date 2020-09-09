import os
from utils.mouselab_hierarichal_simple_VAR import MouselabEnv
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
    DISTS = np.load(cwd + '/dists.npy')
    DIST1 = np.load(cwd + '/dist.npy')

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

    node_types1 = []
    for tpe in DIST1:
        node_types1.append(tpe)

    def reward_complete(i):
        global node_types1
        sigma_val = {'V1': 5, 'V2': 10, 'V3': 20, 'V4': 40, 'G1': 100, 'G2': 120, 'G3': 140, 'G4': 160, 'G5': 180}
        return Normal(mu=0, sigma=sigma_val[node_types1[i]])


    def blackboxfunc_test():
        global node_types1
        num_episodes = 100
        
        def voc_estimate_low(x):
            #features[0] is cost(action)
            #features[1] is myopicVOC(action) [aka VOI]
            #features[2] is vpi_action [aka VPIsub; the value of perfect info of branch]
            #features[3] is vpi(beliefstate)
            #features[4] is expected term reward of current state
            if x == env.low_term_actions[env.selected_option-1]:
                return 0
            state_disc = env.discretize(env.low_state, NO_BINS)
            return env.low_myopic_voc(x, env.selected_option, state_disc) + env.cost # env.cost already takes sign into account

        
        def voc_estimate_high(x):
            #features[0] is estimated reward gain for action
            #features[1] is cost for action
            #features[2] is estimated number of clicks
            if x == env.high_term_action:
                return 0
            return env.high_myopic_voc(env.high_state, x) + env.switch_cost
        
        
        cumreturn = 0
        reward_per_click = 0
        df = pd.DataFrame(columns=['i', 'return','high_actions', 'low_actions','Actual Path','Time','ground_truth'])

        for i in range(num_episodes):
            ep_tic = time.time()
            env = MouselabEnv.new(NO_OPTION, TREE, reward=reward_complete, option_set=OPTION_SET, branch_cost=BRANCH_COST, switch_cost=SWITCH_COST, tau=TAU,
                              seed=1000*SEED + i)
            # print("BO Ground Truth = {}\n".format(env.ground_truth)) 

            # High Action
            possible_high_level_actions = list(range(len(env.init) + env.no_options, len(env.init)+ 2 * env.no_options + 1))
            exp_return = 0
            high_actions = []
            while True:
                #take action that maximises estimated VOC
                high_action_taken = max(possible_high_level_actions, key = voc_estimate_high)
                # print("High Action Taken = {}".format(high_action_taken))
                _, rew, done_high, _= env.high_step(high_action_taken)
                high_actions.append(high_action_taken)
                if done_high:
                # print("High State: {}".format(env.high_state))
                    option_selected_index = env.selected_option - 1
                    possible = env.option_set[option_selected_index]
                    # Low Policy
                    low_actions = []
                    while True:
                        #take action that maximises estimated VOC
                        possible_actions = [x for x in possible if hasattr(env.low_state[x], 'sample')]
                        possible_actions = possible_actions + [env.low_term_actions[option_selected_index]]
                        # print("Possible Actions = {}".format(possible_actions))
                        action_taken = max(possible_actions, key = voc_estimate_low)
                        low_actions.append(action_taken)
                        # print("Low Action taken: {}".format(action_taken))
                        _, rew, done_low, _= env.low_step(action_taken)
                        if done_low:
                            # exp_return += env.high_term_reward()
                            break
                        # print("Net Reward: {} Done: {}".format(exp_return,done_low))
                    break
                else:
                    # exp_return += rew
                    possible_high_level_actions.remove(high_action_taken)
        
            exp_return = sum(env.ground_truth[env.actual_path(env.low_state)] + env.cost* (len(low_actions) - 1) + env.switch_cost*(len(high_actions) -1))
            cumreturn += exp_return
            clicks = (len(high_actions) - 1) + (len(low_actions) - 1)
            reward_per_click += (exp_return / clicks)
            # print(len([i, exp_return, high_actions, low_actions, env.ground_truth]))
            ep_toc = time.time()
            df.loc[i] = [i, exp_return, high_actions, low_actions, env.actual_path(env.low_state), ep_toc - ep_tic, env.ground_truth]
        
        df.to_csv(cwd + '/Hierarchical_Myopic_Results/high_'+ str(NO_BINS)+ '.csv')
        np.save(cwd + '/Hierarchical_Myopic_Results/CumResult_' + str(NO_BINS), cumreturn / num_episodes)
        np.save(cwd + '/Hierarchical_Myopic_Results/RewardPerClick_' + str(NO_BINS), reward_per_click / num_episodes)
        # print("Cumulative Reward".format(cumreturn/num_episodes))
        return -cumreturn/num_episodes

       try:
        os.makedirs(cwd + '/Hierarchical_Myopic_Results')
    except:
        pass

    eval_tic = time.time()
    blackboxfunc_test()
    toc = time.time()
    np.save(cwd + '/Hierarchical_Myopic_Results/Eval_Time_' + str(NO_BINS), toc - eval_tic)
