import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import scipy.misc
import os
import csv
import itertools
import tensorflow.contrib.slim as slim
from tensorflow.contrib.layers.python.layers import initializers
from tensorflow.python.ops import init_ops

from utils.mouselab_hierarichal_simple_VAR import MouselabEnv
from utils.distributions import Normal, Categorical
import random
import math
import time
import pandas as pd
from itertools import compress
import argparse
import GPyOpt
import GPy
import numpy as np

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('no_goals', type=str)
    parser.add_argument('train', type=str)
    args = parser.parse_args()

    NO_OPTION = int(args.no_goals)
    TRAIN_FLAG = int(args.train)
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

    node_types = []
    for tpe in DIST:
        node_types.append(tpe)


    def blackboxfunc_low(W):
        global node_types
        num_episodes = 100
        
        w1 = W[:,0]
        w2 = W[:,1]
        w4 = W[:,2]



        def voc_estimate_low(x):
            #features[0] is cost(action)
            #features[1] is myopicVOC(action) [aka VOI]
            #features[2] is vpi_action [aka VPIsub; the value of perfect info of branch]
            #features[3] is vpi(beliefstate)
            #features[4] is expected term reward of current state
            features = env.low_action_features(x,env.selected_option, bins=NO_BINS)
            w3 = 1 - w1 - w2
            # print("Cost = {}, Myopic VOC = {}, Action = {}, VPI = {}, Term Reward = {}".format(w1*features[0][0] + w3*features[0][1] + w2*features[0][2], features[1], features[2], features[3], features[4]))
            

            # print("Net Value = {}".format(w1*features[1] + w2*features[3] + w3*features[2] + w4*features[0]))
            return w1*features[1] + w2*features[3] + w3*features[2] + w4*(w1*features[0][0] + w3*features[0][1] + w2*features[0][2])  
        
        cumreturn = 0
            
        for i in range(num_episodes):        
            choice = np.random.choice(NO_OPTION)
            # print("Choice = {}".format(choice))
            DIST = DISTS[choice]
            node_types = []
            for tpe in DIST:
                node_types.append(tpe)
            env = MouselabEnv.new(NO_OPTION, TREE, reward=reward, option_set=OPTION_SET, branch_cost=BRANCH_COST, switch_cost=SWITCH_COST, tau=TAU,
                              seed=SEED + i)
            #  print("BO Ground Truth = {}\n".format(env.ground_truth)) 
            # High Action
            possible_high_level_actions = list(range(len(env.init) + env.no_options, len(env.init)+ 2 * env.no_options + 1))

            env.high_step(env.high_term_action)
            exp_return = 0
            option_selected_index = env.selected_option - 1
            # print("Selected Option = {}".format(env.selected_option))
            possible = env.option_set[option_selected_index]
            # Low Policy
            while True:
                #take action that maximises estimated VOC
                possible_actions = [x for x in possible if hasattr(env.low_state[x], 'sample')]
                possible_actions = possible_actions + [env.low_term_actions[option_selected_index]]
                # print("Possible Actions = {}".format(possible_actions))
                action_taken = max(possible_actions, key = voc_estimate_low)
                # print("Low Action taken: {}".format(action_taken))
                _, rew, done_low, _= env.low_step(action_taken)
                exp_return += rew
                # print("Net Reward: {} Done: {}".format(exp_return,done_low))
                if done_low:
                    break
            cumreturn += exp_return
        
        # print(cumreturn/num_episodes)
        return -cumreturn/num_episodes

    def blackboxfunc_low_test(W):
        global node_types
        num_episodes = 100
        
        w1 = W[:,0]
        w2 = W[:,1]
        w4 = W[:,2]
        def voc_estimate_low(x):
            #features[0] is cost(action)
            #features[1] is myopicVOC(action) [aka VOI]
            #features[2] is vpi_action [aka VPIsub; the value of perfect info of branch]
            #features[3] is vpi(beliefstate)
            #features[4] is expected term reward of current state
            features = env.low_action_features(x, env.selected_option, bins=NO_BINS)
            w3 = 1 - w1 - w2
            # print("Weights = {} {} {}".format(w1, w2, w3))
            # print("{}: Myopic VOC = {}, VPI = {}, VPI Action = {}, Cost = {}".format(x, features[1], features[3], features[2], w1*features[0][0] + w3*features[0][1] + w2*features[0][2]))
            # print("Net Value = {}".format( w1*features[1] + w2*features[3] + w3*features[2] + w4*(w1*features[0][0] + w3*features[0][1] + w2*features[0][2])))
            
            return w1*features[1] + w2*features[3] + w3*features[2] + w4*(w1*features[0][0] + w3*features[0][1] + w2*features[0][2])  
        
        cumreturn = 0
        df = pd.DataFrame(columns=['i', 'env_type', 'return','actions', 'Actual Path','ground_truth'])
        for i in range(num_episodes):
            # print("i = {}".format(i))

            choice = np.random.choice(NO_OPTION)
            DIST = DISTS[choice]
            node_types = []
            for tpe in DIST:
                node_types.append(tpe)
            env = MouselabEnv.new(NO_OPTION, TREE, reward=reward, option_set=OPTION_SET, branch_cost=BRANCH_COST, switch_cost=SWITCH_COST, tau=TAU,
                              seed=SEED + i)
            # print("BO Ground Truth = {}\n".format(env.ground_truth)) 
            # High Action
            possible_high_level_actions = list(range(len(env.init) + env.no_options, len(env.init)+ 2 * env.no_options + 1))

            env.high_step(env.high_term_action)
            exp_return = 0
            option_selected_index = env.selected_option - 1
            # print("Selected Option = {}".format(env.selected_option))
            possible = env.option_set[option_selected_index]
            # Low Policy
            actions = []
            while True:
                #take action that maximises estimated VOC


                possible_actions = [x for x in possible if hasattr(env.low_state[x], 'sample')]
                possible_actions = possible_actions + [env.low_term_actions[option_selected_index]]
                # print("Possible Actions = {}".format(possible_actions))
                # print("State = {}".format(env.low_state))
                action_taken = max(possible_actions, key = voc_estimate_low)
                actions.append(action_taken)
                # print("Low Action taken: {}".format(action_taken))
                # print("State Before Action: {}".format(env.low_state))
                _, rew, done_low, _= env.low_step_actual(action_taken)
                # print("State After Action: {}".format(env.low_state))
                exp_return += rew
                # print("Net Reward: {} Done: {}".format(exp_return,done_low))
                if done_low:
                    break
            df.loc[i] = [i, choice, exp_return, actions, env.option_actual_path(option_selected_index+1, env.low_state), env.ground_truth]
            cumreturn += exp_return
            
        df.to_csv(cwd + '/Hierarchical_Results/low_'+ str(NO_BINS)+ '.csv')
        # print(cumreturn/num_episodes)
        np.save(cwd + '/Hierarchical_Results/Low_CumResult_' + str(NO_BINS), cumreturn / num_episodes)
        return -cumreturn/num_episodes

    def blackboxfunc(W): 
        global node_types1
        num_episodes = 100
        
        w5 = W[:,0]
        w6 = 1 - w5
        w7 = W[:,1]
        
        w1 = W_low[:,0]
        w2 = W_low[:,1]
        w4 = W_low[:,2]
        w3 = 1 - w1 - w2
        
        def voc_estimate_low(x):
            #features[0] is cost(action)
            #features[1] is myopicVOC(action) [aka VOI]
            #features[2] is vpi_action [aka VPIsub; the value of perfect info of branch]
            #features[3] is vpi(beliefstate)
            #features[4] is expected term reward of current state
            features = env.low_action_features(x, env.selected_option, bins=NO_BINS)
            # w3 = 1 - w1 - w2
            # print("Weights = {} {} {}".format(w1, w2, w3))
            # print("{}: Myopic VOC = {}, VPI = {}, VPI Action = {}, Cost = {}".format(x, features[1], features[3], features[2], w1*features[0][0] + w3*features[0][1] + w2*features[0][2]))
            # print("Net Value = {}".format( w1*features[1] + w2*features[3] + w3*features[2] + w4*(w1*features[0][0] + w3*features[0][1] + w2*features[0][2])))
            
            return w1*features[1] + w2*features[3] + w3*features[2] + w4*(w1*features[0][0] + w3*features[0][1] + w2*features[0][2])  
        
        
        def voc_estimate_high(x):
            #features[0] is cost for action
            #features[1] is myopicVOC(action)
            #features[2] is vpi(beliefstate)
            #features[3] is expected term reward of current state
            
            features = env.high_action_features(x)
            return w5*features[1] + w6*features[2] + w7*features[0]
        
        cumreturn = 0
        
        for i in range(num_episodes):

            env = MouselabEnv.new(NO_OPTION, TREE, reward=reward_complete, option_set=OPTION_SET, branch_cost=BRANCH_COST, switch_cost=SWITCH_COST, tau=TAU,
                              seed=SEED + i)

            # print("i = {}".format(i))
            # print("BO Ground Truth = {}\n".format(env.ground_truth)) 
            # High Action
            possible_high_level_actions = list(range(len(env.init) + env.no_options, len(env.init)+ 2 * env.no_options + 1))
            exp_return = 0
            while True:
                #take action that maximises estimated VOC
                high_action_taken = max(possible_high_level_actions, key = voc_estimate_high)
                # print("High Action Taken = {}".format(high_action_taken))
                _, rew_high, done_high, _= env.high_step(high_action_taken)
                if done_high:
                    option_selected_index = env.selected_option - 1
                    possible = env.option_set[option_selected_index]
                    # Low Policy
                    while True:
                        #take action that maximises estimated VOC
                        possible_actions = [x for x in possible if hasattr(env.low_state[x], 'sample')]
                        possible_actions = possible_actions + [env.low_term_actions[option_selected_index]]
                        # print("Possible Actions = {}".format(possible_actions))
                        action_taken = max(possible_actions, key = voc_estimate_low)
                        # print("Low Action taken: {}".format(action_taken))
                        _, rew, done_low, _= env.low_step(action_taken)
                        if not done_low:
                            exp_return += rew
                        elif done_low:
                            exp_return += env.high_term_reward()
                            break
                        # print("Net Reward: {} Done: {}".format(exp_return,done_low))
                    break
                else:
                    exp_return += rew_high
                    possible_high_level_actions.remove(high_action_taken)
            # print("Reward for the episode = {}".format(exp_return))
            cumreturn += exp_return
        # print(cumreturn/num_episodes)
        return -cumreturn/num_episodes

    def blackboxfunc_test(W):
        global node_types1
        num_episodes = 100
        
        w5 = W[:,0]
        w6 = 1 - w5
        w7 = W[:,1]
        
        w1 = W_low[:,0]
        w2 = W_low[:,1]
        w4 = W_low[:,2]
        w3 = 1 - w1 - w2
        def voc_estimate_low(x):
            #features[0] is cost(action)
            #features[1] is myopicVOC(action) [aka VOI]
            #features[2] is vpi_action [aka VPIsub; the value of perfect info of branch]
            #features[3] is vpi(beliefstate)
            #features[4] is expected term reward of current state
            features = env.low_action_features(x, env.selected_option, bins=NO_BINS)
            w3 = 1 - w1 - w2
            # print("Weights = {} {} {}".format(w1, w2, w3))
            # print("{}: Myopic VOC = {}, VPI = {}, VPI Action = {}, Cost = {}".format(x, features[1], features[3], features[2], w1*features[0][0] + w3*features[0][1] + w2*features[0][2]))
            # print("Net Value = {}".format( w1*features[1] + w2*features[3] + w3*features[2] + w4*(w1*features[0][0] + w3*features[0][1] + w2*features[0][2])))
            
            return w1*features[1] + w2*features[3] + w3*features[2] + w4*(w1*features[0][0] + w3*features[0][1] + w2*features[0][2])  
        
        
        def voc_estimate_high(x):
            #features[0] is cost for action
            #features[1] is myopicVOC(action)
            #features[2] is vpi(beliefstate)
            #features[3] is expected term reward of current state
            features = env.high_action_features(x)
            return w5*features[1] + w6*features[2] + w7*features[0]
        
        
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
        
        df.to_csv(cwd + '/Hierarchical_Results/high_'+ str(NO_BINS)+ '.csv')
        np.save(cwd + '/Hierarchical_Results/CumResult_' + str(NO_BINS), cumreturn / num_episodes)
        np.save(cwd + '/Hierarchical_Results/RewardPerClick_' + str(NO_BINS), reward_per_click / num_episodes)
        # print("Cumulative Reward".format(cumreturn/num_episodes))
        return -cumreturn/num_episodes


    if(TRAIN_FLAG == 0):  # Testing 
        W_high = np.load(cwd + '/Hierarchical_Results/High_Level_weights_' + str(NO_BINS)+ '.npy')
        W_low = np.load(cwd + '/Hierarchical_Results/Low_Level_weights_' + str(NO_BINS)+ '.npy')

        eval_tic = time.time()
        blackboxfunc_test(W_high)
        toc = time.time()
        np.save(cwd + '/Hierarchical_Results/Eval_Time_' + str(NO_BINS), toc - eval_tic)
    else:  # Training
        space = [{'name': 'w1', 'type': 'continuous', 'domain': (0,1)},
         {'name': 'w2', 'type': 'continuous', 'domain': (0,1)},
         {'name': 'w4', 'type': 'continuous', 'domain': (1,18)}]

         constraints = [{'name': 'part_1', 'constraint': 'x[:,0] + x[:,1] - 1'}]

         feasible_region = GPyOpt.Design_space(space = space, constraints = constraints)

         from numpy.random import seed # fixed seed
        seed(123456)

        initial_design = GPyOpt.experiment_design.initial_design('random', feasible_region, 10)
        # --- CHOOSE the objective
        objective = GPyOpt.core.task.SingleObjective(blackboxfunc_low)

        # --- CHOOSE the model type
        #This model does Maximum likelihood estimation of the hyper-parameters.
        model = GPyOpt.models.GPModel(exact_feval=True,optimize_restarts=10,verbose=False)

        # --- CHOOSE the acquisition optimizer
        aquisition_optimizer = GPyOpt.optimization.AcquisitionOptimizer(feasible_region)

        # --- CHOOSE the type of acquisition
        acquisition = GPyOpt.acquisitions.AcquisitionEI(model, feasible_region, optimizer=aquisition_optimizer)

        # --- CHOOSE a collection method
        evaluator = GPyOpt.core.evaluators.Sequential(acquisition)

        bo = GPyOpt.methods.ModularBayesianOptimization(model, feasible_region, objective, acquisition, evaluator, initial_design)

        # --- Stop conditions
        max_time  = None
        tolerance = 1e-6     # distance between two consecutive observations   
        
        try:
            os.makedirs(cwd + '/Hierarchical_Results')
        except:
            pass       

        # Run the optimization
        max_iter  = 100
        time_start = time.time()
        train_1_tic = time_start
        bo.run_optimization(max_iter = max_iter, max_time = max_time, eps = tolerance, verbosity=True)

        W_low = np.array([bo.x_opt])
        np.save(cwd + '/Hierarchical_Results/Low_Level_weights_' + str(NO_BINS), W_low)

        train_1_toc = time.time()

        space = [{'name': 'w5', 'type': 'continuous', 'domain': (0,1)},
         {'name': 'w7', 'type': 'continuous', 'domain': (1,NO_OPTION)}]

         constraints = []

         feasible_region = GPyOpt.Design_space(space = space, constraints = constraints)
        # --- CHOOSE the intial design
        seed(123456)

        initial_design = GPyOpt.experiment_design.initial_design('random', feasible_region, 10)
        # --- CHOOSE the objective
        objective = GPyOpt.core.task.SingleObjective(blackboxfunc)

        # --- CHOOSE the model type
        #This model does Maximum likelihood estimation of the hyper-parameters.
        model = GPyOpt.models.GPModel(exact_feval=True,optimize_restarts=10,verbose=False)

        # --- CHOOSE the acquisition optimizer
        aquisition_optimizer = GPyOpt.optimization.AcquisitionOptimizer(feasible_region)

        # --- CHOOSE the type of acquisition
        acquisition = GPyOpt.acquisitions.AcquisitionEI(model, feasible_region, optimizer=aquisition_optimizer)

        # --- CHOOSE a collection method
        evaluator = GPyOpt.core.evaluators.Sequential(acquisition)

        bo = GPyOpt.methods.ModularBayesianOptimization(model, feasible_region, objective, acquisition, evaluator, initial_design)

        # --- Stop conditions
        max_time  = None
        tolerance = 1e-6     # distance between two consecutive observations  

        # Run the optimization  
        max_iter  = 100
        time_start = time.time()
        train_2_tic = time_start
        bo.run_optimization(max_iter = max_iter, max_time = max_time, eps = tolerance, verbosity=True)

        W_high = np.array([bo.x_opt])
        np.save(cwd + '/Hierarchical_Results/High_Level_weights_' + str(NO_BINS), W_high)

        train_2_toc = time.time()

        np.save(cwd + '/Hierarchical_Results/Option_Train_Time_' + str(NO_BINS), train_1_toc - train_1_tic)
        np.save(cwd + '/Hierarchical_Results/High_Train_Time_' + str(NO_BINS), train_2_toc - train_2_tic)
        np.save(cwd + '/Hierarchical_Results/Total_Train_Time_' + str(NO_BINS), train_1_toc - train_1_tic + train_2_toc - train_2_tic)







