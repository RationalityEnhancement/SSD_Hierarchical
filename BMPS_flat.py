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

from utils.mouselab_flat import MouselabEnv
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


    tic = time.time()


    def blackboxfunc(W):
        global node_types
        num_episodes = 100
        
        w1 = W[:,0]
        w2 = W[:,1]
        w4 = W[:,2]
        
        def voc_estimate(x):
            #features[0] is cost(action)
            #features[1] is myopicVOC(action) [aka VOI]
            #features[2] is vpi_action [aka VPIsub; the value of perfect info of branch]
            #features[3] is vpi(beliefstate)
            #features[4] is expected term reward of current state
            features = env.action_features(x, bins=NO_BINS)
            w3 = 1 - w1 - w2
            return w1*features[1] + w2*features[3] + w3*features[2] + w4*features[0]
        
        cumreturn = 0
        reward_per_click = 0
        
        for i in range(num_episodes):
            # print("i = {}".format(i))
            env = MouselabEnv.new(NO_OPTION, TREE, reward=reward, option_set=OPTION_SET, branch_cost=BRANCH_COST, switch_cost=SWITCH_COST, tau=TAU,
                              seed=SEED + i)

            exp_return = 0
            actions = []
            while True:
                possible_actions = list(env.actions(env._state))

                #take action that maximises estimated VOC
                action_taken = max(possible_actions, key = voc_estimate)
                actions.append(action_taken)
                _, rew, done, _=env._step(action_taken)
                exp_return+=rew
                if done:
                    break
            cumreturn += exp_return
        return -cumreturn/num_episodes

    def blackboxfunc_test(W):
        global node_types
        num_episodes = 100
        
        w1 = W[:,0]
        w2 = W[:,1]
        w4 = W[:,2]
        
        def voc_estimate(x):
            #features[0] is cost(action)
            #features[1] is myopicVOC(action) [aka VOI]
            #features[2] is vpi_action [aka VPIsub; the value of perfect info of branch]
            #features[3] is vpi(beliefstate)
            #features[4] is expected term reward of current state
            features = env.action_features(x, bins=NO_BINS)
            w3 = 1 - w1 - w2
            return w1*features[1] + w2*features[3] + w3*features[2] + w4*features[0]
        
        cumreturn = 0
        reward_per_click = 0
        df = pd.DataFrame(columns=['i', 'return','actions','Actual Path','Time', 'ground_truth'])
        for i in range(num_episodes):
            # print("i = {}".format(i))
            env = MouselabEnv.new(NO_OPTION, TREE, reward=reward, option_set=OPTION_SET, branch_cost=BRANCH_COST, switch_cost=SWITCH_COST, tau=TAU,
                              seed=1000*SEED + i)
            env_tic = time.time()
            exp_return = 0
            actions = []
            while True:
                possible_actions = list(env.actions(env._state))

                #take action that maximises estimated VOC
                action_taken = max(possible_actions, key = voc_estimate)
                actions.append(action_taken)
                _, rew, done, _=env._step_actual(action_taken)
                exp_return+=rew
                if done:
                    break
            env_toc = time.time()
            df.loc[i] = [i, exp_return, actions, env.actual_path(env._state), env_toc - env_tic, env.ground_truth]
            cumreturn += exp_return
            clicks = len(actions) - 1 
            reward_per_click += (exp_return / clicks)
            #print(exp_return)
        
        df.to_csv(cwd + '/Flat_Results/flat_'+ str(NO_BINS)+ '.csv')
        np.save(cwd + '/Flat_Results/CumResult_' + str(NO_BINS), cumreturn / num_episodes)
        np.save(cwd + '/Flat_Results/RewardPerClick_' + str(NO_BINS), reward_per_click / num_episodes)
     #     print("Cumulative Reward".format(cumreturn/num_episodes))
        return -cumreturn/num_episodes



    if(TRAIN_FLAG == 0):  # Testing 
        W = np.load(cwd + '/Flat_Results/Weights_' + str(NO_BINS)+ '.npy')
        eval_tic = time.time()
        blackboxfunc_test(W)
        toc = time.time()
        np.save(cwd + '/Flat_Results/Eval_Time_' + str(NO_BINS), toc - eval_tic)
    else:  # Training
        space = [{'name': 'w1', 'type': 'continuous', 'domain': (0,1)},
                 {'name': 'w2', 'type': 'continuous', 'domain': (0,1)},
                 {'name': 'w4', 'type': 'continuous', 'domain': (1, NO_OPTION * 18)}]

        constraints = [{'name': 'part_1', 'constraint': 'x[:,0] + x[:,1] - 1'}]

        feasible_region = GPyOpt.Design_space(space = space, constraints = constraints)

        # --- CHOOSE the intial design
        from numpy.random import seed # fixed seed
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

        try:
            os.makedirs(cwd + '/Flat_Results')
        except:
            pass

        # Run the optimization
        max_iter  = 100
        bo.run_optimization(max_iter = max_iter, max_time = max_time, eps = tolerance, verbosity=True)
        W = np.array([bo.x_opt])
        np.save(cwd + '/Flat_Results/Weights_' + str(NO_BINS), W)

