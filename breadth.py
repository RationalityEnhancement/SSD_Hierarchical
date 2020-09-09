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

    def aspiration_val(env):
        state = env.discretize(env._state, NO_BINS)
        return env.expected_term_reward_disc(state)

    def blackboxfunc(lgA):
        global node_types

        #     logA = lgA[:,0]
        #     A = math.exp(logA)
        A = lgA[:, 0]
        print(A)
        num_episodes = 100
        cumreturn = 0

        for i in range(num_episodes):
            #         print("i = {}".format(i))
            env = MouselabEnv.new(NO_OPTION, TREE, reward=reward, option_set=OPTION_SET, branch_cost=BRANCH_COST,
                                  switch_cost=SWITCH_COST, tau=TAU,
                                  seed=SEED + i)
            done_exploration_flag = False
            current_nodes = [0]
            nodes_clicked = []
            exp_return = 0
            env._reset()
            count = 0
            while not done_exploration_flag:
                parent_nodes = list(set(current_nodes))
                random.shuffle(parent_nodes)
                # print("Count = {}, Parent Nodes = {}".format(count, parent_nodes))
                if len(parent_nodes) == 0:
                    done_exploration_flag = True
                    break
                current_nodes = []
                for parent_node in parent_yield(parent_nodes):
                    # print("parent_node = {}".format(parent_node))
                    for c in child_yield(TREE, parent_node):
                        # print(c)
                        current_nodes.append(c)
                        if hasattr(env._state[c], 'sample'):
                            _, r, _, _ = env._step(c)
                            nodes_clicked.append(c)
                            exp_return += r
                        aspiration_value = aspiration_val(env)
                        if aspiration_value >= A:
                            done_exploration_flag = True
                            break
                count+=1
            exp_return += env.term_reward_actual(env._state)
            cumreturn += exp_return

        return -cumreturn / num_episodes

    def blackboxfunc_test(lgA):
        global node_types

        #     logA = lgA[:,0]
        #     A = math.exp(logA)
        A = lgA[:, 0]
        print("A = {}".format(A))
        num_episodes = 100
        cumreturn = 0
        df = pd.DataFrame(columns=['i', 'return', 'actions', 'Actual Path', 'Time', 'ground_truth'])
        for i in range(num_episodes):
            print("i = {}".format(i))
            env = MouselabEnv.new(NO_OPTION, TREE, reward=reward, option_set=OPTION_SET, branch_cost=BRANCH_COST,
                                  switch_cost=SWITCH_COST, tau=TAU,
                                  seed=SEED + i)
            done_exploration_flag = False
            current_nodes = [0]
            nodes_clicked = []
            exp_return = 0
            env._reset()
            count = 0
            while not done_exploration_flag:
                parent_nodes = list(set(current_nodes))
                random.shuffle(parent_nodes)
                # print("Count = {}, Parent Nodes = {}".format(count, parent_nodes))
                if len(parent_nodes) == 0:
                    done_exploration_flag = True
                    break
                current_nodes = []
                for parent_node in parent_yield(parent_nodes):
                    # print("parent_node = {}".format(parent_node))
                    for c in child_yield(TREE, parent_node):
                        # print(c)
                        current_nodes.append(c)
                        if hasattr(env._state[c], 'sample'):
                            _, r, _, _ = env._step(c)
                            nodes_clicked.append(c)
                            exp_return += r
                        aspiration_value = aspiration_val(env)
                        if aspiration_value >= A:
                            done_exploration_flag = True
                            break
                count+=1
            exp_return += env.term_reward_actual(env._state)
            df.loc[i] = [i, exp_return, nodes_clicked, actual_path(env, env._state), env.ground_truth]
            cumreturn += exp_return


        df.to_csv(cwd + '/Breadth_Results/breadth_' + str(NO_BINS) + '.csv')
        np.save(cwd + '/Breadth_Results/CumResult_' + str(NO_BINS), cumreturn / num_episodes)
        print("Cumulative Reward".format(cumreturn / num_episodes))
        return -cumreturn / num_episodes

    if(TRAIN_FLAG == 0):
        A = np.load(cwd + '/Breadth_Results/Value_' + str(NO_BINS)+ '.npy')
        eval_tic = time.time()
        blackboxfunc_test(A)
        toc = time.time()
        np.save(cwd + '/Breadth_Results/Eval_Time_' + str(NO_BINS), toc - eval_tic)

    else:
        space = [{'name': 'A', 'type': 'continuous', 'domain': (0, 300)}]
        constraints = []
        feasible_region = GPyOpt.Design_space(space=space, constraints=constraints)
        # --- CHOOSE the intial design
        from numpy.random import seed  # fixed seed

        seed(123456)

        initial_design = GPyOpt.experiment_design.initial_design('random', feasible_region, 100)

        # --- CHOOSE the objective
        objective = GPyOpt.core.task.SingleObjective(blackboxfunc)

        # --- CHOOSE the model type
        # This model does Maximum likelihood estimation of the hyper-parameters.
        model = GPyOpt.models.GPModel(exact_feval=True, optimize_restarts=10, verbose=False)

        # --- CHOOSE the acquisition optimizer
        aquisition_optimizer = GPyOpt.optimization.AcquisitionOptimizer(feasible_region)

        # --- CHOOSE the type of acquisition
        acquisition = GPyOpt.acquisitions.AcquisitionEI(model, feasible_region, optimizer=aquisition_optimizer)

        # --- CHOOSE a collection method
        evaluator = GPyOpt.core.evaluators.Sequential(acquisition)

        bo = GPyOpt.methods.ModularBayesianOptimization(model, feasible_region, objective, acquisition, evaluator,
                                                        initial_design)

        # --- Stop conditions
        max_time = None
        tolerance = 1e-6  # distance between two consecutive observations

        try:
            os.makedirs(cwd + '/Breadth_Results')
        except:
            pass

        # Run the optimization
        time_start = time.time()
        train_1_tic = time_start
        max_iter = 100
        bo.run_optimization(max_iter=max_iter, max_time=max_time, eps=tolerance, verbosity=True)

        A = np.array([bo.x_opt])
        np.save(cwd + '/Breadth_Results/Value_' + str(NO_BINS), A)

        train_1_toc = time.time()

        np.save(cwd + '/Breadth_Results/Total_Train_Time_' + str(NO_BINS), train_1_toc - train_1_tic)