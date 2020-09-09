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
            #         tc = time.time()
            exp_return = 0
            no_clicks = 0
            nodes_clicked = []
            done_exploration_flag = False
            all_paths = list(env.allpaths)
            while not done_exploration_flag:
                paths = all_paths  # Paths
                if paths == []:
                    break
                paths.sort(key=len)  # Sort the paths according to path length
                #             print("Paths = {}".format(paths))
                path = paths[0]  # Explore shortest path first
                #             print(path)
                all_paths.remove(path)  # Remove from path to explore
                forward_ptr = 1  # 1 to skip the 0th node
                backward_ptr = len(path) - 1
                max_clicks_possible = len(path) - 1
                #             print("Max = {}".format(max_clicks_possible))
                while max_clicks_possible > 0:
                    if path[forward_ptr] not in nodes_clicked:
                        _, r, _, _ = env._step(path[forward_ptr])  # r is the cost of click
                        nodes_clicked.append(path[forward_ptr])
                        print("Action = {}".format(path[forward_ptr]))
                        exp_return += r
                        no_clicks += 1
                    max_clicks_possible -= 1
                    forward_ptr += 1
                    aspiration_value = aspiration_val(env)
                    print("Asp Val = {}".format(aspiration_value))
                    if (aspiration_value >= A):
                        done_exploration_flag = True
                        break

                    #                 print("Backward = {}".format(backward_ptr))
                    #                 print("Backward Node = {}".format(path[backward_ptr]))
                    if path[backward_ptr] not in nodes_clicked:
                        _, r, _, _ = env._step(path[backward_ptr])  # r is the cost of click
                        nodes_clicked.append(path[backward_ptr])
                        print("Action = {}".format(path[backward_ptr]))
                        exp_return += r
                        no_clicks += 1
                    max_clicks_possible -= 1

                    backward_ptr -= 1
                    aspiration_value = aspiration_val(env)
                    print("Asp Val = {}".format(aspiration_value))
                    if (aspiration_value >= A):
                        done_exploration_flag = True
                        break

            _, r, _, _ = env._step(
                env.term_action)  # r is the reward gotten if followed best path according to current belief state
            exp_return += r
            df.loc[i] = [i, exp_return, nodes_clicked, actual_path(env, env._state), env.ground_truth]
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
            env_tic = time.time()
            #         tc = time.time()
            exp_return = 0
            no_clicks = 0
            nodes_clicked = []
            done_exploration_flag = False
            all_paths = list(env.allpaths)
            while not done_exploration_flag:
                paths = all_paths  # Paths
                if paths == []:
                    break
                paths.sort(key=len)  # Sort the paths according to path length
                #             print("Paths = {}".format(paths))
                path = paths[0]  # Explore shortest path first
                #             print(path)
                all_paths.remove(path)  # Remove from path to explore
                forward_ptr = 1  # 1 to skip the 0th node
                backward_ptr = len(path) - 1
                max_clicks_possible = len(path) - 1
                #             print("Max = {}".format(max_clicks_possible))
                while max_clicks_possible > 0:
                    if path[forward_ptr] not in nodes_clicked:
                        _, r, _, _ = env._step(path[forward_ptr])  # r is the cost of click
                        nodes_clicked.append(path[forward_ptr])
                        print("Action = {}".format(path[forward_ptr]))
                        exp_return += r
                        no_clicks += 1
                    max_clicks_possible -= 1
                    forward_ptr += 1
                    aspiration_value = aspiration_val(env)
                    print("Asp Val = {}".format(aspiration_value))
                    if (aspiration_value >= A):
                        done_exploration_flag = True
                        break

                    #                 print("Backward = {}".format(backward_ptr))
                    #                 print("Backward Node = {}".format(path[backward_ptr]))
                    if path[backward_ptr] not in nodes_clicked:
                        _, r, _, _ = env._step(path[backward_ptr])  # r is the cost of click
                        nodes_clicked.append(path[backward_ptr])
                        print("Action = {}".format(path[backward_ptr]))
                        exp_return += r
                        no_clicks += 1
                    max_clicks_possible -= 1

                    backward_ptr -= 1
                    aspiration_value = aspiration_val(env)
                    print("Asp Val = {}".format(aspiration_value))
                    if (aspiration_value >= A):
                        done_exploration_flag = True
                        break

            _, r, _, _ = env._step(
                env.term_action)  # r is the reward gotten if followed best path according to current belief state
            exp_return += r
            df.loc[i] = [i, exp_return, nodes_clicked, actual_path(env, env._state), env.ground_truth]
            cumreturn += exp_return

        df.to_csv(cwd + '/Bidirectional_Results/bidirectional_' + str(NO_BINS) + '.csv')
        np.save(cwd + '/Bidirectional_Results/CumResult_' + str(NO_BINS), cumreturn / num_episodes)
        print("Cumulative Reward".format(cumreturn / num_episodes))
        return -cumreturn / num_episodes

    if(TRAIN_FLAG == 0):
        A = np.load(cwd + '/Bidirectional_Results/Value_' + str(NO_BINS)+ '.npy')
        eval_tic = time.time()
        blackboxfunc_test(A)
        toc = time.time()
        np.save(cwd + '/Bidirectional_Results/Eval_Time_' + str(NO_BINS), toc - eval_tic)

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
            os.makedirs(cwd + '/Bidirectional_Results')
        except:
            pass

        # Run the optimization
        time_start = time.time()
        train_1_tic = time_start
        max_iter = 100
        bo.run_optimization(max_iter=max_iter, max_time=max_time, eps=tolerance, verbosity=True)

        A = np.array([bo.x_opt])
        np.save(cwd + '/Bidirectional_Results/Value_' + str(NO_BINS), A)

        train_1_toc = time.time()

        np.save(cwd + '/Bidirectional_Results/Total_Train_Time_' + str(NO_BINS), train_1_toc - train_1_tic)