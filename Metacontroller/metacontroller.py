from utils.mouselab_metacontroller import MouselabMeta
import numpy as np
import time
import GPyOpt

def meta_controller(W_low, W_high, TREE, INIT, HIGH_COST=1, LOW_COST=1, SWITCH_COST=0, SEED=None, num_episodes=100, term_belief=False, log=False, init_func=None, disable_meta=False, exact_seed=False, cost_function="Basic", ground_truths = None):    
    """ Runs the meta controller for a number of episodes and returns rewards and actions taken.

    Args:
        W_low ([float]): Low level weights
        W_high ([float]): Goal weights
        TREE ([int]): MDP structure
        INIT ([int]): MDP reward distribution per node
        HIGH_COST (int, optional): Cost of computing a goal node. Defaults to 1.
        LOW_COST (int, optional): Cost of computing a non goal node. Defaults to 1.
        SWITCH_COST (int, optional): Cost of switching goals. Defaults to 0.
        SEED (int, optional): Seed to fix random MDP initialization. Defaults to None.
        num_episodes (int, optional): Number of episodes to evaluate the MDP on. Defaults to 100.
        term_belief (bool, optional): If true the expected reward instead of the real reward is used. Defaults to False.
        log (bool, optional): Prints computation steps to the console if true. Defaults to False.
        init_func (func, optional): Optional function to manipulate MDP initialization. Defaults to None.

    Returns:
        [int]: Reward of each episode
        [[int]]: Actions of each episode 
    """

    w5 = W_high[:,0]
    w6 = 1 - w5
    w7 = W_high[:,1]
    
    w1 = W_low[:,0]
    w2 = W_low[:,1]
    w4 = W_low[:,2]
    if cost_function == "Actionweight" or cost_function == "Independentweight":
        w0 = W_low[:,3]
    w3 = 1 - w1 - w2

    if cost_function == "Basic":
        simple_cost = True
    else:
        simple_cost = False

    def voc_estimate_low(env, x):
        features = env.action_features(x)
        if cost_function == "Basic":
            return w1*features[1] + w2*features[3] + w3*features[2] + w4*features[0]
        elif cost_function == "Hierarchical":
            return w1*features[1] + w2*features[3] + w3*features[2] + w4*(w1*features[0][0] + w3*features[0][1] + w2*features[0][2])
        elif cost_function == "Actionweight":
            return w1*features[1] + w2*features[3] + w3*features[2] + w4*(features[0][0] + w0*(features[0][1]))
        elif cost_function == "Novpi":
            return w1*features[1] + w2*features[3] + w3*features[2] + w4*(w1*features[0][0] + w3*(features[0][1]/16))
        elif cost_function == "Proportional":
            return w1*features[1] + w2*features[3] + w3*features[2] + w4*(features[0][0] + (features[0][1]/16))
        elif cost_function == "Independentweight":
            return w1*features[1] + w2*features[3] + w3*features[2] + w4*features[0][0] + w0*features[0][1]
    
    
    def voc_estimate_high(env, x):
        #features[0] is cost(action)
        #features[1] is myopicVOC(action) [aka VOI]
        #features[2] is vpi_action [aka VPIsub; the value of perfect info of branch]
        #features[3] is vpi(beliefstate)
        #features[4] is expected term reward of current state
        features = env.action_features(x) 
        if cost_function == "Basic":
            return w5*features[1] + w6*features[3] + w7*features[0]
        elif cost_function == "Hierarchical":
            return w5*features[1] + w6*features[3] + w7*(w5*features[0][0] + w6*features[0][2])
        else:
            return w5*features[1] + w6*features[3] + w7*(features[0][0])

    episode_rewards = []
    episode_actions = []
    for i in range(num_episodes):
        if log:
            print(f"\nEpisode {i+1}")
    #for truth in ground_truths:
        if init_func is not None:
            INIT = init_func()
        if SEED is not None:
            if not exact_seed:
                env = MouselabMeta(TREE, INIT, term_belief=term_belief, high_cost=HIGH_COST, low_cost=LOW_COST, seed=1000*SEED + i, simple_cost=simple_cost)
            else:
                env = MouselabMeta(TREE, INIT, term_belief=term_belief, high_cost=HIGH_COST, low_cost=LOW_COST, seed=SEED + i, simple_cost=simple_cost)
        else: 
            env = MouselabMeta(TREE, INIT, term_belief=term_belief, high_cost=HIGH_COST, low_cost=LOW_COST, simple_cost=simple_cost)
        #env.ground_truth = truth
        rewards = []
        actions = []
        previous_goal = None # Keep track of the last goal to add switch costs
        done = False
        while not done: 
            # Create high level MDP
            high_env, goal_to_meta, _ = env.get_goal_MDP()
            high_done = False
            high_actions = []
            # Run high level controller until termination
            while not high_done:
                possible_actions = list(high_env.actions(high_env._state))
                action_rewards = [voc_estimate_high(high_env, a) for a in possible_actions]
                action_taken = possible_actions[np.argmax(action_rewards)]
                _, rew, high_done, _ = high_env._step(action_taken)
                if action_taken in goal_to_meta.keys():
                    high_actions.append(action_taken)
                    if log:
                        print(f"{action_taken} (meta:{goal_to_meta[action_taken]}) High level action with VOC {max(action_rewards)} and value {high_env._state[action_taken]}")
            
            # Transfer high level actions to the real MDP
            for high_level_action in high_actions:
                real_action = goal_to_meta[high_level_action]
                _, rew, _, _ = env._step(real_action)
                rewards.append(-HIGH_COST)
                actions.append(real_action)
            del high_env, goal_to_meta

            # Get the 2 best goal rewards
            goal_rewards = [env.get_goal_reward(goal).expectation() if hasattr(env.get_goal_reward(goal), "sample") else env.get_goal_reward(goal) for goal in env.goals]
            best_goal_index = np.argmax(goal_rewards)
            best_goal = env.goals[best_goal_index]
            
            #print(f"Best goal: {best_goal} with reward {goal_rewards[best_goal_index]}")
            # Low level controller
            if disable_meta:
                low_env, low_to_meta, _ = env.get_low_level_MDP(best_goal)
            else:
                # Find best alternative goal and add dummy node
                remaining_goal_rewards = [r for i, r in enumerate(goal_rewards) if i != best_goal_index]
                alternative_goal_reward = max(remaining_goal_rewards)
                low_env, low_to_meta, _ = env.get_low_level_MDP(best_goal, alternative_goal_reward)
            low_done = False
            low_actions = []
            while not low_done:
                possible_actions = list(low_env.actions(low_env._state))
                action_rewards = [voc_estimate_low(low_env, a) for a in possible_actions]
                action_taken = possible_actions[np.argmax(action_rewards)]
                _, rew, low_done, _ = low_env._step(action_taken)
                if action_taken in low_to_meta.keys():
                    low_actions.append(action_taken)
                    if log:
                        print(f"{action_taken} (meta:{low_to_meta[action_taken]}) Low level action with VOC {max(action_rewards)} and value {low_env._state[action_taken]}")

            # Transfer high level actions to the real MDP
            for low_level_action in low_actions:
                meta_action = low_to_meta[low_level_action]
                _, rew, _, _ = env._step(meta_action) # BUG metaaction is sometimes already observed (?)
                rewards.append(-LOW_COST)
                actions.append(meta_action)
                #print("Meta step", meta_action)
            del low_env, low_to_meta

            # Calculate new expected reward of the chosen goal
            expected_reward = env.get_goal_reward(best_goal).expectation() if hasattr(env.get_goal_reward(best_goal), "sample") else env.get_goal_reward(best_goal)

            # Terminate if the expected reward is higher than the alternative goal
            if disable_meta:
                done = True
                if term_belief:
                    rew = expected_reward
                else:
                    path = env.get_max_path(best_goal)
                    rew = sum([env.ground_truth[n] for n in path])
                rewards.append(rew)
            elif (expected_reward >= alternative_goal_reward + SWITCH_COST):
                _, rew, done, _ = env._step(env.term_action)
                rewards.append(rew)
            else:
                rewards.append(-SWITCH_COST)
                # print(f"Better alternative goal discovered: {alternative_goal_reward} > {expected_reward}")
        del env
        episode_rewards.append(sum(rewards))
        if log:
            print(f"Episode reward:", sum(rewards))
        episode_actions.append(actions)

    return episode_rewards, episode_actions

def split_W(W, cost_function):
    """Splits the weight arrray into low and high level weights.

    Args:
        W (np.array): Meta controller weights

    Returns:
        np.array: Low level weights
        np.array: High level weights
    """
    w1 = W[:,0] # Low Myopic 
    w2 = W[:,1] # Low VPI
    w4 = W[:,2] # Low cost
    w5 = W[:,3] # High Myopic
    w7 = W[:,4] # High cos

    if cost_function == "Actionweight" or cost_function == "Independentweight":
        w0 = W[:,5]
        W_low = np.array([[*w1, *w2, *w4, *w0]])
    else:
        W_low = np.array([[*w1, *w2, *w4]])
    W_high = np.array([[*w5, *w7]])
    return W_low, W_high

def optimize(TREE, INIT, LOW_COST, HIGH_COST, SWITCH_COST, SEED, low_nodes, goal_nodes, evaluated_episodes=100, samples=30, iterations=50, disable_meta=False, cost_function="basic"):
    """Optimizes the meta controller weights for BMPS using Bayesian optimization.

    Args:
        TREE ([int]): MDP structure
        INIT ([int]): MDP reward distribution per node
        HIGH_COST (int, optional): Cost of computing a goal node.
        LOW_COST (int, optional): Cost of computing a non goal node.
        SWITCH_COST (int, optional): Cost of switching goals.
        SEED (int, optional): Seed to fix random MDP initialization.
        low_nodes (int): Number of nodes within a subtree to a goal. Used to constrain the cost parameter in the optimization process. If subtree sizes are variable the max size should be used.
        goal_nodes (int): Number of goal nodes.
        samples (int, optional): Number of initial random guesses before optimization. Defaults to 30.
        iterations (int, optional): Number of optimization steps. Defaults to 50.

    Returns:
        np.array: Weights after optimization
        float: Runtime in seconds
    """

    def blackbox(W):
        W_low, W_high = split_W(W, cost_function)

        print("Testing weights:", W[0])

        n = evaluated_episodes
        rewards, actions = meta_controller(W_low, W_high, TREE=TREE, INIT=INIT, HIGH_COST=HIGH_COST, LOW_COST=LOW_COST, SWITCH_COST=SWITCH_COST, SEED=SEED, num_episodes=n, term_belief=True, disable_meta=disable_meta, cost_function=cost_function)
        #rewards, actions = meta_controller(W_low, W_high, TREE=TREE, INIT=INIT, HIGH_COST=HIGH_COST, LOW_COST=LOW_COST, SWITCH_COST=SWITCH_COST, SEED=None, num_episodes=n, term_belief=True, disable_meta=disable_meta, cost_function=cost_function)

        average_reward = sum(rewards)/n
        print("Average reward", average_reward)

        return -average_reward

    if cost_function == "Actionweight" or cost_function == "Independentweight":
        space = [{'name': 'w1', 'type': 'continuous', 'domain': (0,1)},
                {'name': 'w2', 'type': 'continuous', 'domain': (0,1)},
                {'name': 'w4', 'type': 'continuous', 'domain': (1,low_nodes)},
                {'name': 'w5', 'type': 'continuous', 'domain': (0,1)},
                {'name': 'w7', 'type': 'continuous', 'domain': (1,goal_nodes)},
                {'name': 'w0', 'type': 'continuous', 'domain': (0,1)}]
    else:
        space = [{'name': 'w1', 'type': 'continuous', 'domain': (0,1)},
                {'name': 'w2', 'type': 'continuous', 'domain': (0,1)},
                {'name': 'w4', 'type': 'continuous', 'domain': (1,low_nodes)},
                {'name': 'w5', 'type': 'continuous', 'domain': (0,1)},
                {'name': 'w7', 'type': 'continuous', 'domain': (1,goal_nodes)}]

    constraints = [{'name': 'part_1', 'constraint': 'x[:,0] + x[:,1] - 1'}]

    feasible_region = GPyOpt.Design_space(space = space, constraints = constraints)

    np.random.seed(123456)

    # Sets number of random samples before optimization starts
    initial_design = GPyOpt.experiment_design.initial_design('random', feasible_region, samples)
    # --- CHOOSE the objective
    objective = GPyOpt.core.task.SingleObjective(blackbox)

    # --- CHOOSE the model type
    #This model does Maximum likelihood estimation of the hyper-parameters.
    #model = GPyOpt.models.GPModel(exact_feval=False,noise_var=10,optimize_restarts=10,verbose=False)
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
    # Sets number of optimization steps
    max_iter  = iterations
    time_start = time.time()
    train_tic = time_start
    bo.run_optimization(max_iter = max_iter, max_time = max_time, eps = tolerance, verbosity=True)

    W = np.array([bo.x_opt])

    duration = time.time() - train_tic

    print("\nSeconds:", duration)
    print("Weights:", W)
    blackbox(W)

    return W, duration

def eval(W, n, TREE, INIT, HIGH_COST, LOW_COST, SWITCH_COST, SEED, term_belief=False, log=False, disable_meta=False, cost_function="Basic", ground_truths= None):
    """ Evaluates the meta controller and logs the execution time.

    Args:
        W (np.array): BMPS weights
        n (int): Number of episodes for evaluation.
        TREE ([int]): MDP structure
        INIT ([int]): MDP reward distribution per node
        HIGH_COST (int, optional): Cost of computing a goal node. Defaults to 1.
        LOW_COST (int, optional): Cost of computing a non goal node. Defaults to 1.
        SWITCH_COST (int, optional): Cost of switching goals. Defaults to 0.
        SEED (int, optional): Seed to fix random MDP initialization. Defaults to None.
        term_belief (bool, optional): If true the expected reward instead of the real reward is used. Defaults to False.
        log (bool, optional): Prints computation steps to the console if true. Defaults to False.

    Returns:
        [int]: Reward of each episode
        [[int]]: Actions of each episode 
    """

    W_low, W_high = split_W(W, cost_function=cost_function)
    eval_tic = time.time()
    rewards, actions = meta_controller(W_low, W_high, ground_truths=ground_truths, TREE=TREE, INIT=INIT, HIGH_COST=HIGH_COST, LOW_COST=LOW_COST, SWITCH_COST=SWITCH_COST, SEED=SEED, num_episodes=n, term_belief=term_belief, log=log, disable_meta=disable_meta, exact_seed=True, cost_function=cost_function)
    print("Seconds:", time.time() - eval_tic)
    print("Average reward:", np.mean(rewards))
    return rewards, actions

def trace(W, TREE, INIT, HIGH_COST=1, LOW_COST=1, SWITCH_COST=0, SEED=0, term_belief=False, disable_meta=False, cost_function="Basic"):
    W_low, W_high = split_W(W, cost_function)

    w5 = W_high[:,0]
    w6 = 1 - w5
    w7 = W_high[:,1]
    
    w1 = W_low[:,0]
    w2 = W_low[:,1]
    w4 = W_low[:,2]
    if cost_function == "Actionweight" or cost_function == "Independentweight":
        w0 = W_low[:,3]
    w3 = 1 - w1 - w2

    if cost_function == "Basic":
        simple_cost = True
    else:
        simple_cost = False

    def voc_estimate_low(env, x):
        features = env.action_features(x)
        if cost_function == "Basic":
            return w1*features[1] + w2*features[3] + w3*features[2] + w4*features[0]
        elif cost_function == "Hierarchical":
            return w1*features[1] + w2*features[3] + w3*features[2] + w4*(w1*features[0][0] + w3*features[0][1] + w2*features[0][2])
        elif cost_function == "Actionweight":
            return w1*features[1] + w2*features[3] + w3*features[2] + w4*(features[0][0] + w0*(features[0][1]))
        elif cost_function == "Novpi":
            return w1*features[1] + w2*features[3] + w3*features[2] + w4*(w1*features[0][0] + w3*(features[0][1]/16))
        elif cost_function == "Proportional":
            return w1*features[1] + w2*features[3] + w3*features[2] + w4*(features[0][0] + (features[0][1]/16))
        elif cost_function == "Independentweight":
            return w1*features[1] + w2*features[3] + w3*features[2] + w4*features[0][0] + w0*features[0][1]
    
    
    def voc_estimate_high(env, x):
        #features[0] is cost(action)
        #features[1] is myopicVOC(action) [aka VOI]
        #features[2] is vpi_action [aka VPIsub; the value of perfect info of branch]
        #features[3] is vpi(beliefstate)
        #features[4] is expected term reward of current state
        features = env.action_features(x) 
        if cost_function == "Basic":
            return w5*features[1] + w6*features[3] + w7*features[0]
        elif cost_function == "Hierarchical":
            return w5*features[1] + w6*features[3] + w7*(w5*features[0][0] + w6*features[0][2])
        else:
            return w5*features[1] + w6*features[3] + w7*(features[0][0])

    env = MouselabMeta(TREE, INIT, term_belief=term_belief, high_cost=HIGH_COST, low_cost=LOW_COST, seed=SEED, simple_cost=simple_cost)

    print(env.ground_truth)
    ground_truth = env.ground_truth

    previous_goal = None # Keep track of the last goal to add switch costs
    done = False
    actions = []
    while not done: 
        # Create high level MDP
        high_env, goal_to_meta, _ = env.get_goal_MDP()
        print(high_env.ground_truth)
        high_done = False
        high_actions = []
        # Run high level controller until termination
        while not high_done:
            possible_actions = list(high_env.actions(high_env._state))
            action_rewards = [voc_estimate_high(high_env, a) for a in possible_actions]
            action_taken = possible_actions[np.argmax(action_rewards)]
            _, rew, high_done, _ = high_env._step(action_taken)
            if action_taken in goal_to_meta.keys():
                high_actions.append(action_taken)
                print("High action", action_taken, goal_to_meta[action_taken], high_env.ground_truth[action_taken])
            else:
                print("High term action")
        
        # Transfer high level actions to the real MDP
        for high_level_action in high_actions:
            real_action = goal_to_meta[high_level_action]
            _, rew, _, _ = env._step(real_action)
            actions.append(real_action)
        del high_env, goal_to_meta

        # Get the 2 best goal rewards
        goal_rewards = [env.get_goal_reward(goal).expectation() if hasattr(env.get_goal_reward(goal), "sample") else env.get_goal_reward(goal) for goal in env.goals]
        best_goal_index = np.argmax(goal_rewards)
        best_goal = env.goals[best_goal_index]
        
        #print(f"Best goal: {best_goal} with reward {goal_rewards[best_goal_index]}")
        # Low level controller
        if disable_meta:
            low_env, low_to_meta, _ = env.get_low_level_MDP(best_goal)
        else:
            # Find best alternative goal and add dummy node
            remaining_goal_rewards = [r for i, r in enumerate(goal_rewards) if i != best_goal_index]
            alternative_goal_reward = max(remaining_goal_rewards)
            low_env, low_to_meta, _ = env.get_low_level_MDP(best_goal, alternative_goal_reward)
        low_done = False
        low_actions = []
        while not low_done:
            possible_actions = list(low_env.actions(low_env._state))
            action_rewards = [voc_estimate_low(low_env, a) for a in possible_actions]
            #print("Low rewards", action_rewards)
            action_taken = possible_actions[np.argmax(action_rewards)]
            _, rew, low_done, _ = low_env._step(action_taken)
            if action_taken in low_to_meta.keys():
                low_actions.append(action_taken)
                print("Low action", action_taken, low_to_meta[action_taken], low_env.ground_truth[action_taken])
            else:
                print("Low term action")

        # Transfer high level actions to the real MDP
        for low_level_action in low_actions:
            meta_action = low_to_meta[low_level_action]
            _, rew, _, _ = env._step(meta_action) # BUG metaaction is sometimes already observed (?)
            actions.append(meta_action)
        del low_env, low_to_meta

        # Calculate new expected reward of the chosen goal
        expected_reward = env.get_goal_reward(best_goal).expectation() if hasattr(env.get_goal_reward(best_goal), "sample") else env.get_goal_reward(best_goal)

        # Terminate if the expected reward is higher than the alternative goal
        if disable_meta or (expected_reward >= alternative_goal_reward + SWITCH_COST):
            done = True
    
    if not disable_meta:
        paths = list(env.optimal_paths())
        path = paths[0]
        _, rew, done, _ = env._step(env.term_action)
    else: 
        # Best path to selected goal
        path = env.get_max_path(best_goal)

    return actions, ground_truth, path