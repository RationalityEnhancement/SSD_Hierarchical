from __future__ import division
import numpy as np
import random
import tensorflow as tf
import tensorflow.contrib.slim as slim
import os, yaml
import pandas as pd
import shutil
import argparse
import time
from baselines import logger
import glob
from itertools import compress
import time
import sys

NO_BINS = 4


def myopic_voc(env, x):
    # print(x)
    state_disc = env.discretize(env._state, NO_BINS)
    # print(env.myopic_voc(x, state_disc) + env.cost)
    return env.myopic_voc(x, state_disc) + env.cost  # env.cost already takes sign into account


class Qnetwork():
    def __init__(self, h_size, n_obs, n_actions):
        print("h_size = {}, n_obs = {}, n_actions = {}".format(h_size,n_obs,n_actions))
        self.scalarInput = tf.placeholder(
            shape=[None, n_obs], dtype=tf.float32)
        print(self.scalarInput.get_shape())
        self.imageIn = tf.reshape(self.scalarInput, shape=[-1, 1, n_obs, 1])
        print(self.imageIn.get_shape())
        self.conv1 = slim.conv2d(inputs=self.imageIn, num_outputs=16, kernel_size=[
            1, 2], stride=[1, 1], padding='SAME', biases_initializer=None)
        print(self.conv1.get_shape())
        # self.conv2 = slim.conv2d(inputs=self.conv1, num_outputs=32, kernel_size=[
        #                          1, 2], stride=[1, 1], padding='SAME', biases_initializer=None)
        # print(self.conv2.get_shape())
        # self.conv3 = slim.conv2d(inputs=self.conv2, num_outputs=64, kernel_size=[
        #                          1, 2], stride=[1, 1], padding='SAME', biases_initializer=None)
        # print(self.conv3.get_shape())
        self.conv4 = slim.conv2d(inputs=self.conv1, num_outputs=h_size, kernel_size=[
            1, 2], stride=[1, 1], padding='SAME', biases_initializer=None)
        print(self.conv4.get_shape())
        # We take the output from the final convolutional layer and split it into separate advantage and value streams.
        # print_node = tf.print(self.conv3, [tf.shape(self.conv3)], "conv3 size: {}")
        self.streamAC, self.streamVC = tf.split(self.conv4, 2, 3)
        print(self.streamAC.get_shape())
        print(self.streamVC.get_shape())
        self.streamA = slim.flatten(self.streamAC)
        print(self.streamA.get_shape())
        self.streamV = slim.flatten(self.streamVC)
        xavier_init = tf.contrib.layers.xavier_initializer()
        self.AW = tf.Variable(xavier_init([h_size // 2, n_actions]))
        print(self.AW.get_shape())
        self.VW = tf.Variable(xavier_init([h_size // 2, 1]))
        self.Advantage = tf.matmul(self.streamA, self.AW)
        self.Value = tf.matmul(self.streamV, self.VW)
        print(self.Advantage.get_shape())
        print(self.Value.get_shape())
        # Then combine them together to get our final Q-values.
        self.Qout = self.Value + \
                    tf.subtract(self.Advantage, tf.reduce_mean(
                        self.Advantage, axis=1, keep_dims=True))
        print(self.Qout.get_shape())
        self.predict = tf.argmax(self.Qout, 1)
        print(self.predict.get_shape())
        # Below we obtain the loss by taking the sum of squares difference between the target and prediction Q values.
        self.targetQ = tf.placeholder(shape=[None], dtype=tf.float32)
        self.actions = tf.placeholder(shape=[None], dtype=tf.int32)
        self.actions_onehot = tf.one_hot(
            self.actions, n_actions, dtype=tf.float32)

        self.Q = tf.reduce_sum(tf.multiply(
            self.Qout, self.actions_onehot), axis=1)

        self.td_error = tf.square(self.targetQ - self.Q)
        self.loss = tf.reduce_mean(self.td_error)
        self.trainer = tf.train.AdamOptimizer(learning_rate=0.0001)
        self.updateModel = self.trainer.minimize(self.loss)


class experience_buffer():
    def __init__(self, buffer_size=8000):
        self.buffer = []
        self.buffer_size = buffer_size

    def add(self, experience):
        if len(self.buffer) + len(experience) >= self.buffer_size:
            self.buffer[0:(len(experience) + len(self.buffer)) -
                          self.buffer_size] = []
        self.buffer.extend(experience)

    def sample(self, size):
        return np.reshape(np.array(random.sample(self.buffer, size)), [size, 4])


def updateTargetGraph(tfVars, tau):
    total_vars = len(tfVars)
    op_holder = []
    for idx, var in enumerate(tfVars[0:total_vars // 2]):
        op_holder.append(tfVars[idx + total_vars // 2].assign(
            (var.value() * tau) + ((1 - tau) * tfVars[idx + total_vars // 2].value())))
    return op_holder


def updateTarget(op_holder, sess):
    for op in op_holder:
        sess.run(op)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('no_goals', type=str)
    parser.add_argument('train', type=str)
    parser.add_argument('n_env', type=str)
    # parser.add_argument('max_kl', type=float)
    parser.add_argument('start', type=float)
    args = parser.parse_args()

    rs = int(100 * (args.start + 1))
    np.random.seed(rs)

    # Read file and store in dictionary
    with open('experiment_params.yaml', 'r') as stream:
        params = yaml.load(stream)
    # Replacing values in file with arguments passed
    params['n_env'] = args.n_env
    params['algo_type'] = 'DQN'
    params['optimiser'] = 'Adam'
    # params['max_kl'] = args.max_kl
    params['rseed'] = rs

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


    def make_envs(seed=SEED, num_samples=1):
    envs = []
    for i in range(num_samples):
        # Create a Mouselab environment

        envs += \
            [MouselabEnv.new(NO_OPTION, TREE, reward=reward, option_set=OPTION_SET, branch_cost=BRANCH_COST, switch_cost=SWITCH_COST, tau=TAU,
                          seed=seed + i)]
    return envs

    log_path = os.path.join(cwd + 'Adaptive_Metareasoning_Results/results_dqn', args.algo_type + '_' + args.optimiser,
                            'rs_' + str(rs))
    logger.configure(dir=os.path.join(os.getcwd(), log_path))
    with open(os.path.join(log_path, 'params.yaml'), 'w') as outfile:
        yaml.dump(params, outfile, default_flow_style=False)
    path = log_path + "/models/temp/"  # Directory to save models

    if(TRAIN_FLAG == 0):
        print("Making Test Environments")
        test_env_array = make_envs(num_samples=int(args.n_env), seed=0)
        tf.reset_default_graph()
        mainQN = Qnetwork(int(params['h_size']), 1, 2)
        targetQN = Qnetwork(int(params['h_size']), 1, 2)

        init = tf.global_variables_initializer()

        saver = tf.train.Saver()

        trainables = tf.trainable_variables()

        targetOps = updateTargetGraph(trainables, float(params['tau']))

        myBuffer = experience_buffer()
        test_epLength = n_actions
        df = pd.DataFrame(columns=['i', 'RAC', 'return', 'actions', 'Actual Path', 'Time' ,'ground_truth'])
        resum = 0
        with tf.Session() as sess:
            ckpt = tf.train.get_checkpoint_state(path)
            saver.restore(sess, ckpt.model_checkpoint_path)
            for i in range(int(args.n_env)):
                env = test_env_array[i]
                print("i = {}".format(i))
                env_tic = time.time()
                q_max = env.expected_term_reward(tuple(env.ground_truth))
                q, _ = env.term_reward(env._state)
                q /= q_max
                qP = np.array([q])
                actions = []
                d = 0
                rewardAll = 0
                rAll = 0
                j = 0

                while j < test_epLength:
                    j += 1
                    possible_actions = list(env.actions(env._state))
                    possible_actions.remove(env.term_action)
                    if possible_actions == []:  # No more actions, need to terminate
                        actions.append(env.term_action)
                        _, reward, _, _ = env._step(env.term_action)
                        d = 1
                        rewardAll += reward
                        break
                    # take action that maximises estimated VOC
                    action_taken = max(possible_actions, key=lambda x: myopic_voc(env, x))
                    r = myopic_voc(env, action_taken)  # RAC
                    _, reward, _, _ = env._step(action_taken)
                    q1, _ = env.term_reward(env._state)
                    q1 /= q_max
                    q1P = np.array([q1])
                    d = sess.run(mainQN.predict, feed_dict={
                        mainQN.scalarInput: [qP]})[0]
                    print("Q {} d = {}".format(q1, d))
                    rAll += r
                    rewardAll += reward
                    actions.append(action_taken)
                    q = q1
                    qP = q1P
                    if d == 1: # Terminate 
                        actions.append(env.term_action)
                        _, reward, _, _ = env._step(env.term_action)
                        rewardAll += reward
                        break

                if d == 0: # Test steps ran out, time to terminate anyway
                    actions.append(env.term_action)
                    _, reward, _, _ = env._step(env.term_action)
                    rewardAll += reward
                env_toc = time.time()
                df.loc[i] = [i, rAll, rewardAll, actions, env.actual_path(env._state), env_toc - env_tic, env.ground_truth]
                # print(i, rAll, actions)
                resum += rAll

        policy_path = path + 'dqn.csv'
        df.to_csv(policy_path)
        print(resum / int(args.n_env))
        np.save(path + 'Result', resum / int(args.n_env))
        toc = time.perf_counter()
        print("Took {} min {} seconds to run the code".format((toc - tic) // 60, (toc - tic) % 60))
        np.save(path + 'Total_Time_' + str(NO_BINS), toc - tic)


    else:

        # Making environments
        print("Making Environments")
        env_array = make_envs(num_samples=int(args.n_env), seed=0)

        # ### Training the network
        print("Training Begins!")
        tic = time.perf_counter()
        try:
            os.makedirs(path)
        except:
            pass
        # How many steps of training to reduce startE to endE.
        annealing_steps = params['num_episodes'] * 2.2
        print("Annealing Steps {}".format(annealing_steps))
        # How many steps of random actions before training begins.
        pre_train_steps = params['num_episodes']

        env = env_array[np.random.randint(0, int(args.n_env))]
        n_obs = env.n_obs
        n_actions = env.n_actions
        max_epLength = env.n_actions  # The max allowed length of our episode.
        test_epLength = n_actions

        tf.reset_default_graph()
        mainQN = Qnetwork(int(params['h_size']), 1, 2)
        targetQN = Qnetwork(int(params['h_size']), 1, 2)

        init = tf.global_variables_initializer()

        saver = tf.train.Saver()

        trainables = tf.trainable_variables()

        targetOps = updateTargetGraph(trainables, params['tau'])

        myBuffer = experience_buffer()

        # Set the rate of random action decrease.
        e = float(params['startE'])
        stepDrop = (float(params['startE']) - float(params['endE'])) / annealing_steps

        # create lists to contain total rewards and steps per episode
        jList = []
        rList = []
        total_steps = 0

        # try:
        #     shutil.rmtree('./tfb')
        # except OSError:
        #     print ("no dir")
        #
        #
        # files = glob.glob(path+'*')
        # for f in files:
        #     os.remove(f)
        tfb_path = log_path + "/tfb"
        summary_writer = tf.summary.FileWriter(tfb_path)
        df = pd.DataFrame(columns=['i', 'return', 'actions', 'ground_truth'])
        resum = 0
        with tf.Session() as sess:
            sess.run(init)
            if params['load_model'] == True:
                print('Loading Model...')
                ckpt = tf.train.get_checkpoint_state(path)
                saver.restore(sess, ckpt.model_checkpoint_path)
            for i in range(params['num_episodes']):

                episodeBuffer = experience_buffer()
                # Reset environment and get first new observation
                env = env_array[np.random.randint(0, int(args.n_env))]
                q_max = env.expected_term_reward(tuple(env.ground_truth))
                q, _ = env.term_reward(env._state)
                q /= q_max
                qP = np.array([q])
                d = 0
                rAll = 0
                j = 0
                # The Q-Network
                while j < max_epLength:
                    j += 1
                    # Choose an action by greedily (with e chance of random action) from the Q-network
                    if np.random.rand(1) < e or total_steps < pre_train_steps:
                        d = random.choice([0, 1])
                    else:
                        # print("Exploit")
                        # print(qP.shape)
                        d = sess.run(mainQN.predict, feed_dict={
                            mainQN.scalarInput: [qP]})[0]
                    if d == 0:
                        possible_actions = list(env.actions(env._state))
                        possible_actions.remove(env.term_action)
                        if possible_actions == []:  # No more actions, need to terminate
                            action_taken = env.term_action
                            d == 1
                            break

                        # take action that maximises estimated VOC
                        action_taken = max(possible_actions, key=lambda x: myopic_voc(env, x))
                        r = myopic_voc(env, action_taken)  # RAC
                        env._step(action_taken)
                        q1, _ = env.term_reward(env._state)
                        q1 /= q_max
                        q1P = np.array([q1])
                    else:
                        r = 0
                        q1, _ = env.term_reward(env._state)
                        q1 /= q_max
                        q1P = np.array([q1])
                    total_steps += 1
                    # Save the experience to our episode buffer.
                    # print(q, d, r, q1)
                    episodeBuffer.add(np.reshape(np.array([qP, d, r, q1P]), [1, 4]))

                    if total_steps > pre_train_steps:
                        if e > float(params['endE']):
                            e -= stepDrop

                        if total_steps % (params['update_freq']) == 0:
                            # Get a random batch of experiences.
                            trainBatch = myBuffer.sample(params['batch_size'])
                            # print(trainBatch)
                            # print(np.vstack(trainBatch[:, 3]))
                            # Below we perform the Double-DQN update to the target Q-values

                            Q1 = sess.run(mainQN.predict, feed_dict={
                                mainQN.scalarInput: np.vstack(trainBatch[:, 3])})

                            Q2 = sess.run(targetQN.Qout, feed_dict={
                                targetQN.scalarInput: np.vstack(trainBatch[:, 3])})
                            end_multiplier = -(trainBatch[:, 3] - 1)
                            doubleQ = Q2[range(params['batch_size']), Q1]
                            targetQ = trainBatch[:, 2] + (float(params['gamma']) * doubleQ * end_multiplier)
                            # Update the network with our target values.
                            _ = sess.run(mainQN.updateModel, feed_dict={mainQN.scalarInput: np.vstack(
                                trainBatch[:, 0]), mainQN.targetQ: targetQ, mainQN.actions: trainBatch[:, 1]})

                            # Update the target network toward the primary network.
                            updateTarget(targetOps, sess)



                    rAll += r
                    # print(i, a, rAll, e)
                    q = q1
                    if d == 1:
                        break

                if i % 500 == 0:
                    mean_reward = np.mean(rList[-500:])
                    summary = tf.Summary()
                    summary.value.add(tag='Perf/Reward', simple_value=float(mean_reward))
                    summary_writer.add_summary(summary, i)
                    summary_writer.flush()
                    print(i, e, mean_reward)
                myBuffer.add(episodeBuffer.buffer)
                jList.append(j)
                rList.append(rAll)

                if i % 1000 == 0:
                    print("{} Done: Saving".format(i))
                    saver.save(sess, path + str(i) + '.ckpt')
