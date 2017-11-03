import random
from collections import deque

import numpy as np

import gym
import tensorflow as tf

np.random.seed(42)


def get_state(state):
    return np.identity(16)[state: state + 1]


def Qaction_predict(X):
    n_input = int(X.shape[1])
    with tf.variable_scope("predict"):
        weights = tf.Variable(
            tf.random_normal([n_input, 4], stddev=0.005), name="weights")
        Qaction = tf.matmul(X, weights) # shape [1, 4]
    return Qaction


def greedy_action(Qaction):
    return tf.argmax(Qaction, axis=1)


def Qaction_train(Qaction, nextQ):
    loss = tf.reduce_sum(tf.square(nextQ - Qaction))
    trainer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
    updateModel = trainer.minimize(loss)
    return updateModel


env = gym.make("FrozenLake-v0")
gamma = 0.99
# epsilon = 0.1
epsilon = 0.5
num_episodes = 10000
performance = deque(maxlen=100)
performance.append(0.0)


with tf.Session() as sess:
    # establish the feed-forward part of the network used to choose actions
    X = tf.placeholder(shape=[None, 16], dtype=tf.float32)
    Qout = Qaction_predict(X)
    predict = greedy_action(Qout)
    nextQ = tf.placeholder(shape=[None, 4], dtype=tf.float32)
    updateModel = Qaction_train(Qout, nextQ)

    init = tf.global_variables_initializer()
    sess.run(init)

    episode = 0
    while episode < num_episodes and np.mean(performance) < 0.78:
        episode += 1
        state = env.reset()

        steps = 0
        rewardAll = 0
        # the Q-network
        while steps < 100:
            steps += 1
            # epsilon-greedy action
            actions, allQ = sess.run(
                [predict, Qout], feed_dict={
                    X: get_state(state)
                    }
                )
            if np.random.rand(1) < epsilon:
                actions[0] = env.action_space.sample()
            # get the new state
            next_state, reward, done, _ = env.step(actions[0])
            # obtain the next Q values by feeding the state through the network
            predNextQ = sess.run(
                Qout, feed_dict={
                    X: get_state(next_state)
                    }
                )
            targetQ = allQ
            targetQ[0, actions[0]] = reward + gamma * np.max(predNextQ)
            # train the networkd using the target and predicted Q values
            sess.run(
                updateModel, feed_dict={
                    X: get_state(state),
                    nextQ: targetQ
                    }
                )
            rewardAll += reward
            state = next_state
            if done and epsilon > 0.01 and rewardAll > 0:
                # reduce the chance of random action as we train the model
                epsilon = 1. / (episode / 50 + 10)
                break
        performance.append(rewardAll)
        print("episode {} steps {} rewards {} epsilon {}".format(episode, steps, np.mean(performance), np.around(epsilon, 4)))
