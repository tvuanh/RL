import random
from collections import deque

import numpy as np

import gym


class FrozenLakeQTable(object):

    actions_space = range(4)

    def __init__(self, gamma=0.99):
        self.gamma = gamma
        self.Qtable = np.zeros(
            (16, len(self.actions_space))
            ) # 16 state-rows, 4 action-columns
        self.counters = np.ones(self.Qtable.shape) # episodes counter per state

    def epsilon_greedy_action(self, state, epsilon):
        if random.random() < epsilon:
            return np.random.choice(self.actions_space)
        else:
            Qstate = self.Qtable[state, :]
            maxQstate = np.max(Qstate)
            possible_actions = [a for a in self.actions_space if Qstate[a] >= maxQstate]
            return np.random.choice(possible_actions)

    def train(self, state, action, reward, next_state, done):
        R = self.transform_reward(reward, done)
        maxNextQ = np.max(self.Qtable[next_state, :])
        update = R + self.gamma * maxNextQ - self.Qtable[state, action]
        self.Qtable[state, action] += update / self.counters[state, action]
        self.counters[state, action] += 1

    def transform_reward(self, reward, done):
        # penalise extra steps even if not terminal: in theory one
        # can make at most 4 steps from any place to any other place
        if reward >= 1:
            return 10.
        elif not done:
            return -0.1
        else:
            return -2.0


if __name__ == '__main__':

    # best so far: epsilon = 0.5, alpha = 0.5
    env = gym.make('FrozenLake-v0')
#     env = gym.wrappers.Monitor(env, '/tmp/frozenlake-v0', force=True)

    Qtable = FrozenLakeQTable(gamma=0.95)

    epsilon = 0.5
    target = 10.0

    performance = deque(maxlen=100)
    performance.append(0.)
    episode = 0
    while episode < 20000 and np.mean(performance) < 0.78:
        episode += 1
        state = env.reset()

        steps, rewards, done = 0, [], False
        while not done:
            steps += 1
            action = Qtable.epsilon_greedy_action(state, epsilon)
            # Execute the action and get feedback
            next_state, reward, done, _ = env.step(action)
            Qtable.train(state, action, reward, next_state, done)
            rewards.append(reward)
            state = next_state
        performance.append(np.sum(rewards))
        if epsilon > 0.01 and np.mean(rewards) > 0 and episode >= 1000:
            epsilon *= 1.- np.mean(rewards) / target
        print(
            "episode {} steps {} total reward {} performance {} epsilon {}".format(
                episode, steps, np.sum(rewards), np.mean(performance), epsilon)
            )
    print(Qtable.Qtable)
    print(Qtable.counters)
