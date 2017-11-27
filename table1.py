import random
from collections import deque

import numpy as np

import gym


class FrozenLakeQTable(object):

    actions_space = range(4)

    def __init__(self, gamma=1., minvisits=10, minvar=1.e-10):
        self.gamma = gamma
        self.minvisits = minvisits
        self.minvar = minvar
        self.Qmean = np.zeros(
            (16, len(self.actions_space))
            ) # 16 state-rows, 4 action-columns
        self.visits = np.ones(self.Qmean.shape) # episodes counter per state
        self.Sr2 = np.zeros(self.Qmean.shape)
        self.Qvar = np.ones(self.Qmean.shape) * np.inf

    def epsilon_greedy_action(self, state, epsilon):
        if random.random() < epsilon:
            return np.random.choice(self.actions_space)
        else:
            return self.action_from_values(self.Qmean[state, :])

    def optimal_action(self, state):
        visits = self.visits[state, :]
        if np.min(visits) < self.minvisits:
            return np.random.choice(self.actions_space)
        else:
            means, variances = self.Qmean[state, :], self.Qvar[state, :]
            sigmas = np.sqrt(np.divide(variances, visits))
            values = means + np.random.randn(len(self.actions_space)) * sigmas
            return self.action_from_values(values)

    def action_from_values(self, values):
        maxvalue = np.max(values)
        possible_actions = [a for a in self.actions_space if values[a] >= maxvalue]
        return np.random.choice(possible_actions)

    def train(self, state, action, reward, next_state, done):
        R = transform_reward(reward, done)
        future_reward = R + self.gamma * np.max(self.Qmean[next_state, :])
        # update mean, sum squared rewards and variance in exact order
        self.update_mean(state, action, future_reward)
        self.update_sum_squared_rewards(state, action, future_reward)
        self.update_variance(state, action)
        self.visits[state, action] += 1

    def update_mean(self, state, action, future_reward):
        visits = self.visits[state, action]
        Qm = self.Qmean[state, action]
        self.Qmean[state, action] += (future_reward - Qm) / visits

    def update_sum_squared_rewards(self, state, action, future_reward):
        self.Sr2[state, action] += future_reward * future_reward

    def update_variance(self, state, action):
        visits = self.visits[state, action]
        if visits > 1:
            sr2 = self.Sr2[state, action]
            Qm = self.Qmean[state, action]
            self.Qvar[state, action] = max(
                (sr2 - visits * Qm * Qm) / (visits - 1),
                self.minvar
            )


def transform_reward(reward, done):
    # penalise extra steps even if not terminal: in theory one
    # can make at most 4 steps from any place to any other place
    if reward >= 1:
        return 10.
    elif not done:
        return -0.1
    else:
        return -2.0


def play(episodes, verbose=False):
    env = gym.make('FrozenLake-v0')

    Qtable = FrozenLakeQTable(gamma=0.8, minvisits=20)

    performance = deque(maxlen=100)
    performance.append(0.)

    episode = 0
    while episode < episodes and np.mean(performance) < 0.78:
        episode += 1
        state = env.reset()

        steps, rewards, done = 0, [], False
        while not done:
            steps += 1
            action = Qtable.optimal_action(state)
            # Execute the action and get feedback
            next_state, reward, done, _ = env.step(action)
            Qtable.train(state, action, reward, next_state, done)
            rewards.append(reward)
            state = next_state
        performance.append(np.sum(rewards))
        if verbose:
            print(
                "episode {} steps {} total reward {} performance {}".format(
                    episode, steps, np.sum(rewards), np.mean(performance))
                )
    return episode


if __name__ == '__main__':
    episodes = 10000
    nplays = 50
    results = np.array([play(episodes) for _ in range(nplays)])
    success = results < episodes
    print("Total number of successful plays is {}/{}".format(np.sum(success), nplays))
    print("Average number of episodes before success per play {}".format(np.mean(results[success])))
