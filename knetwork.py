import random
from collections import deque

import numpy as np

import gym
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.initializers import RandomNormal


class Controller(object):

    def __init__(self, n_input, n_output, gamma=0.99, batch_size=50):
        self.n_input = n_input
        self.n_output = n_output
        self.action_space = range(n_output)
        self.gamma = gamma
        self.batch_size = batch_size
        self.memory = deque(maxlen=10000)

        # action neural network
        self.action_model = Sequential()
        self.action_model.add(
            Dense(
                self.n_output, input_dim=self.n_input, activation="linear",
                kernel_initializer=RandomNormal(
                    mean=0.0, stddev=0.005, seed=None
                    ),
                use_bias=False
                )
            )
        self.action_model.compile(
            loss="mse", optimizer=Adam(lr=0.01, decay=0.01)
            )

    def preprocess_state(self, state):
        return np.identity(self.n_input)[state : state + 1]

    def memorize(self, state, action, reward, next_state, done):
        self.memory.append(
            (
                self.preprocess_state(state),
                action, reward,
                self.preprocess_state(next_state),
                done
                )
            )

    def epsilon_greedy_action(self, state, epsilon):
        if np.random.rand(1) < epsilon:
            return np.random.choice(self.action_space)
        else:
            Q = self.action_model.predict(self.preprocess_state(state))
            return np.argmax(Q)

    def replay(self):
        if len(self.memory) <= self.batch_size:
            minibatch = self.memory
        else:
            minibatch = random.sample(self.memory, k=self.batch_size)
        x_batch, y_batch = list(), list()
        for state, action, reward, next_state, _ in minibatch:
            y_target = self.action_model.predict(state)
            y_target[0, action] = reward + self.gamma * np.max(
                self.action_model.predict(next_state)
                )
            x_batch.append(state[0])
            y_batch.append(y_target[0])
        self.action_model.fit(
            np.array(x_batch), np.array(y_batch), batch_size=len(x_batch),
            verbose=False)


def play(episodes):
    env = gym.make("FrozenLake-v0")
    controller = Controller(
        n_input=env.observation_space.n, n_output=env.action_space.n)

    epsilon = 1.0
    target = 1.
    benchmark = 0.78
    scores = deque(maxlen=100)
    scores.append(0.0)
    maxscore = np.mean(scores)

    episode = 0
    while episode < episodes and np.mean(scores) < 0.78:
        episode += 1
        state = env.reset()
        done = False
        intra_episode_total_reward = 0
        steps = 0
        while not done:
            action = controller.epsilon_greedy_action(state, epsilon)
            next_state, reward, done, _ = env.step(action)
            controller.memorize(state, action, reward, next_state, done)
            state = next_state
            intra_episode_total_reward += reward
            steps += 1
        scores.append(intra_episode_total_reward)

        if epsilon > 0.01 and intra_episode_total_reward > 0 and np.mean(scores) > maxscore:
            maxscore = np.mean(scores)
            epsilon *= 1. - intra_episode_total_reward / target / steps
        # print(
        #     "episode {} steps {} score {} average score {} epsilon {} maxscore {}".format(
        #         episode, steps, intra_episode_total_reward, np.mean(scores), epsilon, maxscore
        #         )
        #     )
        controller.replay()
    return episode


if __name__ == "__main__":
    episodes = 10000
    nplays = 10
    results = np.array([play(episodes) for _ in range(nplays)])
    success = results < episodes
    print("Total number of successful plays is {}/{}".format(np.sum(success), nplays))
    print("Average number of episodes before success per play {}".format(np.mean(results[success])))
