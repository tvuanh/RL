import random
from collections import deque

import numpy as np

import gym
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.initializers import RandomNormal

# result location
result_location = "/tmp/cartpole-keras-1"

# set random seed
np.random.seed(100)


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
        size = min(len(self.memory), self.batch_size)
        minibatch = random.sample(self.memory, k=size)
        x_batch, y_batch = list(), list()
        for state, action, reward, next_state, done in minibatch:
            y_target = self.action_model.predict([state])
            y_target[0, action] = reward + done * self.gamma * np.max(
                self.action_model.predict(next_state)
                )
            x_batch.append(state[0])
            y_batch.append(y_target[0])
        self.action_model.fit(
            np.array(x_batch), np.array(y_batch), batch_size=len(x_batch),
            verbose=False)


if __name__ == "__main__":
    env = gym.make("FrozenLake-v0")
    controller = Controller(
        n_input=env.observation_space.n, n_output=env.action_space.n)

    n_episodes = 2000
    epsilon = 0.2
    target = 1.
    benchmark = 0.78
    scores = deque(maxlen=100)

    for episode in range(n_episodes):
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
        if epsilon > 0.01 and intra_episode_total_reward > 0:
            epsilon *= 1. - intra_episode_total_reward / target / steps
        print(
            "episode {} steps {} score {} average score {} epsilon {}".format(
                episode, steps, intra_episode_total_reward, np.mean(scores), epsilon
                )
            )
        if np.mean(scores) >= benchmark:
            print("Learned how to play after {} episodes".format(episode))
        controller.replay()
