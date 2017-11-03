from collections import deque
import numpy as np
import gym


env = gym.make("FrozenLake-v0")

Q = np.zeros([env.observation_space.n, env.action_space.n])

alpha = 0.85
gamma = 0.95

episode, num_episodes = 0, 20000
performance = deque(maxlen=100)
performance.append(0)

while episode < num_episodes and np.mean(performance) < 0.78:
    episode +=1 

    state = env.reset()
    rAll = 0
    steps = 0
    done = False

    while steps < 99 and not done:
        steps += 1
        action = np.argmax(
            Q[state, :] + np.random.randn(1, env.action_space.n) * 1. / (episode + 1)
            )
        next_state, reward, done, _ = env.step(action)
        Q[state, action] = Q[state, action] + alpha * (
            reward + gamma * np.max(Q[next_state, :]) - Q[state, action]
            )
        rAll += reward
        state = next_state

    performance.append(rAll)

    print(
        "episode {} steps {} total reward {} average performance {}".format(
            episode, steps, rAll, np.mean(performance)
            )
        )

print(Q)
