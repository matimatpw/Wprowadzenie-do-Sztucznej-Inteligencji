import gym
import numpy as np
import time

def q_learning(env, learning_rate=0.1, discount_factor=0.9, exploration_prob=0.1, num_episodes=1000):
    state_space_size = env.observation_space.n
    action_space_size = env.action_space.n

    q_table = np.zeros((state_space_size, action_space_size))

    for episode in range(num_episodes):
        state = env.reset()[0]
        done = False

        while not done:
            # Exploration-exploitation trade-off
            # if np.random.rand() < exploration_prob:
            #     action = env.action_space.sample()  # Explore
            # else:
            action = np.argmax(q_table[state])  # Exploit

            next_state, reward, done, _, _ = env.step(action)

            # Q-value update
            q_table[state, action] = (1 - learning_rate) * q_table[state, action] + \
                                     learning_rate * (reward + discount_factor * np.max(q_table[next_state, :]))

            state = next_state

        print("Episode:", episode + 1, "Total Reward:", np.sum(q_table))
    
    return q_table

def evaluate_policy(q_table, env, num_episodes=1, exploration_prob=0.1):
    total_rewards = 0
    for episode in range(num_episodes):
        state = env.reset()[0]
        done = False
        while not done:
            # if np.random.rand() < exploration_prob:
            #     action = env.action_space.sample()  # Explore
            # else:
            action = np.argmax(q_table[state])  # Exploit

            state, reward, done, _, _ = env.step(action)
            total_rewards += reward

        print("Episode:", episode + 1, "Total Reward:", total_rewards)

    average_reward = total_rewards / num_episodes
    return average_reward

# Example usage
env = gym.make('Taxi-v3')

# Experiment with different hyperparameters and exploration strategies
learning_rates = [0.1]
exploration_probs = [0.1]

for lr in learning_rates:
    for exploration_prob in exploration_probs:
        q_table = q_learning(env, learning_rate=lr, exploration_prob=exploration_prob)
        avg_reward = evaluate_policy(q_table, env)
        print(f'Learning Rate: {lr}, Exploration Probability: {exploration_prob}, Average Reward: {avg_reward}')
