import gym
import numpy as np
import time
import matplotlib.pyplot as plt

def epsilon_greedy(q_values, epsilon):
    if np.random.rand() < epsilon:
        action = np.random.choice(len(q_values))
        return action
    else:
        return np.argmax(q_values)

def boltzmann_exploration(q_values, temperature=1.0):
    probabilities = np.exp(q_values / temperature) / np.sum(np.exp(q_values / temperature), axis=0)
    action = np.random.choice(len(q_values), p=probabilities)
    return action

def q_learning(env, learning_rate=0.1, discount_factor=0.9, exploration_prob=0.1, num_episodes=1000, temperature=1.0, exploration_strategy='epsilon-greedy'):
    state_space_size = env.observation_space.n
    action_space_size = env.action_space.n

    q_table = np.zeros((state_space_size, action_space_size))

    for episode in range(num_episodes):
        state = env.reset()[0]
        done = False

        while not done:
            if exploration_strategy == 'epsilon-greedy':
                action = epsilon_greedy(q_table[state], exploration_prob)
            elif exploration_strategy == 'boltzmann':
                action = boltzmann_exploration(q_table[state], temperature)
            else:
                raise ValueError("Invalid exploration strategy")

            next_state, reward, done, _, _ = env.step(action)

            # Q-value update
            q_table[state, action] = (1 - learning_rate) * q_table[state, action] + \
                                     learning_rate * (reward + discount_factor * np.max(q_table[next_state, :]))

            state = next_state

    return q_table

def evaluate_policy(q_table, env, num_episodes=10, epsilon=0.1, temperature=1.0, exploration_strategy='epsilon-greedy'):
    total_rewards = 0
    episode_rewards = []
    episode_steps = []
    for episode in range(num_episodes):
        state = env.reset()[0]
        done = False
        episode_reward = 0  # Initialize the episode reward for the current episode
        
        steps = 0
        while not done:
            if exploration_strategy == 'epsilon-greedy':
                action = epsilon_greedy(q_table[state], epsilon)
            elif exploration_strategy == 'boltzmann':
                action = boltzmann_exploration(q_table[state], temperature)
            else:
                raise ValueError("Invalid exploration strategy")

            steps += 1
            state, reward, done, _, _ = env.step(action)
            episode_reward += reward  # Accumulate the reward for the current episode

        print("Episode:", episode + 1, "Episode Reward:", episode_reward)
        total_rewards += episode_reward
        episode_rewards.append(episode_reward)
        episode_steps.append(steps)

    
    average_reward = total_rewards / num_episodes

    return average_reward, episode_rewards, episode_steps

# Example usage
env = gym.make('Taxi-v3')

learning_rate = 0.1
discount_factor = 0.9
num_episodes = 1000

test_episodes = 30

epsilon_value = 0.1
temperatures = [1.0, 5.0 , 10.0]
learning_rates = [0.1, 0.5, 0.9]




def main():
    for learning_rate in learning_rates:
        for temperature in temperatures:

            q_table_epsilon = q_learning(env, learning_rate=learning_rate, exploration_prob=epsilon_value, num_episodes=num_episodes, exploration_strategy='epsilon-greedy')
            avg_reward_epsilon, episode_rewards_greedy, episode_steps_greedy = evaluate_policy(q_table_epsilon, env, epsilon=epsilon_value, num_episodes=test_episodes, exploration_strategy='epsilon-greedy')
            print(f'Epsilon-Greedy - Learning Rate: {learning_rate}, Average Reward: {avg_reward_epsilon}') 

            q_table_boltzmann = q_learning(env, learning_rate=learning_rate, temperature=temperature, num_episodes=num_episodes, exploration_strategy='boltzmann')
            avg_reward_boltzmann, episode_rewards_boltz, episode_steps_bolts = evaluate_policy(q_table_boltzmann, env, temperature=temperature, num_episodes=test_episodes, exploration_strategy='boltzmann')
            print(f'Boltzmann - Temperature: {temperature}, Average Reward: {avg_reward_boltzmann}')
        
            episodes_arr = np.array(range(1, test_episodes + 1))

            plt.plot(episodes_arr, episode_rewards_greedy, label=f'Epsilon-Greedy' , color = 'blue')
            plt.plot(episodes_arr, episode_steps_greedy, label=f'Greedy steps', color = 'red')

            plt.plot(episodes_arr, episode_rewards_boltz, label=f'Boltzman', color = 'green')
            plt.plot(episodes_arr, episode_steps_bolts, label=f'Boltzman steps', color = 'orange')
            plt.title(f'Episode Rewards Over Episodes - lr_{learning_rate} & T_{temperature}')
            plt.xlabel('Episode Number')
            plt.ylabel('Episode Reward')
            plt.legend()
            plt.savefig(f'testowy Q-Learn lr_ {learning_rate} & T_ {temperature}.pdf')
            plt.close()

            print("PLOT DONE")


if __name__ == '__main__':
    main()
