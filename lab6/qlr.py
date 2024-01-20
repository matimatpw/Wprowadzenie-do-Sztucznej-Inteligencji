import gym
import numpy as np
import time
import matplotlib.pyplot as plt

class data:
    def __init__(self, importance_future, num_episodes, evaluate_episodes, steps_limit, epsilon_val, temperatures, learning_rates, exploration_strategy='epsilon-greedy') -> None:
        self.env = gym.make('Taxi-v3')
        self.importance_future = importance_future
        self.num_episodes = num_episodes
        self.evaluate_episodes = evaluate_episodes
        self.steps_limit = steps_limit
        self.epsilon_value = epsilon_val
        self.temperatures = temperatures
        self.learning_rates = learning_rates
        self.exploration_strategy = exploration_strategy

    def set_temperature(self, temperature):
        self.temperature = temperature

    def set_learning_rate(self, learning_rate):
        self.learning_rate = learning_rate

def epsilon_greedy(q_values, epsilon):
    if np.random.rand() < epsilon:
        action = np.random.choice(len(q_values))
        return action
    else:
        return np.argmax(q_values)

def boltzmann_exploration(q_values, temperature):
    probabilities = np.exp(q_values / temperature) / np.sum(np.exp(q_values / temperature), axis=0)
    action = np.random.choice(len(q_values), p=probabilities)
    return action

def q_learning(data:data):
    state_space_size = data.env.observation_space.n
    action_space_size = data.env.action_space.n

    q_table = np.zeros((state_space_size, action_space_size))

    for episode in range(data.num_episodes):

        state = data.env.reset()[0]
        done = False
        steps = 0

        episode_reward = 0
        while not done and steps < data.steps_limit:
            steps += 1
            if data.exploration_strategy == 'epsilon-greedy':
                action = epsilon_greedy(q_table[state], data.epsilon_value)
            elif data.exploration_strategy == 'boltzmann':
                action = boltzmann_exploration(q_table[state], data.temperature)
            else:
                raise ValueError("Invalid exploration strategy")

            next_state, reward, done, _, _ = data.env.step(action)

            q_table[state, action] = (1 - data.learning_rate) * q_table[state, action] + data.learning_rate * (reward + data.importance_future * np.max(q_table[next_state, :]))
            state = next_state

            episode_reward += reward 

    return q_table

def evaluate_policy(data:data, q_table):
    total_rewards = 0
    episode_rewards = []
    episode_steps = []

    counter = 0
    counter_arr = []
    counter_arr_rewards = []
    counter_rewards = 0
    for episode in range(data.evaluate_episodes):

        if episode % 100 == 0 and episode != 0:
            counter += 1
            counter_arr.append(counter)
            counter_arr_rewards.append(counter_rewards / 100)
            counter_rewards = 0

        state = data.env.reset()[0]
        done = False
        episode_reward = 0 
        
        steps = 0
        while not done and steps < data.steps_limit:
            if data.exploration_strategy == 'epsilon-greedy':
                action = epsilon_greedy(q_table[state], data.epsilon_value)
            elif data.exploration_strategy == 'boltzmann':
                action = boltzmann_exploration(q_table[state], data.temperature)
            else:
                raise ValueError("Invalid exploration strategy")

            steps += 1
            state, reward, done, _, _ = data.env.step(action)
            episode_reward += reward 

        total_rewards += episode_reward
        counter_rewards += episode_reward

        episode_rewards.append(episode_reward)
        episode_steps.append(steps)

    average_reward = total_rewards / data.evaluate_episodes

    return average_reward, episode_rewards, episode_steps, counter_arr, counter_arr_rewards

# env = gym.make('Taxi-v3')

# importance_future = 0.9
# num_episodes = 4000

# evaluate_episodes = 1000
# steps_limit = 50


# epsilon_value = 0.1
# temperatures = [1.0, 5.0 , 10.0]
# learning_rates = [0.1, 0.5, 0.9]




def main():
    input_data = data(importance_future=0.9, num_episodes=4000, evaluate_episodes=1000, steps_limit=50, epsilon_val=0.1, temperatures=[1.0, 5.0 , 10.0], learning_rates= [0.1, 0.5, 0.9])

    for learning_rate in input_data.learning_rates:
        for temperature in input_data.temperatures:

            input_data.set_temperature(temperature)
            input_data.set_learning_rate(learning_rate)

            q_table_epsilon = q_learning(input_data)
            avg_reward_epsilon, episode_rewards_greedy, episode_steps_greedy, c_x_eps, c_y_eps = evaluate_policy(input_data, q_table_epsilon)
            print(f'Epsilon-Greedy - Learning Rate: {learning_rate}, Average Reward: {avg_reward_epsilon}')

            q_table_boltzmann = q_learning(input_data)
            avg_reward_boltzmann, episode_rewards_boltz, episode_steps_bolts, c_x_bolt, c_y_bolt = evaluate_policy(input_data, q_table_boltzmann)
            print(f'Boltzmann - Temperature: {temperature}, Average Reward: {avg_reward_boltzmann}')

            plt.plot(c_x_eps, c_y_eps, label=f'Epsilon-Greedy avg', color = 'red',linestyle='-', alpha=0.8)
            plt.plot(c_x_bolt, c_y_bolt, label=f'Boltzman avg', color = 'orange',linestyle='-', alpha=0.8)
            
            plt.title(f'Episode Rewards Over Episodes - lr_{learning_rate} & T_{temperature}')
            plt.xlabel('Episode Number')
            plt.ylabel('Episode Reward')
            plt.legend()
            plt.savefig(f'Qlr - lr_ {learning_rate} & T_ {temperature}.pdf')
            plt.close()

            print("PLOT DONE")


if __name__ == '__main__':
    main()
