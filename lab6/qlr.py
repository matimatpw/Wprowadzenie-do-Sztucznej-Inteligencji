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

def boltzmann_exploration(q_values, temperature):
    probabilities = np.exp(q_values / temperature) / np.sum(np.exp(q_values / temperature), axis=0)
    action = np.random.choice(len(q_values), p=probabilities)
    return action

def q_learning(env, learning_rate=0.1, importance_future=0.9, exploration_prob=0.1, num_episodes=5000, temperature=1.0, exploration_strategy='epsilon-greedy', max_steps=50):
    state_space_size = env.observation_space.n
    action_space_size = env.action_space.n

    q_table = np.zeros((state_space_size, action_space_size))
    

    counter = 0
    counter_arr = []
    counter_arr_rewards = []
    counter_rewards = 0

    for episode in range(num_episodes):


        if episode % 100 == 0:
            counter += 1
            counter_arr.append(counter)
            counter_arr_rewards.append(counter_rewards / 100)
            counter_rewards = 0

        state = env.reset()[0]
        done = False
        steps = 0

        episode_reward = 0
        while not done and steps < max_steps:
            steps += 1
            if exploration_strategy == 'epsilon-greedy':
                action = epsilon_greedy(q_table[state], exploration_prob)
            elif exploration_strategy == 'boltzmann':
                action = boltzmann_exploration(q_table[state], temperature)
            else:
                raise ValueError("Invalid exploration strategy")

            next_state, reward, done, _, _ = env.step(action)

            # Q-value update
            q_table[state, action] = (1 - learning_rate) * q_table[state, action] + learning_rate * (reward + importance_future * np.max(q_table[next_state, :]))
            state = next_state

            episode_reward += reward 
            counter_rewards += episode_reward


    return q_table, counter_arr, counter_arr_rewards

def evaluate_policy(q_table, env, num_episodes=100, epsilon=0.1, temperature=1.0, exploration_strategy='epsilon-greedy', max_steps=50):
    total_rewards = 0
    episode_rewards = []
    episode_steps = []

    counter = 0
    counter_arr = []
    counter_arr_rewards = []
    counter_rewards = 0
    for episode in range(num_episodes):

        if episode % 100 == 0 and episode != 0:
            counter += 1
            counter_arr.append(counter)
            counter_arr_rewards.append(counter_rewards / 100)
            counter_rewards = 0

        state = env.reset()[0]
        done = False
        episode_reward = 0 
        
        steps = 0
        while not done and steps < max_steps:
            if exploration_strategy == 'epsilon-greedy':
                action = epsilon_greedy(q_table[state], epsilon)
            elif exploration_strategy == 'boltzmann':
                action = boltzmann_exploration(q_table[state], temperature)
            else:
                raise ValueError("Invalid exploration strategy")

            steps += 1
            state, reward, done, _, _ = env.step(action)
            episode_reward += reward 

        print("Episode:", episode + 1, "Episode Reward:", episode_reward, "Steps:", steps)
        total_rewards += episode_reward
        counter_rewards += episode_reward

        episode_rewards.append(episode_reward)
        episode_steps.append(steps)

    
    average_reward = total_rewards / num_episodes

    return average_reward, episode_rewards, episode_steps, counter_arr, counter_arr_rewards

env = gym.make('Taxi-v3')

importance_future = 0.9
num_episodes = 2000

evaluate_episodes = 1000
steps_limit = 30


epsilon_value = 0.1
temperatures = [1.0, 5.0 , 10.0]
learning_rates = [0.1, 0.5, 0.9]




def main():
    for learning_rate in learning_rates:
        for temperature in temperatures:

            q_table_epsilon, c_x_eps_learn, c_y_eps_learn = q_learning(env, learning_rate=learning_rate, exploration_prob=epsilon_value, num_episodes=num_episodes, exploration_strategy='epsilon-greedy')
            avg_reward_epsilon, episode_rewards_greedy, episode_steps_greedy, c_x_eps, c_y_eps = evaluate_policy(q_table_epsilon, env, epsilon=epsilon_value, num_episodes=evaluate_episodes, exploration_strategy='epsilon-greedy', max_steps=steps_limit)
            print(f'Epsilon-Greedy - Learning Rate: {learning_rate}, Average Reward: {avg_reward_epsilon}') 

            q_table_boltzmann, c_x_bolt_learn, c_y_bolt_learn = q_learning(env, learning_rate=learning_rate, importance_future=importance_future, temperature=temperature, num_episodes=num_episodes, exploration_strategy='boltzmann')
            avg_reward_boltzmann, episode_rewards_boltz, episode_steps_bolts, c_x_bolt, c_y_bolt = evaluate_policy(q_table_boltzmann, env, temperature=temperature, num_episodes=evaluate_episodes, exploration_strategy='boltzmann', max_steps=steps_limit)
            print(f'Boltzmann - Temperature: {temperature}, Average Reward: {avg_reward_boltzmann}')
        
            episodes_arr = np.array(range(1, evaluate_episodes + 1))

            # plt.plot(episodes_arr, episode_rewards_greedy, label=f'Epsilon-Greedy' , color = 'blue')
            # plt.plot(episodes_arr, episode_steps_greedy, label=f'Greedy steps', color = 'red',linestyle='-', alpha=0.8)
            # plt.plot(episodes_arr, episode_rewards_boltz, label=f'Boltzman', color = 'green')
            # plt.plot(episodes_arr, episode_steps_bolts, label=f'Boltzman steps', color = 'orange', linestyle='-', alpha=0.8)

            plt.plot(c_x_eps, c_y_eps, label=f'Epsilon-Greedy avg', color = 'red',linestyle='-', alpha=0.8)
            plt.plot(c_x_bolt, c_y_bolt, label=f'Boltzman avg', color = 'orange',linestyle='-', alpha=0.8)
            
            plt.title(f'Episode Rewards Over Episodes - lr_{learning_rate} & T_{temperature}')
            plt.xlabel('Episode Number')
            plt.ylabel('Episode Reward')
            plt.legend()
            plt.savefig(f'Rewards After lr_ {learning_rate} & T_ {temperature}.pdf')
            plt.close()

            print("PLOT DONE")


if __name__ == '__main__':
    main()
