import time
import gym
import numpy as np
import random
import matplotlib.pyplot as plt

# Create environment
env = gym.make('CartPole-v1', render_mode='human')  # 'human' mode for real-time rendering
env.reset()
total_reward = 0.0
total_steps = 0

# Print observation and action space details
print(env.observation_space.high)  # Highest values for observations
print(env.observation_space.low)
print(env.action_space.n)          # Number of actions

# Discretization parameters
discrete_os_size = [20] * len(env.observation_space.high)  # Number of bins per observation dimension
discrete_os_window_size = (env.observation_space.high - env.observation_space.low) / np.array(discrete_os_size)
print('Discrete window size:', discrete_os_window_size)

# Initialize Q-table
q_table = np.random.uniform(low=-5, high=5, size=(discrete_os_size + [env.action_space.n]))
print('Q-table shape:', q_table.shape)
print('Q-table:', q_table)

def discretize_state(state):
    """Discretize the continuous state."""
    state_indices = []
    for i, val in enumerate(state):
        index = int(np.digitize(val, bins=np.linspace(env.observation_space.low[i], env.observation_space.high[i], discrete_os_size[i] + 1)) - 1)
        state_indices.append(index)
    return tuple(state_indices)

while True:
    # Select a random action
    action = env.action_space.sample()
    
    # Step in the environment
    obs, reward, done, _, _ = env.step(action)
    
    # Discretize the state
    discretized_state = discretize_state(obs)
    
    # Update rewards and steps
    total_reward += reward
    total_steps += 1
    
    # Render the environment
    env.render()
    
    if done:
        break

print('Episode done in %d steps, total reward %.2f' % (total_steps, total_reward))
env.close()
