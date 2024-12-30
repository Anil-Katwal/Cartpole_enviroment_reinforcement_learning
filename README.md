# CartPole Balancing using Reinforcement Learning

This project demonstrates solving the CartPole balancing problem using Reinforcement Learning (RL) techniques. The CartPole environment is provided by OpenAI's Gym library and is a popular benchmark problem in RL.

## Overview

The CartPole problem involves a pole attached to a cart, where the goal is to keep the pole balanced by applying forces to move the cart left or right. The environment provides a reward for each timestep the pole remains balanced.

### Environment Details
- **Environment Name**: `CartPole-v1`
- **State Space**: A 4-dimensional vector representing the position and velocity of the cart, and the angle and angular velocity of the pole.
- **Action Space**: Two discrete actions: move left or move right.
- **Reward**: +1 for every timestep the pole remains balanced.
- **Episode End**:
  - The pole falls past 15 degrees.
  - The cart moves out of bounds.
  - The episode reaches 500 timesteps.

## Approach

The solution is implemented using the following reinforcement learning techniques:
1. **Q-Learning**: A tabular method where the Q-table stores the value of state-action pairs.
2. **Deep Q-Networks (Optional)**: An advanced approach that uses a neural network to approximate the Q-function.

### Steps to Solve
1. **Environment Setup**: Use `gym` to create the `CartPole-v1` environment.
2. **Q-Table Initialization**: Initialize a table to store Q-values for each state-action pair.
3. **Training**: Implement the Q-learning algorithm to iteratively update Q-values.
4. **Evaluation**: Test the trained agent on new episodes to measure performance.

## Installation

### Prerequisites
Ensure you have Python 3.8 or higher installed on your system.

### Required Libraries
Install the following dependencies:

```bash
pip install -r requirements.txt
```

### `requirements.txt`
```
gym==0.26.2
matplotlib==3.7.1
numpy==1.25.0
```

## Running the Code

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/cartpole-balancing.git
   cd cartpole-balancing
   ```

2. Train the agent:
   ```bash
   python train_cartpole.py
   ```

3. Visualize the results:
   ```bash
   python evaluate_cartpole.py
   ```

4. View plots and performance metrics in the generated output files.

## Results

The agent achieves a balance score of 500 (maximum reward) after sufficient training episodes. Performance can vary based on hyperparameters such as:
- Learning rate
- Discount factor
- Exploration rate (epsilon)

## Visualization
- **Training Rewards**: Plots showing the cumulative reward per episode.
- **Evaluation Episodes**: Visualizations of the cart and pole in action.

## Contributing

Feel free to contribute to this project by submitting issues or pull requests. Suggestions for improvement or additional features are welcome!

## License

This project is licensed under the MIT License. See the LICENSE file for details.



