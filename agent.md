# Explanation of `agent.py`

This file implements the Dueling Double Deep Q-Network (D3QN) agent along with a Prioritized Experience Replay buffer.

*   `import torch`, `import torch.nn as nn`, `import torch.optim as optim`: Imports PyTorch and its neural network and optimization modules.
*   `import random`, `import numpy as np`, `import json`, `import sys`, `import os`: Imports standard Python libraries for random number generation, array operations, JSON parsing, and system/OS utilities.
*   `from collections import deque`: Used to import `deque` for potential replay buffer implementations.
*   `sys.path.append(...)`: Adds the parent directory to the system path to allow importing from the `schemas` module.
*   `from schemas import Action, ActionType`: Imports the `Action` and `ActionType` classes from the parent directory's `schemas.py` file.

### `DuelingQNetwork(nn.Module)`
*   `class DuelingQNetwork(nn.Module):`: Defines the network architecture inheriting from PyTorch's `nn.Module`.
*   `def __init__(self, input_dim, output_dim):`: Initializes the network with input and output dimensions.
*   `super(DuelingQNetwork, self).__init__()`: Calls the parent class constructor.
*   `self.fc1 = nn.Linear(input_dim, 256)`: The first fully connected layer.
*   `self.fc2 = nn.Linear(256, 128)`: The second fully connected layer.
*   `self.v1 = nn.Linear(128, 64)` & `self.v2 = nn.Linear(64, 1)`: Value stream layers calculating the state value $V(s)$.
*   `self.a1 = nn.Linear(128, 64)` & `self.a2 = nn.Linear(64, output_dim)`: Advantage stream layers calculating the advantage $A(s, a)$ for each action.
*   `def forward(self, x):`: The forward pass of the network.
*   The lines inside `forward` apply ReLU activations to the intermediate layers and then separately compute the value `v` and advantage `a`.
*   `q = v + (a - a.mean(dim=1, keepdim=True))`: Combines the value and advantage streams to output the final $Q(s, a)$ values, subtracting the mean of the advantages for identifiability.

### `PrioritizedReplayBuffer`
*   `class PrioritizedReplayBuffer:`: A class that implements prioritized experience replay using proportional sampling.
*   `def __init__(...):` Initializes the buffer with capacity and hyperparameters `alpha` and `beta`.
*   `def push(...):` Adds an experience tuple to the buffer. If the buffer isn't full, it appends; otherwise, it overwrites based on `pos`. New experiences are given the maximum priority to guarantee they are sampled at least once.
*   `def sample(self, batch_size):` Samples a batch of experiences.
    *   Calculates probabilities based on priorities raised to the power of `alpha`.
    *   Selects indices based on these probabilities.
    *   Retrieves the batch elements and computes importance sampling weights based on `beta` to correct for bias introduced by non-uniform sampling.
*   `def update_priorities(...):` Updates the priorities of sampled transitions using the latest TD errors.
*   `def __len__(self):` Returns the current size of the buffer.

### `DQNSOCAgent`
*   `class DQNSOCAgent:`: The main Reinforcement Learning agent.
*   `def __init__(...):` Sets up the agent's parameters, PyTorch device, networks (both policy `q_network` and `target_network`), Adam optimizer, and prioritized replay memory. It also defines hyperparameters like `gamma` (discount factor), `batch_size`, and exploration `eps` (epsilon).
*   `def get_state_vector(self, obs):` Extracts a fixed-length feature vector from the observation logs using `json.loads` and mapping expected keys to a numpy array, replacing missing values, NaNs, or INFs with zeros.
*   `def select_action(self, state, eval_mode=False):` Chooses an action via $\epsilon$-greedy policy. With probability $\epsilon$, it explores randomly; otherwise, it exploits the learned $Q$-values. In `eval_mode`, it has an extra heuristic to reduce false alarms by defaulting to "BENIGN" if the max $Q$-value is too low.
*   `def optimize_model(self):` Performs one step of gradient descent.
    *   Samples a batch from the replay memory.
    *   Computes current $Q$-values with the policy network.
    *   Uses Double DQN logic: selects the best next action using the policy network, but evaluates its value using the target network.
    *   Calculates TD errors, updates memory priorities, computes the MSE loss weighted by importance sampling weights, and backpropagates to update the network parameters.
*   `def run_episode(self, env, eval_mode=False):` Steps through one full episode using the environment until `done` is True. Automatically manages taking actions, storing transitions, and running `optimize_model()`. Returns the total reward.
