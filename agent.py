import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
import json
import sys
import os
from collections import deque

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from schemas import Action, ActionType

class DuelingQNetwork(nn.Module):
    """A Dueling Double DQN for complex flow classification."""
    def __init__(self, input_dim, output_dim):
        super(DuelingQNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 128)

        # Value stream
        self.v1 = nn.Linear(128, 64)
        self.v2 = nn.Linear(64, 1)

        # Advantage stream
        self.a1 = nn.Linear(128, 64)
        self.a2 = nn.Linear(64, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))

        v = torch.relu(self.v1(x))
        v = self.v2(v)

        a = torch.relu(self.a1(x))
        a = self.a2(a)

        # Q(s,a) = V(s) + (A(s,a) - mean(A))
        q = v + (a - a.mean(dim=1, keepdim=True))
        return q

class PrioritizedReplayBuffer:
    """A simplified Prioritized Experience Replay (PER) buffer using proportional sampling."""
    def __init__(self, capacity=100000, alpha=0.6, beta=0.4):
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.buffer = []
        self.pos = 0
        self.priorities = np.zeros((capacity,), dtype=np.float32)

    def push(self, state, action_idx, reward, next_state, done):
        max_prio = self.priorities.max() if self.buffer else 1.0
        
        if len(self.buffer) < self.capacity:
            self.buffer.append((state, action_idx, reward, next_state, done))
        else:
            self.buffer[self.pos] = (state, action_idx, reward, next_state, done)
            
        self.priorities[self.pos] = max_prio
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size):
        if len(self.buffer) == 0:
            return None
            
        prios = self.priorities[:len(self.buffer)]
        probs = prios ** self.alpha
        probs /= probs.sum()
        
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        batch = [self.buffer[idx] for idx in indices]
        
        states, actions, rewards, next_states, dones = zip(*batch)
        
        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-self.beta)
        weights /= weights.max()
        
        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32),
            indices,
            np.array(weights, dtype=np.float32)
        )

    def update_priorities(self, batch_indices, batch_priorities):
        for idx, prio in zip(batch_indices, batch_priorities):
            self.priorities[idx] = max(prio, 1e-5)

    def __len__(self):
        return len(self.buffer)

class DQNSOCAgent:
    def __init__(self, action_space, state_dim=8):
        self.action_space = action_space
        self.action_dim = len(action_space)
        self.state_dim = state_dim

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.q_network = DuelingQNetwork(state_dim, self.action_dim).to(self.device)
        self.target_network = DuelingQNetwork(state_dim, self.action_dim).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=1e-3, weight_decay=1e-5)
        self.memory = PrioritizedReplayBuffer(capacity=100000)

        self.batch_size = 128
        self.gamma = 0.99 
        self.eps = 1.0
        self.eps_min = 0.01

        self.update_target_steps = 1000
        self.steps_done = 0

    def get_state_vector(self, obs):
        logs_list = json.loads(obs.logs) if obs.logs else []
        log = logs_list[0] if logs_list else {}
        
        # Features are already StandardScaled via DataLoader/train.py, just map directly.
        state = np.array([
            log.get("Destination_Port", 0),
            log.get("Flow_Duration", 0), 
            log.get("Total_Fwd_Packets", 0),
            log.get("Total_Bwd_Packets", 0),
            log.get("Fwd_Packet_Length_Max", 0),
            log.get("Bwd_Packet_Length_Max", 0),
            log.get("Flow_Bytes_s", 0),
            log.get("Flow_Packets_s", 0)
        ], dtype=np.float32)
        
        # Replace remaining Infs or NaNs robustly just in case
        np.nan_to_num(state, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        return state

    def select_action(self, state, eval_mode=False):
        current_eps = 0.0 if eval_mode else self.eps
        if random.random() < current_eps:
            action_idx = random.randint(0, self.action_dim - 1)
        else:
            with torch.no_grad():
                self.q_network.eval()
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values = self.q_network(state_tensor)
                
                max_q, max_action_tensor = q_values.max(dim=1)
                max_q_val = max_q.item()
                action_idx = max_action_tensor.item()
                
                # Threshold tuning for inference mode to explicitly reduce false alarms
                if eval_mode and max_q_val < 0.2:  # Threshold can be continuously tuned
                    if "BENIGN" in self.action_space:
                        action_idx = self.action_space.index("BENIGN")
                        
                self.q_network.train()
        return action_idx

    def optimize_model(self):
        if len(self.memory) < self.batch_size:
            return 0.0

        sample_data = self.memory.sample(self.batch_size)
        if sample_data is None: return 0.0
        
        states, actions, rewards, next_states, dones, indices, weights = sample_data

        states_t = torch.FloatTensor(states).to(self.device)
        actions_t = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards_t = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states_t = torch.FloatTensor(next_states).to(self.device)
        dones_t = torch.FloatTensor(dones).unsqueeze(1).to(self.device)
        weights_t = torch.FloatTensor(weights).unsqueeze(1).to(self.device)

        q_values = self.q_network(states_t).gather(1, actions_t)

        with torch.no_grad():
            # Double DQN formula
            next_action_idx = self.q_network(next_states_t).argmax(1).unsqueeze(1)
            next_q_values = self.target_network(next_states_t).gather(1, next_action_idx)
            target_q_values = rewards_t + (self.gamma * next_q_values * (1 - dones_t))

        # PER Update Priorities
        td_errors = (target_q_values - q_values).squeeze().detach().cpu().numpy()
        self.memory.update_priorities(indices, np.abs(td_errors) + 1e-6)

        # Weighted MSE Loss
        loss = (weights_t * nn.MSELoss(reduction='none')(q_values, target_q_values)).mean()

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()

        return loss.item()

    def run_episode(self, env, eval_mode=False):
        obs = env.reset()
        done = False
        total_r = 0.0
        
        state = self.get_state_vector(obs)

        for _ in range(env.max_steps):
            action_idx = self.select_action(state, eval_mode=eval_mode)
            ans = self.action_space[action_idx]
            
            action = Action(action_type=ActionType.submit, final_answer=ans)
            next_obs, reward, done, info = env.step(action)
            r_val = float(reward.value)
            
            next_state = self.get_state_vector(next_obs)
            
            if not eval_mode:
                self.memory.push(state, action_idx, r_val, next_state, done)
                self.optimize_model()
                self.steps_done += 1
                if self.steps_done % self.update_target_steps == 0:
                    self.target_network.load_state_dict(self.q_network.state_dict())
            
            state = next_state
            total_r += r_val
            if done: break
            
        return {'score': total_r, 'epsilon': self.eps}