import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import torch.nn.functional as F

class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )
        
        # Khởi tạo trọng số
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')

    def forward(self, x):
        return self.fc(x)

class PrioritizedReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.priorities = []
        self.pos = 0
        self.eps = 1e-6
        self.alpha = 0.6
        self.beta = 0.4
        self.beta_increment = 0.001

    def push(self, state, action, reward, next_state, done):
        max_priority = max([p.max() if isinstance(p, np.ndarray) else p for p in self.priorities]) if self.priorities else 1.0
        
        if len(self.buffer) < self.capacity:
            self.buffer.append((state, action, reward, next_state, done))
            self.priorities.append(max_priority)
        else:
            self.buffer[self.pos] = (state, action, reward, next_state, done)
            self.priorities[self.pos] = max_priority
        
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size):
        if len(self.buffer) == 0:
            return None
        
        priorities = np.array([p.max() if isinstance(p, np.ndarray) else p for p in self.priorities])
        probs = priorities ** self.alpha
        probs /= probs.sum()
        
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]
        
        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-self.beta)
        weights /= weights.max()
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        return samples, indices, weights

    def update_priorities(self, indices, priorities):
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = float(priority.max() if isinstance(priority, np.ndarray) else priority) + self.eps

class DQNAgent:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = DQN(4, 2).to(self.device)
        self.target = DQN(4, 2).to(self.device)
        self.target.load_state_dict(self.model.state_dict())
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.00025)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=100, gamma=0.95)
        
        self.memory = PrioritizedReplayBuffer(50000)
        self.batch_size = 128
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        self.learn_step = 0
        self.update_freq = 5
        self.reward_scale = 0.1

    def act(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, 1)
        
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            return torch.argmax(self.model(state)).item()

    def remember(self, state, action, reward, next_state, done):
        # Reward shaping
        shaped_reward = reward
        if not done:
            shaped_reward += self.reward_scale * (1 if action == 1 else 0)  # Khuyến khích flap
        
        self.memory.push(state, action, shaped_reward, next_state, done)

    def learn(self):
        if len(self.memory.buffer) < self.batch_size:
            return

        batch, indices, weights = self.memory.sample(self.batch_size)
        if batch is None:
            return

        states, actions, rewards, next_states, dones = zip(*batch)
        
        states = torch.from_numpy(np.stack(states)).float().to(self.device)
        actions = torch.tensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.tensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.from_numpy(np.stack(next_states)).float().to(self.device)
        dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(1).to(self.device)
        weights = torch.tensor(weights).float().to(self.device)

        # Double DQN
        with torch.no_grad():
            next_actions = self.model(next_states).max(1)[1].unsqueeze(1)
            next_q = self.target(next_states).gather(1, next_actions)
            target_q = rewards + self.gamma * next_q * (1 - dones)

        current_q = self.model(states).gather(1, actions)
        
        # Huber loss với importance sampling weights
        td_error = torch.abs(current_q - target_q)
        loss = (weights * F.smooth_l1_loss(current_q, target_q, reduction='none')).mean()

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()

        # Cập nhật priorities trong replay buffer
        self.memory.update_priorities(indices, td_error.detach().cpu().numpy())

        self.learn_step += 1
        if self.learn_step % self.update_freq == 0:
            self.target.load_state_dict(self.model.state_dict())
        
        if self.learn_step % 100 == 0:
            self.scheduler.step()
            
        if self.learn_step % 10 == 0:
            print(f"Loss: {loss.item():.4f}, LR: {self.scheduler.get_last_lr()[0]:.6f}")

    def save(self, path="dqn_model.pth"):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'epsilon': self.epsilon
        }, path)

    def load(self, path="dqn_model.pth"):
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.target.load_state_dict(self.model.state_dict())
