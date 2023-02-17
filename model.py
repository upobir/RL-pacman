import os

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class Qnet(nn.Module):
    def __init__(self, output_size):
        super().__init__()
        self.output_size = output_size
        self.conv1 = nn.Conv2d(4, 16, kernel_size=(8, 8), stride=(4, 4)) # 4x84x84 -> 16x20x20
        # relu
        self.conv2 = nn.Conv2d(16, 32, kernel_size=(4, 4), stride=(2, 2)) # 16x20x20 -> 32x9x9
        # relu
        # flatten
        self.fc1 = nn.Linear(32 * 9 * 9, 512)
        # relu
        self.fc2 = nn.Linear(512, 256)
        # relu
        self.fc3 = nn.Linear(256, output_size)


    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)

        x = self.conv2(x)
        x = F.relu(x)

        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = F.relu(x)

        x = self.fc2(x)
        x = F.relu(x)

        x = self.fc3(x)
        
        return x


    def save(self, path):
        torch.save(self.state_dict(), path)
        return


class QTrainer:
    def __init__(self, model, lr, gamma):
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.optimizer = optim.AdamW(model.parameters(), lr=lr, amsgrad=True)
        self.criterion = nn.SmoothL1Loss()


    def train_step(self, state, action, reward, next_state, terminated):
        state = torch.tensor(state, dtype=torch.float32)
        next_state = torch.tensor(next_state, dtype=torch.float32)
        reward = torch.tensor(reward, dtype=torch.float32)
        terminated = torch.tensor(terminated, dtype=torch.float32)

        if len(state.shape) == 3:
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            reward = torch.unsqueeze(reward, 0)
            terminated = torch.unsqueeze(terminated, 0)

            action_tensor = torch.zeros((1, self.model.output_size), dtype=torch.bool)
            action_tensor[0, action] = True
            action = action_tensor
        else:
            action_tensor = torch.zeros((len(action), self.model.output_size), dtype=torch.bool)
            action_tensor[range(len(action)), action] = True
            action = action_tensor


        batch_size = 128

        for i in range(0, len(state), batch_size):
            sz = min(batch_size, len(state) - i)

            state_batch = state[i:i+sz]
            next_state_batch = next_state[i:i+sz]
            reward_batch = reward[i:i+sz]
            terminated_batch = terminated[i:i+sz]
            action_batch = action[i:i+sz]

            prediction = self.model(state_batch)
            target = prediction.clone()
            target[action_batch] = reward_batch + self.gamma * torch.max(self.model(next_state_batch), dim=1)[0] * (1 - terminated_batch)

            self.optimizer.zero_grad()
            loss = self.criterion(target, prediction)
            loss.backward()
            nn.utils.clip_grad_value_(self.model.parameters(), 100)
            self.optimizer.step()
            

        with torch.no_grad():
            prediction = self.model(state)
            target = prediction.clone()
            target[action] = reward + self.gamma * torch.max(self.model(next_state), dim=1)[0] * (1 - terminated)
            loss_val = self.criterion(target, prediction)

        return loss_val