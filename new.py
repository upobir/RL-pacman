import torch
import gymnasium as gym
import cv2
import numpy as np
from math import log
import torch.nn as nn
from torch.nn import functional as F
import random
from collections import namedtuple, deque
import matplotlib.pyplot as plt
import sys

WALL_COLOR = [228, 111, 111]
tonumpy = lambda color: np.array([[color]], dtype=np.uint8)
togray = lambda color: cv2.cvtColor(tonumpy(color), cv2.COLOR_RGB2GRAY)[0][0]
WALL_COLOR_GRAY = togray(WALL_COLOR)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DQN(nn.Module):

    CONV_N_MAPS = [4, 32, 32]
    CONV_KERNEL_SIZES = [(4, 4), (2, 2)]
    CONV_STRIDES = [2, 2]
    CONV_PADDINGS = [2, 0]
    N_HIDDEN_IN = 32 * 11 * 10
    N_HIDDEN = [512, 128]

    def __init__(self, outputs):
        super(DQN, self).__init__()
        conv2d = lambda i: nn.Conv2d(
            self.CONV_N_MAPS[i],
            self.CONV_N_MAPS[i + 1],
            kernel_size=self.CONV_KERNEL_SIZES[i],
            stride=self.CONV_STRIDES[i],
            padding=self.CONV_PADDINGS[i],
        )
        self.conv1 = conv2d(0)
        self.bn1 = nn.BatchNorm2d(self.CONV_N_MAPS[1])
        self.conv2 = conv2d(1)
        self.bn2 = nn.BatchNorm2d(self.CONV_N_MAPS[2])

        self.hidden1 = nn.Linear(self.N_HIDDEN_IN, self.N_HIDDEN[0])
        self.hidden2 = nn.Linear(self.N_HIDDEN[0], self.N_HIDDEN[1])
        self.output = nn.Linear(self.N_HIDDEN[1], outputs)

    def forward(self, x):
        x = x.to(device)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.hidden1(x))
        x = F.relu(self.hidden2(x))
        return self.output(x)

def optimize_model(policy_DQN, target_DQN, memory, optimizer, learn_counter, device):
    if len(memory) < 128:
        return learn_counter
    learn_counter += 1
    states, actions, rewards, next_states, dones = memory.sample()

    predicted_targets = policy_DQN(states).gather(1, actions)

    target_values = target_DQN(next_states).detach().max(1)[0]
    labels = rewards + 0.99 * (1 - dones.squeeze(1)) * target_values

    criterion = torch.nn.SmoothL1Loss()
    loss = criterion(predicted_targets, labels.detach().unsqueeze(1)).to(device)
    # print(loss)

    optimizer.zero_grad()
    loss.backward()
    for param in policy_DQN.parameters():
       param.grad.data.clamp_(-1, 1)
    optimizer.step()


    return learn_counter

Experience = namedtuple("Experience", ("state", "action", "reward", "next_state", "done"))
REVERSED = {0: 1, 1: 0, 2: 3, 3: 2}
EPS_MAX = 1.0
EPS_MIN = 0.1
EPS_DECAY = 1_000_000

class DecisionMaker:
    def __init__(self, steps_done, policy_DQN):
        self.steps_done = steps_done
        self.old_action = 3

    def select_action(self, state, policy_DQN, learn_counter, explore = True):
        sample = random.random()
        eps_threshold = max(EPS_MIN, EPS_MAX - (EPS_MAX - EPS_MIN) * learn_counter / EPS_DECAY)
        self.steps_done += 1
        with torch.no_grad():
            q_values = policy_DQN(state)
        # display.data.q_values.append(q_values.max(1)[0].item())
        if not explore or sample > eps_threshold:
            # Optimal action
            action = q_values.max(1)[1].view(1, 1)
            # print(action)
            return action
        else:
            # Random action
            action = random.randrange(4)
            while action == REVERSED[self.old_action]:
                action = random.randrange(4)
            return torch.tensor([[action]], device=device, dtype=torch.long)

class ReplayMemory:
    def __init__(self, capacity, batch_size):
        self.states = deque([], maxlen=capacity)
        self.actions = deque([], maxlen=capacity)
        self.rewards = deque([], maxlen=capacity)
        self.next_states = deque([], maxlen=capacity)
        self.dones = deque([], maxlen=capacity)
        self.size = 0
        self.capacity = capacity
        self.batch_size = batch_size

    def push(self, state, action, reward, next_state, done):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.next_states.append(next_state)
        self.dones.append(done)
        self.size = min(self.size + 1, self.capacity)

    def sample(self):
        assert self.size >= self.batch_size
        indices = random.sample(range(self.size), k=self.batch_size)
        exps = (self.states, self.actions, self.rewards, self.next_states)
        extract = lambda list_: [list_[i] for i in indices]
        states, actions, rewards, next_states = map(torch.cat, map(extract, exps))
        dones = torch.from_numpy(np.vstack(extract(self.dones)).astype(np.uint8))
        tofloat = lambda x: x.float().to(device)
        tolong = lambda x: x.long().to(device)
        return (
            tofloat(states),
            tolong(actions),
            tofloat(rewards),
            tofloat(next_states),
            tofloat(dones),
        )

    def __len__(self):
        return self.size

def optimization(it, r):
    return it % 2 == 0 and r

def extend_walls(img):
    extension = np.array([[WALL_COLOR_GRAY for x in range(160)] for y in range(3)])
    return np.concatenate([extension, img, extension])

def unit_prepr_obs(obs):
    gray_img = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
    trimmed_img = gray_img[1:171]
    extended_img = extend_walls(trimmed_img)
    final_img = extended_img[1::4, 1::4]
    return np.stack([final_img.astype(np.float32)])

def preprocess_observation(observations, new_obs):
    for i in range(3):
        observations[3 - i] = observations[2 - i]
    observations[0] = unit_prepr_obs(new_obs)
    state = np.concatenate(observations)
    screen = torch.from_numpy(state)
    return screen.unsqueeze(0)

def init_obs(env):
    return [unit_prepr_obs(env.step(0)[0]) for i_step in range(4)][::-1]

ACTIONS = {
    1: [1, 4, 6, 5],
    2: [5, 7, 3, 2],
    3: [6, 8, 3, 2],
    4: [1, 4, 8, 7],
    5: [1, 4, 3, 2],
    6: [1, 4, 3, 2],
    7: [1, 4, 3, 2],
    8: [1, 4, 3, 2],
}

REWARDS = {
    "default": -0.2,
    200: 20,
    50: 15,
    10: 10,
    0: 0,
    "lose": -log(20, 1000),
    "win": 10,
    "reverse": -2,
}

def transform_reward(reward):
    return log(reward, 1000) if reward > 0 else reward

torch.autograd.set_detect_anomaly(True)


policy_dqn = DQN(4).to(device)
target_dqn = DQN(4).to(device)
target_dqn.load_state_dict(policy_dqn.state_dict())
    
optimizer = torch.optim.Adam(policy_dqn.parameters(), lr = 2.5e-4)

memory = ReplayMemory(6000, 128)

dmaker = DecisionMaker(0, policy_dqn)

def show_state(state):
    # show 4 images in 4 subplots
    plt.figure(figsize=(10, 10))
    for i in range(4):
        # show in 4 corners
        plt.subplot(2, 2, i + 1)
        plt.imshow(state[0][i], cmap='gray')
        plt.axis('off')
    plt.show()

def train():
    env = gym.make("ALE/MsPacman-v5", render_mode = "rgb_array")
    episodes = 0
    learn_counter = 0

    while True:
        if dmaker.steps_done > 2_000_000:
            break
        episodes += 1

        observation, info = env.reset()
        lives = 3
        jump_dead_step = False
        old_action = 0

        for i_step in range(50):
            observation, reward, termianted, truncated, info = env.step(3)

        observations = init_obs(env)
        observation, reward, termianted, truncated, info = env.step(3)
        state = preprocess_observation(observations, observation)

        got_reward = False
        old_action = 3

        while True:
            if dmaker.steps_done > 2_000_000:
                break

            action = dmaker.select_action(state, policy_dqn, learn_counter)
            action_ = ACTIONS[old_action][action.item()]

            observation, reward_, terminated, truncated, info = env.step(action_)

            reward = transform_reward(reward_)

            if info["lives"] < lives:
                lives -= 1
                jump_dead_step = True
                got_reward = False
                reward += REWARDS["lose"]
                dmaker.old_action = 3

            if (terminated or truncated) and lives > 0:
                reward += REWARDS["win"]

            got_reward = got_reward or reward > 0
            reward = torch.tensor([reward], device = device)

            old_action = action_

            if reward != 0:
                dmaker.old_action = action.item()

            next_state = preprocess_observation(observations, observation)

            # if dmaker.steps_done % 30 == 0:
            #     show_state(state)

            if got_reward:
                memory.push(state, action, reward, next_state, terminated or truncated)

            state = next_state
            if optimization(dmaker.steps_done, got_reward):
                learn_counter = optimize_model(policy_dqn, target_dqn, memory, optimizer, learn_counter, device)

            if dmaker.steps_done % 8_000 == 0:
                target_dqn.load_state_dict(policy_dqn.state_dict())

            if terminated or truncated:
                torch.cuda.empty_cache()
                break

            if jump_dead_step:
                for i_dead in range(20):
                    observation, reward, termianted, truncated, info = env.step(0)
                jump_dead_step = False
            torch.cuda.empty_cache()
        print(episodes)
        if episodes % 10 == 0:
            torch.save(policy_dqn.state_dict(), "policy_dqn.pth")
            torch.save(target_dqn.state_dict(), "target_dqn.pth")

    torch.save(policy_dqn.state_dict(), "policy_dqn.pth")
    torch.save(target_dqn.state_dict(), "target_dqn.pth")

def test():
    env = gym.make("ALE/MsPacman-v5", render_mode = "human")
    policy_dqn.load_state_dict(torch.load("policy_dqn.pth"))
    target_dqn.load_state_dict(torch.load("target_dqn.pth"))

    observation, info = env.reset()
    lives = 3
    jump_dead_step = False
    old_action = 0

    for i_step in range(50):
        observation, reward, termianted, truncated, info = env.step(3)

    observations = init_obs(env)
    observation, reward, termianted, truncated, info = env.step(3)
    state = preprocess_observation(observations, observation)

    old_action = 3
    learn_counter = 0

    while True:
        if dmaker.steps_done > 2_000_000:
            break

        action = dmaker.select_action(state, policy_dqn, learn_counter, False)
        action_ = ACTIONS[old_action][action.item()]

        observation, reward_, terminated, truncated, info = env.step(action_)

        reward = transform_reward(reward_)

        if info["lives"] < lives:
            lives -= 1
            jump_dead_step = True
            reward += REWARDS["lose"]
            dmaker.old_action = 3

        if (terminated or truncated) and lives > 0:
            reward += REWARDS["win"]

        old_action = action_

        if reward != 0:
            dmaker.old_action = action.item()

        next_state = preprocess_observation(observations, observation)

        state = next_state

        if terminated or truncated:
            break

        if jump_dead_step:
            for i_dead in range(20):
                observation, reward, termianted, truncated, info = env.step(0)
            jump_dead_step = False

    env.close()


if sys.argv[1] == "train":
    train()
elif sys.argv[1] == "test":
    test()