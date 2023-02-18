import sys
from collections import deque
from math import log, exp
import random

import gymnasium as gym
import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.autograd.set_detect_anomaly(True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MAX_EPISODES = 10000
MAX_STEPS_IN_EPISODE = 10000
MEAN_WINDOW_LEN = 50
REPORT_PERIOD = 5
TARGET_UPDATE_PERIOD = 8_000
SAVE_PERIOD = 10
START_FRAME_CNT = 50
DEAD_FRAME_CNT = 20
DEFAULT_MOVE = 3
IMAGE_SHAPE = (84, 84)
EPS_MIN = 0.1
EPS_MAX = 1.0
EPS_DECAY = 60_000
GAMMA = 0.999
LEARNING_RATE = 0.00025
MEMORY_CAPACITY = 6_000
BATCH_SIZE = 128
PREFIX = ""
ACTION_MAP = {
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

scores = deque(maxlen=MEAN_WINDOW_LEN)
max_score = 0
state = None
steps_done = 0
steps_in_episode = 0
steps_explored_in_episode = 0
policy_dqn = None
target_dqn = None
memory = None
optimizer = None

class PaperDQN(nn.Module):

    def __init__(self, output_size):
        super().__init__()
        self.output_size = output_size
        shape = IMAGE_SHAPE
        self.conv1 = nn.Conv2d(4, 16, kernel_size=(8, 8), stride=(4, 4)) 
        shape = ((shape[0] - 8) // 4 + 1, (shape[1] - 8) // 4 + 1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=(4, 4), stride=(2, 2))
        shape = ((shape[0] - 4) // 2 + 1, (shape[1] - 4) // 2 + 1)
        self.fc1 = nn.Linear(32 * shape[0] * shape[1], 256)
        self.fc2 = nn.Linear(256, output_size)

    def forward(self, x):
        x = x.to(device)

        x = self.conv1(x)
        x = F.relu(x)

        x = self.conv2(x)
        x = F.relu(x)

        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = F.relu(x)

        x = self.fc2(x)
        return x

class ReplayMemory:
    def __init__(self, capacity, batch_size):
        self.frame_sequences = deque([], maxlen=capacity)
        self.actions = deque([], maxlen=capacity)
        self.rewards = deque([], maxlen=capacity)
        self.dones = deque([], maxlen=capacity)
        self.size = 0
        self.capacity = capacity
        self.batch_size = batch_size

    def push(self, state, action, reward, next_state, done):
        assert (state[:-1] == next_state[1:]).all()
        frame_sequence = np.concatenate((next_state[0:1], state), axis=0)
        self.frame_sequences.append(frame_sequence)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.size = min(self.size + 1, self.capacity)

    def sample_torch(self):
        assert self.size >= self.batch_size
        indices = random.sample(range(self.size), k=self.batch_size)
        mems = (self.frame_sequences, self.actions, self.rewards, self.dones)
        frame_sequences, actions, rewards, dones = map(lambda mem: [mem[i] for i in indices], mems)

        frame_sequences = np.array(frame_sequences)


        states = torch.from_numpy(frame_sequences[:, 1:]).float().to(device)
        next_states = torch.from_numpy(frame_sequences[:, :-1]).float().to(device)
        actions = torch.from_numpy(np.vstack(actions)).long().to(device)
        rewards = torch.from_numpy(np.vstack(rewards)).float().to(device)
        dones = torch.from_numpy(np.vstack(dones).astype(np.uint8)).float().to(device)
        
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return self.size

def report(episode_cnt, score, loss, explore_ratio, steps_in_episode):
    global scores, max_score
    scores.append(score)
    max_score = max(score, max_score)

    loss = round(loss, 5)
    explore_ratio = round(explore_ratio*100, 2)

    avg_score = sum(scores)/len(scores)
    avg_score = round(avg_score, 2)

    if episode_cnt == 1:
        with open(PREFIX + "record.csv", "w") as f:
            f.write("episode,score,avg_score,max_score,loss,explore_ratio,duration\n")
    with open(PREFIX + "record.csv", "a") as f:
        f.write(f"{episode_cnt},{score},{avg_score},{max_score},{loss},{explore_ratio},{steps_in_episode}\n")

    if REPORT_PERIOD > 1 and episode_cnt % REPORT_PERIOD != 1:
        return

    print(f">> Episode: {episode_cnt}")
    print(f"Score: {score},\tAverage score: {avg_score},\tmax_score: {max_score}")
    print(f"Loss: {loss},\texplore_ratio: {explore_ratio},\tduration: {steps_in_episode}")
    print()


def preprocess(frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

    frame = frame[1:171]

    GRAY = cv2.cvtColor(np.array([[[228, 111, 111]]], dtype=np.uint8), cv2.COLOR_RGB2GRAY)[0][0]
    wall = [[GRAY for _ in range(160)] for _ in range(3)]
    frame = np.concatenate([wall, frame, wall], axis=0)

    frame = cv2.resize(frame, IMAGE_SHAPE)
    frame = frame / 255.0

    return frame


def initialize_state(frame):
    global state
    frame = preprocess(frame)
    state = np.zeros((4, *IMAGE_SHAPE), dtype=np.float32)
    state[0] = frame.copy()
    state[1] = frame.copy()
    state[2] = frame.copy()
    state[3] = frame.copy()


def add_frame(state, frame):
    next_state = np.empty((4, *IMAGE_SHAPE), dtype=np.float32)
    next_state[0] = frame
    next_state[1:] = state[:-1]
    return next_state


def show_state():
    global state
    plt.figure(figsize=(8, 8))
    for i in range(4):
        plt.subplot(2, 2, i+1)
        plt.imshow(state[i], cmap='gray')
    plt.show()
    return


def process_reward(reward, terminated, truncated, info, lives):
    just_died = False
    reward = log(reward, 1000) if reward > 0 else reward

    if info["lives"] < lives:
        lives -= 1
        just_died = True
        reward += REWARDS["lose"]
    
    if (terminated or truncated) and lives > 0:
        reward += REWARDS["win"]

    return reward, lives, just_died


def do_nothing(env, frame_cnt):
    for _ in range(frame_cnt):
        frame, reward, terminated, truncated, info = env.step(DEFAULT_MOVE)
        assert reward == 0
        assert terminated == False
        assert truncated == False
    return frame


def optimize_model():
    global policy_dqn, target_dqn, memory, optimizer
    if len(memory) < BATCH_SIZE:
        return

    states, actions, rewards, next_states, dones = memory.sample_torch()

    predicted_targets = policy_dqn(states)
    predicted_targets = predicted_targets.gather(1, actions)

    target_values = target_dqn(next_states).detach()
    target_values = target_values.max(1)[0].unsqueeze(1)

    labels = rewards + (1 - dones) * GAMMA * target_values
    
    criterion = nn.SmoothL1Loss()
    loss = criterion(predicted_targets, labels.detach()).to(device)

    optimizer.zero_grad()
    loss.backward()
    for param in policy_dqn.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()

    return loss.item()


def select_action_by_policy(state):
    global policy_dqn
    with torch.no_grad():
        state = torch.from_numpy(state).unsqueeze(0).to(device)
        q_values = policy_dqn(state)
        action = q_values.max(1)[1].view(1, 1)
    return action.item()


def select_action(state, explore, old_action):
    global steps_done, steps_explored_in_episode, steps_in_episode
    steps_done += 1
    steps_in_episode += 1
    sample = random.random()
    eps_threshold = EPS_MIN + (EPS_MAX - EPS_MIN) * exp(-1. * steps_done / EPS_DECAY)

    if explore and sample < eps_threshold:
        action = random.randint(0, 3)
        steps_explored_in_episode += 1
        while action ^ 1 == old_action:
            action = random.randint(0, 3)
    else:
        action = select_action_by_policy(state)

    return action


def train_episode(env):
    global state, steps_in_episode, steps_explored_in_episode, memory, steps_done

    frame, info = env.reset()
    lives = 3
    just_died = False
    old_actual_action = 3
    old_action = 3

    steps_in_episode = 0
    steps_explored_in_episode = 0

    frame = do_nothing(env, START_FRAME_CNT)
    initialize_state(frame)

    got_reward = False
    score = 0
    loss = 0
    while True:
        if steps_in_episode > MAX_STEPS_IN_EPISODE:
            break

        action = select_action(state, True, old_action)
        actual_action = ACTION_MAP[old_actual_action][action]

        frame, reward, terminated, truncated, info = env.step(actual_action)
        score += reward

        processed_reward, lives, just_died = process_reward(reward, terminated, truncated, info, lives)

        if just_died:
            got_reward = False
            old_action = 3
        if processed_reward != 0:
            got_reward = True

        old_actual_action = actual_action

        frame = preprocess(frame)
        next_state = add_frame(state, frame)

        if got_reward:
            memory.push(state, action, processed_reward, next_state, terminated or truncated)
            pass

        if reward != 0:
            old_action = action
        
        state = next_state
        if steps_in_episode % 2 == 0:
            loss = optimize_model()
        if steps_done % TARGET_UPDATE_PERIOD == 0:
            target_dqn.load_state_dict(policy_dqn.state_dict())

        torch.cuda.empty_cache()

        if terminated or truncated:
            break

        if just_died:
            frame = do_nothing(env, DEAD_FRAME_CNT)
            just_died = False

    explore_ratio = steps_explored_in_episode / steps_in_episode
    return score, loss, explore_ratio, steps_in_episode


def save_model(episode_cnt):
    if SAVE_PERIOD == 1 or episode_cnt % SAVE_PERIOD == 0:
        torch.save(policy_dqn.state_dict(), PREFIX+"policy_dqn.pth")
        torch.save(target_dqn.state_dict(), PREFIX+"target_dqn.pth")
        print(">>Saved model at episode", episode_cnt)


def make_ddqn(new_model):
    policy_dqn = PaperDQN(4).to(device)
    target_dqn = PaperDQN(4).to(device)

    if new_model:
        target_dqn.load_state_dict(policy_dqn.state_dict())
    else:
        policy_dqn.load_state_dict(torch.load(PREFIX+"policy_dqn.pth", device))
        target_dqn.load_state_dict(torch.load(PREFIX+"target_dqn.pth", device))
    return policy_dqn, target_dqn


def train(new_model):
    global policy_dqn, target_dqn, memory, optimizer
    env = gym.make('MsPacman-v0', render_mode = "rgb_array")
    episode_cnt = 0

    policy_dqn, target_dqn = make_ddqn(new_model)
    memory = ReplayMemory(MEMORY_CAPACITY, BATCH_SIZE)
    optimizer = torch.optim.Adam(policy_dqn.parameters(), lr=LEARNING_RATE)

    while True:
        if episode_cnt >= MAX_EPISODES:
            break
        score, loss, explore_ratio, steps_in_episode = train_episode(env)
        episode_cnt += 1

        report(episode_cnt, score, loss, explore_ratio, steps_in_episode)
        
        save_model(episode_cnt)
    
    env.close()


def test():
    global state, policy_dqn, target_dqn
    env = gym.make('MsPacman-v0', render_mode = "human")

    policy_dqn, target_dqn = make_ddqn(False)

    frame, info = env.reset()
    old_actual_action = 3

    frame = do_nothing(env, START_FRAME_CNT)
    initialize_state(frame)

    score = 0
    while True:
        action = select_action(state, False, None)
        actual_action = ACTION_MAP[old_actual_action][action]

        frame, reward, terminated, truncated, info = env.step(actual_action)
        score += reward

        old_actual_action = actual_action

        frame = preprocess(frame)
        next_state = add_frame(state, frame)
        
        state = next_state

        torch.cuda.empty_cache()

        if terminated or truncated:
            break

    env.close()

    return


if __name__ == '__main__':
    if sys.argv[1] == "train":
        train(True)
    elif sys.argv[1] == "retrain":
        train(False)
    elif sys.argv[1] == "test":
        test()
    else:
        print("Invalid command")