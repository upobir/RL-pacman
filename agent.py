import random
from collections import deque, namedtuple
import sys
import math
import csv

import cv2
import torch
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt


from model import Qnet, QTrainer

# TODO negative reward for dying
# TODO negative reward for not eating

# Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'terminated'))

# class ReplayMemory(object):
#     def __init__(self, capacity):
#         self.capacity = capacity
#         self.memory = deque(maxlen=capacity)

#     def push(self, *args):
#         """Saves a transition."""
#         self.memory.append(Transition(*args))
#         return

#     def sample(self, batch_size):
#         """Randomly sample a batch of transitions from memory."""
#         return random.sample(self.memory, batch_size)

#     def __len__(self):
#         return len(self.memory)


class Agent:
    MAX_MEMORY = 4_000
    BATCH_SIZE = 128
    LR = 0.001
    GAMMA = 0.999
    EPS_START = 0.9
    EPS_END = 0.05
    EPS_DECAY = 15000
    # TAU = 0.005

    def __init__(self, max_actions):
        """
        initializes the agent with memory, image size, model, trainer and game count
        """
        self.n_games = 0
        self.memory0 = deque(maxlen=self.MAX_MEMORY)
        self.memory1 = deque(maxlen=self.MAX_MEMORY)
        self.memory2 = deque(maxlen=self.MAX_MEMORY)
        self.image_size = (84, 84)
        self.max_actions = max_actions
        self.epsilon = 1
        self.model = Qnet(self.max_actions)
        self.trainer = QTrainer(self.model, lr=self.LR, gamma=self.GAMMA)

        self.last_ate = 0
        self.last_lives = 2
        self.explore_moves = 0
        self.total_moves = 0
        self.steps_done = 0


    def preprocess(self, frame):
        """
        converts single rgb frame to a processed frame
        """
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(frame, self.image_size)
        frame = frame / 255.0
        return frame


    def processed_reward(self, frame, reward, terminated):
        """
        returns a processed reward
        """
        live1 = frame[70, 15]
        live2 = frame[70, 7]
        cur_lives = (1 if live1 > 0.1 else 0) + (1 if live2 > 0.1 else 0)

        if terminated:
            self.last_ate = 0
            reward = -1
        elif cur_lives < self.last_lives:
            reward = -1
            self.last_lives = cur_lives
            self.last_ate = 0
        elif reward == 0:
            self.last_ate += 1
            reward = 0
        else:
            reward = 1
            self.last_ate = 0

        return reward


    def remember(self, state, action, reward, next_frame, terminated):
        """
        adds to memory, state should be last 4 frames.
        """
        if reward == -1:
            self.memory0.append((state, action, reward, next_frame, terminated))
        elif reward == 0:
            self.memory1.append((state, action, reward, next_frame, terminated))
        else:
            self.memory2.append((state, action, reward, next_frame, terminated))
        return


    def train_long_memory(self, batch_size):
        """
        uses memory to train
        """
        states_list = []
        actions_list = []
        rewards_list = []
        next_frames_list = []
        terminateds_list = []
        for memory in [self.memory0, self.memory1, self.memory2]:
            if len(memory) > batch_size:
                mini_sample = random.sample(memory, batch_size)
            else:
                mini_sample = memory

            states, actions, rewards, next_frames, terminateds = zip(*mini_sample)

            states_list += states
            actions_list += actions
            rewards_list += rewards
            next_frames_list += next_frames
            terminateds_list += terminateds

        states = np.array(states_list)
        actions = np.array(actions_list)
        rewards = np.array(rewards_list)
        next_frames = np.array(next_frames_list).reshape(-1, 1, *self.image_size)
        next_states = np.concatenate((states[:, 1:], next_frames), axis=1)
        terminateds = np.array(terminateds_list)

        indices = np.random.permutation(len(states))
        states = states[indices]
        actions = actions[indices]
        rewards = rewards[indices]
        next_states = next_states[indices]
        terminateds = terminateds[indices]

        loss = self.trainer.train_step(states, actions, rewards, next_states, terminateds)
        return loss


    def train_short_memory(self, state, action, reward, next_state, terminated):
        """
        trains with single action, state should be last 4 frames.
        """
        self.trainer.train_step(state, action, reward, next_state, terminated)
        return


    def get_action(self, state, explore = False):
        """
        given state, returns action, state should be last 4 frames.
        """
        sample = random.random()
        eps_threshold = self.EPS_END + (self.EPS_START - self.EPS_END) * math.exp(-1. * self.steps_done / self.EPS_DECAY)
        if eps_threshold < 2 * self.EPS_END:
            eps_threshold = self.EPS_START
            self.steps = 0
        self.steps_done += 1
        self.total_moves += 1

        if sample > eps_threshold:
            state = torch.tensor(state, dtype=torch.float)
            state = state.unsqueeze(0)
            with torch.no_grad():
                prediction = self.model(state)
            return torch.argmax(prediction).item()
        else:
            self.explore_moves += 1
            return random.randint(0, self.max_actions - 1)


    def end_game(self):
        """
        called at end of game
        """
        self.n_games += 1
        self.epsilon = max(0.1, 1 / max(1, self.n_games-5))
        self.last_ate = 0
        self.last_lives = 2
        self.explore_moves = 0
        self.total_moves = 0

# episode_durations = []

# def plot_durations(show_results = False):
#     plt.figure(1)
#     durations_t = torch.tensor(episode_durations, dtype=torch.float)
#     if show_results:
#         plt.title('Result')
#     else:
#         plt.clf()
#         plt.title('Training...')
#     plt.xlabel('Episode')
#     plt.ylabel('Duration')
#     plt.plot(durations_t.numpy())
#     # Take 100 episode averages and plot them too
#     if len(durations_t) >= 100:
#         means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
#         means = torch.cat((torch.zeros(99), means))
#         plt.plot(means.numpy())

#     plt.pause(0.001)  # pause a bit so that plots are updated

SKIP = 3

def train(filename = None):
    """
    train dqn
    """
    with open("record.csv", "w") as f:
        f.write("n_games,score,processed_score,explore_ratio,mean_score,mean_processed_score,loss,duration\n")
    score_window = deque(maxlen = 10)
    processed_score_window = deque(maxlen = 10)
    cur_score = 0
    cur_processed_score = 0
    record = 0
    game = gym.make("ALE/MsPacman-v5", render_mode="rgb_array")
    agent = Agent(game.action_space.n)
    if filename is not None:
        agent.model.load_state_dict(torch.load(filename))

    frames = deque(maxlen=4)

    frame, info = game.reset()
    frame = agent.preprocess(frame)
    for _ in range(4):
        frames.append(frame)

    while True:
        old_state = np.array(frames)
        action = agent.get_action(old_state, explore=True)
        frame, reward, terminated, truncated, info = game.step(action)
        frame = agent.preprocess(frame)
        if agent.last_ate > 500:
            terminated = True
        processed_reward = agent.processed_reward(frame, reward, terminated or truncated)
        cur_score += reward
        cur_processed_score += processed_reward

        frames.append(frame)
        cur_state = np.array(frames)

        agent.train_short_memory(old_state, action, processed_reward, cur_state, terminated or truncated)
        agent.remember(old_state, action, processed_reward, frame, terminated or truncated)

        for i in range(SKIP):
            if terminated or truncated:
                break
            old_state = np.array(frames)
            frame, reward, terminated, truncated, info = game.step(0)
            frame = agent.preprocess(frame)
            processed_reward = agent.processed_reward(frame, reward, terminated)
            cur_score += reward
            cur_processed_score += processed_reward
            
            frames.append(frame)
            cur_state = np.array(frames)
            if processed_reward != 0:
                agent.train_short_memory(old_state, 0, processed_reward, cur_state, terminated or truncated)
                agent.remember(old_state, 0, processed_reward, frame, terminated or truncated)

        if terminated or truncated:
            if cur_score > record:
                record = cur_score
            
            explore_ratio = agent.explore_moves / agent.total_moves
            duration = agent.total_moves
            agent.end_game()
            
            print(f"Game: {agent.n_games}")
            print(f"Processed Score: {cur_processed_score}\t\t Score: {cur_score}")
            score_window.append(cur_score)
            processed_score_window.append(cur_processed_score)
            mean_score = np.mean(score_window)
            mean_processed_score = np.mean(processed_score_window)
            print(f"Mean Processed Score: {round(mean_processed_score, 2)}\t Mean Score: {round(mean_score, 2)}")
            print(f"Explore Ratio: {round(explore_ratio*100, 2)}%, duration: {duration}")

            if agent.n_games % 5 == 0:
                torch.save(agent.model.state_dict(), "model.pth")

            frame, info = game.reset()
            frame = agent.preprocess(frame)

            for _ in range(4):
                frames.append(frame)

            loss = agent.train_long_memory(512)
            print("loss:", round(loss.item(), 5))
            print()

            with open("record.csv", "a") as f:
                f.write(f"{agent.n_games},{cur_score},{cur_processed_score},{explore_ratio},{mean_score},{mean_processed_score},{loss.item()},{duration}\n")

            cur_score = 0
            cur_processed_score = 0

            if agent.n_games == 5000:
                break
        

def test():
    """
    test dqn
    """
    cur_score = 0
    cur_processed_score = 0
    game = gym.make("ALE/MsPacman-v5", render_mode="human")
    agent = Agent(game.action_space.n)
    agent.model.load_state_dict(torch.load("model.pth"))

    frames = deque(maxlen=4)

    frame, info = game.reset()
    frame = agent.preprocess(frame)
    for _ in range(4):
        frames.append(frame)

    while True:
        old_state = np.array(frames)
        action = agent.get_action(old_state)

        frame, reward, terminated, truncated, info = game.step(action)
        frame = agent.preprocess(frame)
        processed_reward = agent.processed_reward(frame, reward, terminated)
        cur_score += reward
        cur_processed_score += processed_reward

        frames.append(frame)

        for i in range(SKIP):
            if terminated or truncated:
                break
            frame, reward, terminated, truncated, info = game.step(0)
            frame = agent.preprocess(frame)
            processed_reward = agent.processed_reward(frame, reward, terminated)
            cur_score += reward
            cur_processed_score += processed_reward

            frames.append(frame)

        if terminated or truncated:
            print(f"Processed Score: {cur_processed_score}, Score: {cur_score}")
            break

if __name__ == "__main__":
    if sys.argv[1] == "test":
        test()
    elif sys.argv[1] == 'retrain':
        train(sys.argv[2])
    elif sys.argv[1] == "train":
        train()