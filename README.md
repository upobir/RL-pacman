## RL-Pacman 

![image not loaded](https://github.com/upobir/RL-pacman/blob/main/RL-Pacman.png)

RL-Pacman is a framework which plays the famous game Pacman using Reinforcement learning. We have used the OpenAI Gymnasium project for game logic implementation and interaction. Our RL model has been trained using double deep Q-learning. It consists of two CNN's (a policy and a target) that are used to predict Q values for each move given a "state" from the game. A "state" is defined as a sequence of 4 frames as seen from the game screen.
We have made the model play many games or episodes. To choose action at a given state, we have used an epsilon threshold to choose random action with decreasing probability. Otherwise the policy network is used to choose a move.
We have stored states, actions, rewards, next states as a tuple in a "replay buffer" of a fixed maximum size deque. 
Periodically (after few frames) we train our two CNN's on a random sample from the buffer.
