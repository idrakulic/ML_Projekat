#!/usr/bin/env python
# coding: utf-8

# In[3]:


import gym
import random
import os
import numpy as np
from collections      import deque
from keras.models     import Sequential, clone_model
from keras.layers     import Dense, Convolution2D, Flatten
from keras.optimizers import adam_v2
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image
import json
from copy import deepcopy
from time import sleep


# In[4]:


class Agent():
    def __init__(self, state_size, action_size, cold_start=True, episode_goal=0, current_episode =0):
        self.weight_backup = "space_invaders_weight_batch_16_improved.h5"
        self.parameters_backup = "space_invaders_parameters_batch_16_improved.json"
        self.scores_backup = "scores_batch_16_improved.json"
            
        self.current_episode = current_episode
        self.goal_episode = current_episode + episode_goal
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=10000)
        self.learning_rate = 0.001
        self.gamma = 0.95
        self.exploration_rate = 1.0
        self.exploration_min = 0.01
        self.exploration_decay = 0.95

        self.current_network = self._build_model()
        if not cold_start:
            net_to_load = "episode_goal_"+ str(current_episode) +"_" + self.weight_backup
            parameters_to_load = "episode_goal_"+ str(current_episode) +"_" + self.parameters_backup
            if os.path.isfile(net_to_load):
                self.load_model(net_to_load)
                self.load_parameters(parameters_to_load)
            else:
                print("Nema mreze!")

        self.target_network = clone_model(self.current_network)

    def _build_model(self):
        """Builds a neural network for DQN to use."""
        model = Sequential()
        model.add(Convolution2D(32, 9, strides=2, activation='relu',
                         input_shape=self.state_size, data_format="channels_last"))
        model.add(Convolution2D(24, 7, strides=2, activation='relu'))
        model.add(Convolution2D(16, 5, strides=2, activation='relu'))
        model.add(Flatten())
        model.add(Dense(512, activation='relu'))
        model.add(Dense(128, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=adam_v2.Adam(learning_rate=self.learning_rate))
        
        print(model.summary())
        
        return model

    def save_model(self, scores):
        """Saves the model weights to the path set in the object."""
        net_to_save = "episode_goal_"+ str(self.current_episode) +"_" + self.weight_backup
        self.current_network.save(net_to_save)

        if self.exploration_rate > self.exploration_min:
            self.exploration_rate *= self.exploration_decay

        self.save_parameters()
        self.target_network = clone_model(self.current_network)
        self.save_scores(scores)

    def load_model(self, net_to_load):
        """LOads the model weights from the path set in the object."""
        self.current_network.load_weights(net_to_load)

    def save_parameters(self):
        fajl = {}
        fajl["exploration_rate"] = self.exploration_rate
        fajl["exploration_decay"] = self.exploration_decay
        fajl["exploration_min"] = self.exploration_min
        fajl["gamma"] = self.gamma

        parameters_to_save = "episode_goal_"+ str(self.current_episode) +"_" + self.parameters_backup

        with open(parameters_to_save, "w") as f:
            json.dump(fajl, f)
          
    def load_parameters(self, parameters_to_load):
        with open(parameters_to_load, "r") as f:
            fajl = json.load(f)

        self.exploration_rate = fajl["exploration_rate"]
        self.exploration_decay = fajl["exploration_decay"]
        self.exploration_min = fajl["exploration_min"]
        self.gamma = fajl["gamma"]

    def save_scores(self, scores):
        scores_to_save = "episode_goal_"+ str(self.current_episode) +"_" + self.scores_backup
        with open(scores_to_save, "w") as f:
            json.dump(scores,f)

    def act(self, state):
        """Acts according to epsilon greedy policy."""
        if np.random.rand() <= self.exploration_rate:
            return random.randrange(self.action_size)
        act_values = self.target_network.predict(state)
        return np.argmax(act_values[0])

    # Cuvamo iskustvo u memoriji
    def remember(self, state, action, reward, next_state, done):
        """Saves experience in memory."""
        self.memory.append((state, action, reward, next_state, done))

    # Ucimo na osnovu informacija u memoriji
    def learn(self, sample_batch_size):
        if len(self.memory) < sample_batch_size:
            return
        sample_batch = random.sample(self.memory, sample_batch_size)

        states =[]
        q_values = []

        for state, action, reward, next_state, done in sample_batch:

            next_state_prediction = self.target_network.predict(next_state)
            next_q_value = np.max(next_state_prediction)
            q = self.current_network.predict(state)

            target = reward
            if not done:
                target = reward + self.gamma * next_q_value
            q[0][action] = target

            states.append(state)
            q_values.append(q)

        states = np.reshape(states, (sample_batch_size, 100,80,3))
        q_values = np.reshape(q_values, (sample_batch_size, 6))

        self.current_network.fit(states, q_values, epochs=1, verbose=0, batch_size=sample_batch_size)


# In[ ]:




