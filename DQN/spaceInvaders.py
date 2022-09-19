#!/usr/bin/env python
# coding: utf-8

# In[2]:


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

from agent import Agent

# In[ ]:


class SpaceInvaders:
    def __init__(self, continue_training=False, episodes_to_train = 100, current_episode=0):
        self.sample_batch_size = 128
        self.steps_frequency_train = 32
        self.episodes = episodes_to_train
        self.env = gym.make('SpaceInvaders-v4')#, render_mode="human")

        self.state_size = (100,80,3)
        self.action_size = self.env.action_space.n
        self.agent = Agent(self.state_size, self.action_size, cold_start= (not continue_training), current_episode= current_episode)
        self.scores = []

        if(continue_training):
            self.scores = self.load_scores()

    def load_model(self):
        self.agent.load_model()
        
    def preprocess_image(self, img):
        img_prep = img[:200,:]

        thumbnail_size = img_prep.shape
        thumbnail_size = (thumbnail_size[1]//2, thumbnail_size[0]//2)
        img_prep = Image.fromarray(img_prep)
        img_prep.thumbnail(thumbnail_size)
        img_prep = np.asarray(img_prep)

        return img_prep
        
    def load_scores(self):
        scores_to_load = "episode_goal_"+ str(self.agent.current_episode) + "_" + self.agent.scores_backup
        with open(scores_to_load, "r") as f:
            scores_loaded = json.load(f)
        
        return scores_loaded
        
    def run_test(self):
        state = self.env.reset()

        #self.env.render()
        done = False
        score = 0

        index = 0
        while not done:
            state = self.preprocess_image(state)
            state = np.reshape(state, [1] + list(self.state_size))
            action = self.agent.act(state)
            next_state, reward, done, _ = self.env.step(action)
            score  += reward
            state = next_state
            index += 1
            #self.env.render()
        print(f'Score: {score}')
        self.env.close()
        return score

    def run_train(self):
        
        ispis = ""
        for index_episode in tqdm(range(self.episodes)):
            state = self.env.reset()
            state = self.preprocess_image(state)
            
            state = np.reshape(state, [1] + list(self.state_size))

            done = False
            index = 0
            lives = 3
            score = 0
            while not done:
                action = self.agent.act(state)

                # Azuriramo stanje okruzenja.
                next_state, reward, done, info = self.env.step(action)
                next_state = self.preprocess_image(next_state)
                score += reward
                if(info["lives"]<lives):
                    score -= 100
                    lives -= 1
                # Menjamo oblik vektora za stanje kako bi stanje kasnije
                # mogli da propustimo kroz mrezu.
                next_state = np.reshape(next_state, [1] + list(self.state_size))

                # Cuvamo iskustvo.
                self.agent.remember(state, action, reward, next_state, done)
                state = next_state
                index += 1

                if(index%self.steps_frequency_train==0):
                    self.agent.learn(self.sample_batch_size)


            self.agent.current_episode+=1
            
            print(f'Episode {self.agent.current_episode}/{self.agent.goal_episode} ; Score: {score} ; Duzina epizode: {index}\n')

            # Belezimo nagradu koju je agent osvojio u epizodi.
            self.scores.append(score)

            if((index_episode+1)%5 == 0):
                scores_to_store = deepcopy(self.scores)
                self.agent.save_model(scores_to_store)

            self.agent.learn(self.sample_batch_size)

        
        # Kada se optimizacioni proces zavrsi, cuvamo tezine mreze.
        #scores_to_store = deepcopy(self.scores)
        #self.agent.save_model(scores_to_store)

        return self.scores

