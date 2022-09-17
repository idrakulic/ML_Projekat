#!/usr/bin/env python
# coding: utf-8

# In[1]:


import gym
import matplotlib.pyplot as plt
import random
import numpy as np
import os
from collections import deque, OrderedDict
import cv2
from copy import deepcopy


# In[2]:


class OtherObjects():
    def __init__(self, purple_look, purple_filter, barrier_initial):
        self.purple_look = purple_look
        self.purple_filter = purple_filter
        self.barrier_initial = barrier_initial
        self.frame_height, self.frame_width = 210, 160
        self.purple_height, self.purple_width = 8, 7
        self.barrier_positions = [[157,42],[157,74],[157,106]]
        self.barrier_size = (18, 8)
        self.rgb_barrier = np.array([181, 83, 40], dtype="uint8")
        self.max_barrier_value = 112
        
    def find_purple(self, image):
        found = False
        purple_coords = None
        for i in range(self.frame_width-8):
            if(np.array_equal(image[12:20, i:i+7]*self.purple_filter, self.purple_look)):
                found = True
                purple_coords = (12,i)
                break
        
        return purple_coords
    
    def count_barrier_health(self, image):
        h, w = self.barrier_size
        procenti = []
        
        for b in self.barrier_positions:
            k = 0 # brojimo crvene kockice
            x, y = b
            
            for i in range(h):
                for j in range(w):
                    if(np.array_equal(image[x+i,y+j], self.rgb_barrier)):
                        k+=1
            
            procenti.append(int(round(100*k/self.max_barrier_value)))
            
        return procenti


# In[ ]:




