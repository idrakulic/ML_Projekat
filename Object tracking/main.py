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


# In[5]:


import spaceInvaders, otherObjects, missiles_Tracker, drawObjectTracked

# In[ ]:


if __name__ == "__main__":
    env = spaceInvaders.SpaceInvaders()

    env.run()