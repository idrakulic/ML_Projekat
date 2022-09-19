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
from spaceInvaders import SpaceInvaders

if __name__ == "__main__":
    game = SpaceInvaders(episodes_to_train=10,continue_training=True, current_episode=140)
    game.run_train()