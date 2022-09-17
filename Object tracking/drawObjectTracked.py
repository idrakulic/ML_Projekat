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


# In[ ]:


class DrawObjectTracked:
    def __init__(self):
        pass
    
    def draw_left_arrow(self, image, rgb = np.array([0,255, 0], dtype="uint8")):
        new_image = image.copy()
        
        h = len(new_image)
        w = len(new_image[0])
        
        middle_row = h//2
        
        for i in range(middle_row):
            #if(middle_row-i-1 < 0 or i+1>w-1):
            #    break
            
            new_image[middle_row-i-1,i+1] = rgb
            new_image[middle_row+i+1,i+1] = rgb
            
        for i in range(2,w):
            new_image[middle_row, i] = rgb
        
        new_image[middle_row, 0] = rgb
            
        return new_image
    
    def draw_right_arrow(self, image, rgb = np.array([0,255, 0], dtype="uint8")):
        new_image = image.copy()
        
        h = len(new_image)
        w = len(new_image[0])
        
        middle_row = h//2
        
        for i in range(middle_row):
            #if(middle_row-i-1 < 0 or i+1<0):
            #    break
            
            new_image[middle_row-i-1,w-2-i] = rgb
            new_image[middle_row+i+1,w-2-i] = rgb
            
        for i in range(w-3,-1,-1):
            new_image[middle_row, i] = rgb
            
        new_image[middle_row, w-1] = rgb
            
        return new_image
            
    def draw_percent(self, image ,rgb = np.array([0,0,255], dtype="uint8")):
        new_image = image.copy()
        
        h = len(new_image)
        w = len(new_image[0])
        
        for i in range(h):
            new_image[i,w-1-i] = rgb
            
        for i in range(3):
            new_image[0,i] = rgb
            new_image[i,0] = rgb
            new_image[2,i] = rgb
            new_image[i,2] = rgb
            
            new_image[h-1,w-1-i] = rgb
            new_image[h-3,w-1-i] = rgb
            new_image[h-1-i,w-1] = rgb
            new_image[h-1-i,w-3] = rgb
            
        return new_image
    
    def draw_box(self, image, rgb):
        new_image = image.copy()
        height = len(new_image)
        width = len(new_image[0])
        
        for i in range(height):
            new_image[i,0] = rgb
        for i in range(width):
            new_image[0,i] = rgb
        for i in range(height):
            new_image[i,width-1] = rgb
        for i in range(width):
            new_image[height-1,i] = rgb
            
        return new_image
    
    def draw_id(self, image, number, rgb=np.array([255,0,0], dtype="uint8")):
        new_image = image.copy()
        height = len(new_image)
        width = len(new_image[0])
        
        if(number == 0):
            new_image = self.desna_vertikalna(new_image, rgb)
            new_image = self.leva_vertikalna(new_image, rgb)
            new_image = self.gornja_horizontalna(new_image, rgb)
            new_image = self.donja_horizontalna(new_image, rgb)

            return new_image
        
        elif(number == 1):
            new_image = self.desna_vertikalna(new_image, rgb)
            
            return new_image
        
        elif(number == 2):
            new_image = self.gornja_horizontalna(new_image, rgb)
            new_image = self.donja_leva(new_image, rgb)
            new_image = self.gornja_desna(new_image, rgb)
            new_image = self.srednja_horizontalna(new_image, rgb)
            new_image = self.donja_horizontalna(new_image, rgb)
            
            return new_image
        
        elif(number == 3):
            new_image = self.desna_vertikalna(new_image, rgb)
            new_image = self.gornja_horizontalna(new_image, rgb)
            new_image = self.donja_horizontalna(new_image, rgb)
            new_image = self.srednja_horizontalna(new_image, rgb)
            
            return new_image
        
        elif(number == 4):
            new_image = self.gornja_leva(new_image, rgb)
            new_image = self.srednja_horizontalna(new_image, rgb)
            new_image = self.desna_vertikalna(new_image, rgb)
            
            return new_image
        
        elif(number == 5):
            new_image = self.gornja_leva(new_image, rgb)
            new_image = self.srednja_horizontalna(new_image, rgb)
            new_image = self.donja_desna(new_image, rgb)
            new_image = self.gornja_horizontalna(new_image, rgb)
            new_image = self.donja_horizontalna(new_image, rgb)
            
            return new_image
            
        elif(number == 6):
            new_image = self.leva_vertikalna(new_image, rgb)
            new_image = self.srednja_horizontalna(new_image, rgb)
            new_image = self.gornja_horizontalna(new_image, rgb)
            new_image = self.donja_horizontalna(new_image, rgb)
            new_image = self.donja_desna(new_image, rgb)
            
            return new_image
        
        elif(number == 7):
            new_image = self.gornja_horizontalna(new_image, rgb)
            new_image = self.desna_vertikalna(new_image, rgb)
            
            return new_image
        
        elif(number == 8):
            new_image = self.gornja_horizontalna(new_image, rgb)
            new_image = self.desna_vertikalna(new_image, rgb)
            new_image = self.leva_vertikalna(new_image, rgb)
            new_image = self.donja_horizontalna(new_image, rgb)
            new_image = self.srednja_horizontalna(new_image, rgb)

            return new_image
        
        elif(number == 9):
            new_image = self.desna_vertikalna(new_image, rgb)
            new_image = self.srednja_horizontalna(new_image, rgb)
            new_image = self.gornja_horizontalna(new_image, rgb)
            new_image = self.donja_horizontalna(new_image, rgb)
            new_image = self.gornja_leva(new_image, rgb)
            
            return new_image
        
    def desna_vertikalna(self,new_image, rgb):
        height = len(new_image)
        width = len(new_image[0])
        
        for i in range(height):
            new_image[i, width-1] = rgb
                
        return new_image
                
    def leva_vertikalna(self,new_image, rgb):
        height = len(new_image)
        width = len(new_image[0])
        
        for i in range(height):
            new_image[i, 0] = rgb
                
        return new_image

                
    def donja_horizontalna(self,new_image, rgb):
        height = len(new_image)
        width = len(new_image[0])
        
        for i in range(width):
            new_image[height-1, i] = rgb
                
        return new_image
                
    def gornja_horizontalna(self,new_image, rgb):
        height = len(new_image)
        width = len(new_image[0])
        
        for i in range(width):
            new_image[0, i] = rgb
                
        return new_image
                
    def srednja_horizontalna(self,new_image, rgb):
        height = len(new_image)
        width = len(new_image[0])
        
        srednji_red = height//2
        for i in range(width):
            new_image[srednji_red,i] = rgb
            
        return new_image
    
    def gornja_leva(self,new_image, rgb):
        height = len(new_image)
        width = len(new_image[0])
        
        srednji_red = height//2
        for i in range(srednji_red+1):
            new_image[i,0] = rgb
            
        return new_image
            
    def donja_leva(self,new_image, rgb):
        height = len(new_image)
        width = len(new_image[0])
        
        srednji_red = height//2
        for i in range(srednji_red, height):
            new_image[i,0] = rgb
            
        return new_image
            
    def gornja_desna(self,new_image, rgb):
        height = len(new_image)
        width = len(new_image[0])

        srednji_red = height//2
        for i in range(srednji_red+1):
            new_image[i,width-1] = rgb

        return new_image
    
    def donja_desna(self,new_image, rgb):
        height = len(new_image)
        width = len(new_image[0])
        
        srednji_red = height//2
        for i in range(srednji_red, height):
            new_image[i,width-1] = rgb
            
        return new_image

