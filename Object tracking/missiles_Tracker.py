#!/usr/bin/env python
# coding: utf-8

# In[3]:


import gym
import matplotlib.pyplot as plt
import random
import numpy as np
import os
from collections import deque, OrderedDict
import cv2
from copy import deepcopy


# In[4]:


class MissilesTracker:
    def __init__(self):
        self.nextObjectID = 0
        self.bot_missile = OrderedDict()
        self.alien_missiles = OrderedDict()
        
        self.rgb_missile_value = np.array([142,142,142], dtype="uint8")
        self.bot_missile_speed = 2
        self.alien_missile_speed = 1
        
    def register(self, centroid, frame_number_spotted, bot=False):
        # when registering an object we use the next available object
        # ID to store the centroid
        if(self.nextObjectID>9):
            self.nextObjectID = 0

        for i in range(self.nextObjectID,10):
            if(self.nextObjectID in self.bot_missile or self.nextObjectID in self.alien_missiles):
                self.nextObjectID += 1
            else:
                break
            
        if(bot):
            self.bot_missile[self.nextObjectID] = (centroid, frame_number_spotted)
            self.nextObjectID += 1
        else:
            self.alien_missiles[self.nextObjectID] = (centroid, frame_number_spotted)
            self.nextObjectID += 1
        
        
    def deregister(self, objectID, bot=False):
        # to deregister an object ID we delete the object ID from
        # both of our respective dictionaries
        if(bot):
            del self.bot_missile[objectID]
        else:
            del self.alien_missiles[objectID]
            
    def predict_unseen_alien_missiles(self,frame_image, current_frame_number):
        new_missiles = OrderedDict()
        
        for ms_id, (centr, frame_num) in self.alien_missiles.items():
            x_tmp, y_tmp = self.predict_alien_missile_position((centr, frame_num), current_frame_number)
            new_missiles[ms_id] = ((x_tmp,y_tmp), current_frame_number)
        
        return new_missiles
    
    def predict_unseen_bot_missiles(self, frame_image, current_frame_number):
        new_missiles = OrderedDict()
        
        for ms_id, (centr, frame_num) in self.bot_missile.items():
            x_pr, y_pr = self.predict_bot_missile_position(centr, frame_num, current_frame_number)
            
            if(x_pr > 8):
                new_missiles[ms_id] = ((x_pr,y_pr), current_frame_number)
        
        return new_missiles
    
    def update_bot_missile(self, frame_image, current_frame_number):
        if(len(self.bot_missile) > 0):
            missile_found = self.find_bot_missile_by_prediction(frame_image, current_frame_number)
            
            for m_id in self.bot_missile:
                missile_id = m_id
            
            if(missile_found is None):       
                self.deregister(missile_id, bot=True)
            else:
                centroid_tmp = (int(np.average(missile_found[0])) , missile_found[1])
                self.bot_missile[missile_id] = (centroid_tmp, current_frame_number)

                return
                  
        pixel_found = self.find_bot_missile_pixel(frame_image)
        if pixel_found is None:
            return
            
        missile_found = self.locate_missile_by_pixel(frame_image, pixel_found)
        centroid_tmp = (int(np.average(missile_found[0])) , missile_found[1])
        self.register(centroid_tmp, current_frame_number, bot=True)
        
    def update_alien_missiles(self, frame_image, current_frame_number):
        all_missiles = self.find_alien_missiles(frame_image)
        #print("All missiles: ",all_missiles)
        paired_missiles = []
        za_dodati = []
        for am in all_missiles:
            (paired, miss_id) = self.pair_and_update_with_existing_alien_missile(am, current_frame_number)
            if(paired):
                paired_missiles.append(miss_id)
                continue
            else:
                (x_bot,x_top), y_tmp = am
                centroid_tmp = (int(np.average([x_bot,x_top])), y_tmp)
                za_dodati.append((centroid_tmp, current_frame_number))
        
        to_deregister = []
        for miss_id, miss_info in self.alien_missiles.items():
            if(miss_id not in paired_missiles):
                to_deregister.append(miss_id)
                
                
        for td in to_deregister:
            self.deregister(td)
            
        for zd in za_dodati:
            self.register(zd[0], zd[1])
                
    def pair_and_update_with_existing_alien_missile(self, missile, current_frame_number):
        
        for missile_id, missile_info in self.alien_missiles.items():
            (prediction_x, prediction_y) = self.predict_alien_missile_position(missile_info, current_frame_number)
            
            if(prediction_y == missile[1] and prediction_x<=missile[0][0]+2 and prediction_x>=missile[0][1]-2):
                    (x_bot,x_top), y_tmp = missile
                    centroid_tmp = (int(np.average([x_bot,x_top])), y_tmp)
                    self.alien_missiles[missile_id] = (centroid_tmp, current_frame_number)
                    return (True, missile_id)
            
        return (False, None)
    
    def find_alien_missiles(self, frame_image):
        all_pixels_found = []
        
        for x in range(0,210,4):
            for y in range(0,160):
                if(np.array_equal(frame_image[x,y], self.rgb_missile_value)):
                    missile_pixel_location = (x,y)
                    all_pixels_found.append(missile_pixel_location)
                
        all_missiles = set()
        
        for apf in all_pixels_found:
            missile_tmp = self.locate_missile_by_pixel(frame_image, apf)
            all_missiles.add(missile_tmp)
            
        return list(all_missiles)
    
    def predict_alien_missile_position(self, missile_info , current_frame_number):
        centroid, frame_spotted_before = missile_info
        prediction_x = centroid[0] + (current_frame_number-frame_spotted_before)
        
        return (prediction_x, centroid[1])
    
    def find_close_missile_to_predicted_if_exists(self, frame_image, predicted_location, all_missiles):
        for ind, am in enumerate(all_missiles):
            if(predicted_location[1]==am[1] and predicted_location[0] <= am[0][0] 
               and predicted_location[0] >= am[0][1]):
                return ind
            
            for i in range(1,4):
                if(predicted_location[1]==am[1] and predicted_location[0]+i <= am[0][0] 
                   and predicted_location[0]+i >= am[0][1]):
                    return ind
            
                if(predicted_location[1]==am[1] and predicted_location[0]-i <= am[0][0] 
                   and predicted_location[0]-i >= am[0][1]):
                    return ind
            
        return None
    
    def find_bot_missile_pixel(self, frame_image):
        found_missile = False
        missile_pixel_location = None
        
        for x in range(0,210,4):
            for y in range(34,123):
                if(np.array_equal(frame_image[x,y], self.rgb_missile_value)):
                    found_missile = True
                    missile_pixel_location = (x,y)
                if(found_missile):
                       break
                   
            if(found_missile):
                break
        
        if(found_missile):
            return missile_pixel_location
        else:
            return None
    
    def locate_missile_by_pixel(self, frame_image, pixel):
        
        top_pixel_of_missile = pixel[0]
        bottom_pixel_of_missile = pixel[0]
        
        for x in range(1,16):
            if(np.array_equal(frame_image[pixel[0]+x,pixel[1]], self.rgb_missile_value)):
                bottom_pixel_of_missile += 1
            else:
                break
                
        for x in range(1,16):
            if(np.array_equal(frame_image[pixel[0]-x,pixel[1]], self.rgb_missile_value)):
                top_pixel_of_missile -= 1
            else:
                break
                
        return ((bottom_pixel_of_missile, top_pixel_of_missile), pixel[1])
    
    def find_bot_missile_by_prediction(self, frame_image, current_frame_number):
        missile_id = next(iter(self.bot_missile))
        centroid, frame_spotted_before = self.bot_missile[missile_id]
        
        prediction_x = centroid[0] - (current_frame_number-frame_spotted_before)*2
        if(prediction_x<0):
            return None

        if (np.array_equal(frame_image[prediction_x, centroid[1]], self.rgb_missile_value)):
            missile = self.locate_missile_by_pixel(frame_image, (prediction_x, centroid[1]))
            return missile
        
        for i in range(16):
            if(np.array_equal(frame_image[prediction_x+i, centroid[1]], self.rgb_missile_value)):
                missile = self.locate_missile_by_pixel(frame_image, (prediction_x+i, centroid[1]))
                return missile
            
            if(prediction_x-i<0):             
                if(np.array_equal(frame_image[prediction_x-i, centroid[1]], self.rgb_missile_value)):
                    missile = self.locate_missile_by_pixel(frame_image, (prediction_x-i, centroid[1]))
                    return missile
                
    def predict_bot_missile_position(self, centr, frame_num_before, curr_frame_num):
        x, y = centr
        
        x_pr = x - (curr_frame_num - frame_num_before)*2
        
        return (x_pr,y)


# In[ ]:




