#!/usr/bin/env python
# coding: utf-8

# In[2]:


import gym
import matplotlib.pyplot as plt
import random
import numpy as np
import os
from collections import deque, OrderedDict
import cv2
from copy import deepcopy
from missiles_Tracker import MissilesTracker
from drawObjectTracked import DrawObjectTracked
from otherObjects import OtherObjects

# In[3]:


class SpaceInvaders:
    def __init__(self):
        self.env = gym.make('SpaceInvaders-v4')#, render_mode="human")#, new_step_api=True)
        self.last_frame = None
        self.last_frame_number = 0
        self.frame_height, self.frame_width = 210, 160
        
        self.last_alien_rectangle_coords = None
        self.second_last_alien_rectangle_coords = None
        self.arrow = None
        
        self.yellow_aliens_looks = self.read_yellow_alliens_images()
        self.bot_look, self.purple_alien_look, self.barrier_initial_look = self.read_other_objects_images()
        
        self.yellow_aliens_filters = self.read_yellow_alliens_filters()
        self.bot_filter, self.purple_filter = self.read_other_filters()
        
        self.first_row_alien_position = 31
        self.first_row_alien_type = 0
        self.current_alien_look = None
        
        self.current_bot_position = None
        
        self.aliens_distance_columns = 16
        self.alien_distance_rows = 18
        
        self.missiles = MissilesTracker()
        self.drawer = DrawObjectTracked()
        self.other_objects = OtherObjects(self.purple_alien_look, self.purple_filter, self.barrier_initial_look)
        
    # igra ne pocinje pre 129 frejma
    def fast_forward_start_position(self):
        while(self.last_frame_number<129):
            next_state, reward, done, info = self.env.step(0)
            self.last_frame_number = info["frame_number"]
            self.last_frame = next_state
            
    def act(self):
        akcija = np.random.choice([1,4,5])

        return akcija                
            
    def run(self):
        
        self.last_frame = self.env.reset()
        #self.fast_forward_start_position()
        
        ###video maker
        video_name = 'video.mp4'
        height, width, frames_per_second = 210,160,10

        video_writer = cv2.VideoWriter(video_name, fourcc=cv2.VideoWriter_fourcc(*"mp4v"),
                                       fps=float(frames_per_second), frameSize=(width, height), isColor=True)
        ####
        
        for i in range(10000):
            akcija = self.act()
            self.last_frame, reward, done, info = self.env.step(akcija)
            if(done):
                break
            
            self.last_frame_number = info["episode_frame_number"]
            curr_frame = self.last_frame_number
            
            img_to_plot = self.last_frame.copy()
            
            if(curr_frame%2==0):
                self.missiles.update_bot_missile(self.last_frame, curr_frame)
                
                for objectID, (centroid, frame_tmp) in self.missiles.bot_missile.items():
                    if(centroid[0]<8):
                        continue
                        
                    if(centroid[1]-6>=0):
                        x_tmp, y_tmp = (centroid[0]-3, centroid[1]-6)
                    else:
                        x_tmp, y_tmp = (centroid[0]-3, centroid[1]+6)
                    
                    tmp_field = self.last_frame[x_tmp:x_tmp+7, y_tmp:y_tmp+5]
                    img_to_plot[x_tmp:x_tmp+7, y_tmp:y_tmp+5] = self.drawer.draw_id(tmp_field,
                                                          objectID, rgb=[0,255,0])
                
                pred_mss = self.missiles.predict_unseen_alien_missiles(self.last_frame,curr_frame)
                
                for objectID, (centroid, frame_tmp) in pred_mss.items():
                    if(centroid[0]<8):
                        continue
                        
                    if(centroid[1]-6>=0):
                        x_tmp, y_tmp = (centroid[0]-3, centroid[1]-6)
                    else:
                        x_tmp, y_tmp = (centroid[0]-3, centroid[1]+6)
                    
                    tmp_field = self.last_frame[x_tmp:x_tmp+7, y_tmp:y_tmp+5]
                    img_to_plot[x_tmp:x_tmp+7, y_tmp:y_tmp+5] = self.drawer.draw_id(tmp_field,
                                                          objectID, rgb=[128,0,0])

            elif(curr_frame%2==1):
                self.missiles.update_alien_missiles(self.last_frame, curr_frame)

                for objectID, (centroid, frame_tmp) in self.missiles.alien_missiles.items():
                    if(centroid[0]<8):
                        continue
                        
                    if(centroid[1]-6>=0):
                        x_tmp, y_tmp = (centroid[0]-3, centroid[1]-6)
                    else:
                        x_tmp, y_tmp = (centroid[0]-3, centroid[1]+6)
                    
                    tmp_field = self.last_frame[x_tmp:x_tmp+7, y_tmp:y_tmp+5]
                    img_to_plot[x_tmp:x_tmp+7, y_tmp:y_tmp+5] = self.drawer.draw_id(tmp_field,
                                                          objectID)
                
                pred_mss = self.missiles.predict_unseen_bot_missiles(self.last_frame,curr_frame)
                for objectID, (centroid, frame_tmp) in pred_mss.items():
                    if(centroid[0]<8):
                        continue
                    
                    if(centroid[1]-6>=0):
                        x_tmp, y_tmp = (centroid[0]-3, centroid[1]-6)
                    else:
                        x_tmp, y_tmp = (centroid[0]-3, centroid[1]+6)
                    
                    tmp_field = self.last_frame[x_tmp:x_tmp+7, y_tmp:y_tmp+5]
                    img_to_plot[x_tmp:x_tmp+7, y_tmp:y_tmp+5] = self.drawer.draw_id(tmp_field,
                                                          objectID, rgb=[0,128,0])

            alive_aliens, (min_x, min_y, max_x, max_y) = self.find_aliens()
            
            if(self.last_alien_rectangle_coords is None):
                self.last_alien_rectangle_coords = (min_x, min_y, max_x, max_y)
            else:
                self.second_last_alien_rectangle_coords = self.last_alien_rectangle_coords
                self.last_alien_rectangle_coords = (min_x, min_y, max_x, max_y)
                
            tmp_last = self.last_alien_rectangle_coords
            tmp_second_last = self.second_last_alien_rectangle_coords
            
            if (tmp_last is not None and tmp_second_last is not None):
                if(tmp_last != tmp_second_last):
                    if(tmp_last[1] < tmp_second_last[1]):
                        self.arrow = "left"
                        
                    else:
                        self.arrow = "right"        
                        
            if(self.arrow is not None):
                x_tmp = self.first_row_alien_position - 7
                y_tmp = int(np.average([tmp_last[1],tmp_last[3]]))
                tmp_field = img_to_plot[x_tmp-3:x_tmp+4, y_tmp-4:y_tmp+5]
                
                if(self.arrow == "left"):
                    img_to_plot[x_tmp-3:x_tmp+4, y_tmp-4:y_tmp+5] = self.drawer.draw_left_arrow(tmp_field)
                else:
                    img_to_plot[x_tmp-3:x_tmp+4, y_tmp-4:y_tmp+5] = self.drawer.draw_right_arrow(tmp_field)
            
            for i in range(6):
                for j in range(6):
                    if(alive_aliens[i,j]==1):
                        x_tmp = min_x + i*18
                        y_tmp = min_y + j*16
                        tmp_field = img_to_plot[x_tmp-1:x_tmp+12, y_tmp-1:y_tmp+10]
                        img_to_plot[x_tmp-1:x_tmp+12, y_tmp-1:y_tmp+10] = \
                                            self.drawer.draw_box(tmp_field, rgb=[255,0,0])
            
            bot_pos = self.find_bot()
            
            if (bot_pos is not None):
                x_tmp, y_tmp = bot_pos
                tmp_field = img_to_plot[x_tmp-1:x_tmp+12, y_tmp-1:y_tmp+9]
                img_to_plot[x_tmp-1:x_tmp+12, y_tmp-1:y_tmp+9] = \
                                self.drawer.draw_box(tmp_field, rgb=[0,255,0])
                
            purple_pos = self.other_objects.find_purple(self.last_frame)
            
            if(purple_pos is not None):
                x_tmp, y_tmp = purple_pos
                if(y_tmp>0 and y_tmp<width-1):
                    tmp_field = img_to_plot[x_tmp-1:x_tmp+10, y_tmp-1:y_tmp+9]
                    img_to_plot[x_tmp-1:x_tmp+10, y_tmp-1:y_tmp+9] = \
                                    self.drawer.draw_box(tmp_field, rgb=[255,0,0])
            
            healths = self.other_objects.count_barrier_health(img_to_plot)
            if(len(healths)>0):
                imgs_tmp = self.plot_barrier_healths(healths, img_to_plot)
                y_centers = [46, 78, 110]
                
                for ind,yc in enumerate(y_centers):
                    img_to_plot[205-3:205+4, yc-9:yc+10] = imgs_tmp[ind]
                
            ###video maker
            frame = cv2.cvtColor(img_to_plot, cv2.COLOR_RGB2BGR)
            video_writer.write(frame)
            ###
            
        self.env.close()
        cv2.destroyAllWindows()
        video_writer.release()
            
    def plot_barrier_healths(self, healths, image):
        y_centers = [46, 78, 110]
        
        base_image = image[205-3:205+4, 46-9:46+10].copy()
        new_images = []
        
        for ind,yc in enumerate(y_centers):
            
            tmp = base_image.copy()
            
            health_tmp = healths[ind]

            if(health_tmp==100):
                health_tmp-=1
                
            tmp_field1 = tmp[0:7, 0:5].copy()
            tmp_field2 = tmp[0:7, 6:11].copy()
            tmp_field3 = tmp[0:7, 12:19].copy()
            
            tmp_field2 = self.drawer.draw_id(tmp_field2, health_tmp%10, rgb=[0,0,255])
            health_tmp = health_tmp // 10
            tmp_field1 = self.drawer.draw_id(tmp_field1, health_tmp, rgb=[0,0,255])
            tmp_field3 = self.drawer.draw_percent(tmp_field3)
            
            tmp[0:7, 0:5] = tmp_field1
            tmp[0:7, 6:11] = tmp_field2
            tmp[0:7, 12:19] = tmp_field3
            
            new_images.append(tmp)
        
        return new_images
            
    def read_yellow_alliens_images(self):
        yellow_aliens = np.empty((6,2,10,8,3), dtype="uint8")
        for file in os.listdir("./slike_objekata/slike zutih vanzemaljaca/"):
            if file.endswith(".npy"):
                with open("./slike_objekata/slike zutih vanzemaljaca/"+file, "rb") as f:
                    slika = np.load(f)

                row = file.find("_row")
                row = int(file[row-3])
                col = file.find("_look")
                col = int(file[col-3])
                
                yellow_aliens[row-1,col-1] = slika
        
        return yellow_aliens
    
    def read_other_objects_images(self):
        with open("./slike_objekata/ostale_slike/bot_look.npy", "rb") as f:
            our_bot = np.load(f)
        
        with open("./slike_objekata/ostale_slike/purple_alien.npy", "rb") as f:
            purple_alien = np.load(f)
            
        with open("./slike_objekata/ostale_slike/barrier_initial_look.npy", "rb") as f:
            barrier = np.load(f)
        
        return (our_bot, purple_alien, barrier)
    
    def read_yellow_alliens_filters(self):
        yellow_aliens = np.empty((6,2,10,8,3), dtype="uint8")
        for file in os.listdir("./slike_objekata/filteri_zutih_vanzemaljaca/"):
            if file.endswith(".npy"):
                with open("./slike_objekata/filteri_zutih_vanzemaljaca/"+file, "rb") as f:
                    slika = np.load(f)

                row = file.find("row")
                row = int(file[row+4])
                col = file.find("look")
                col = int(file[col+5])
                
                yellow_aliens[row-1,col-1] = slika
        
        return yellow_aliens
    
    def read_other_filters(self):
        with open("./slike_objekata/ostali_filteri/bot.npy", "rb") as f:
            our_bot = np.load(f)
        
        with open("./slike_objekata/ostali_filteri/purple.npy", "rb") as f:
            purple_alien = np.load(f)
        
        return (our_bot, purple_alien)
    
    def find_aliens(self):
        one_alien = self.find_one_alien_by_first_row_position(self.first_row_alien_position)
        
        if(one_alien[0]==(0,0)):
            found_position = False

            one_alien = self.find_one_alien_by_first_row_position(self.first_row_alien_position+10)
            if(one_alien[0]!=(0,0)):
                self.first_row_alien_position += 10
                found_position = True
            
            if(not found_position):
                one_alien = self.find_one_alien_by_first_row_position(self.first_row_alien_position+18)
                if(one_alien[0]!=(0,0)):
                    self.first_row_alien_position += 18
                    self.first_row_alien_type = one_alien[1][0]
                    self.current_alien_look = one_alien[1][1]
                    found_position = True
                    
            if(not found_position):
                for search_x in range(self.first_row_alien_position-1,self.frame_height-15-10):
                    one_alien = self.find_one_alien_by_first_row_position(search_x)
                    if(one_alien[0]==(0,0)):
                        found_position = True
                        self.first_row_alien_position = search_x
                        self.first_row_alien_type = one_alien[1][0]
                        self.current_alien_look = one_alien[1][1]
                        break
                    
        else:
            self.current_alien_look = one_alien[1][1]
            self.first_row_alien_type = one_alien[1][0]
        
        alive_aliens, (min_x, min_y, max_x, max_y) = self.find_all_aliens_from_one_alien(one_alien)
        
        return alive_aliens, (min_x, min_y, max_x, max_y)
        
    def find_bot_using_current_position(self):
        if(self.current_bot_position is None):
            bot_pos = self.find_bot()
            if(bot_pos is not None):
                self.current_bot_position = bot_pos
                return bot_pos
            else:
                raise ValueError("Warning: Bot cannot be found!")
                
        # dodati da trazi na osnovu pozicije bota brze
            
    def find_bot(self):
        found_bot = False
        bot_pos = None
        
        for y in range(34,117):
            if(np.array_equal(self.last_frame[185:195, y:y+7]*self.bot_filter, self.bot_look)):
                found_bot = True
                bot_pos = (185,y)
                break
                
        if(found_bot):
            return bot_pos
        else:
            return None

    def find_one_alien_by_first_row_position(self, first_row_position):
        found_one = False
        found_coords = (0,0)
        alien_row_type_0to6 = None
        
        for i in range(self.frame_width-8):
            for alien_row in range(6):
                for alien_look in range(2):
                    current_look = self.last_frame[first_row_position:first_row_position+10,i:i+8]

                    if (np.array_equal(current_look * self.yellow_aliens_filters[alien_row,alien_look], 
                        self.yellow_aliens_looks[alien_row,alien_look])):
                        
                        found_one = True
                        found_coords = (first_row_position,i)
                        alien_type_row0to6_look0to1 = (alien_row, alien_look)
                        break
                if(found_one):
                    break
            if(found_one):
                break
                
        if(not found_one):
            return (found_coords, None)
        
        return (found_coords, alien_type_row0to6_look0to1)
        
    def find_all_aliens_from_one_alien(self, alien):
        first_alien_position, current_alien_type = alien
        
        x0, y0 = first_alien_position
        current_alien_row, current_alien_look = current_alien_type
        current_alien_row -= 1
        aliens_found = []
        wrote = False
        
        for x in range(x0, x0+18*(5-alien[1][0])+1,18):
            current_alien_row += 1
            if (x>self.frame_height-18):
                break
            
            aliens_found_in_row = 0
            
            for y in range(y0,y0+16*5+1,16):
                if(y>self.frame_width-16):
                    break

                if(np.array_equal(self.last_frame[x:x+10, y:y+8] * 
                   self.yellow_aliens_filters[current_alien_row, current_alien_look],
                   self.yellow_aliens_looks[current_alien_row, current_alien_look])):
                    
                    aliens_found.append(((x,y),(current_alien_row,current_alien_look)))
                    aliens_found_in_row += 1
                    
            if(aliens_found_in_row==6):
                continue

            for y in range(y0-16,y0-16*5-1,-16):
                if(y<0):
                    break

                if(np.array_equal(self.last_frame[x:x+10, y:y+8] * 
                   self.yellow_aliens_filters[current_alien_row, current_alien_look],
                   self.yellow_aliens_looks[current_alien_row, current_alien_look])):

                    aliens_found.append(((x,y),(current_alien_row,current_alien_look)))

                if(aliens_found_in_row==6):
                    break
            
        aliens_alive = np.zeros((6,6), dtype="uint8")
        
        min_x = max_x =  aliens_found[0][0][0]
        min_y = max_y = aliens_found[0][0][1]
        
        for a in aliens_found:
            tmp_x, tmp_y = a[0]
            
            if(tmp_x < min_x):
                min_x = tmp_x
                
            if(tmp_y < min_y):
                min_y = tmp_y
                
            if( tmp_x > max_x):
                max_x = tmp_x
                
            if(tmp_y > max_y):
                max_y = tmp_y
        
        for a in aliens_found:
            cur_x, cur_y = a[0]
            matrix_x_position = int((cur_x - min_x)/18)
            matrix_y_position = int((cur_y - min_y)/16)
            aliens_alive[matrix_x_position, matrix_y_position] = 1
            
        return (aliens_alive, (min_x, min_y, max_x, max_y))


# In[ ]:




