#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Originally based off of the 'Build a Chrome Dino Game AI Model with Python' tutorial by Nicholas Renotte
# Link: https://www.youtube.com/watch?v=vahwuupy81A


# In[ ]:


# Installing Dependencies
# To avoid the zsh not found error, use backslashes before special characters such as '?', '[', '=' and others


# In[ ]:


# Reinforcement learning framework, cousin to sklearn
#!pip3 install stable-baselines3[extra] protobuf==3.20.* # Regular Installation
#!pip3 install stable-baselines3\[extra\] protobuf\=\=3.20.\* # ZSH Installation

#!pip3 install gym
#!pip3 install mss
#!pip install opencv-python

#!pip3 install pydirectinput # For Windows
#!pip3 install pyautogui # Works fine on macOS directly & browser based games like 'Dino Game'

#!pip3 install pynput
#!pip3 install python-libxdo

# Multiple Dependencies including Python 3.6+, Python Imaging Library or Pillow and Google Tesseract OCR
# Google Tesseract OCR has binaries for Ubuntu & Windows, for Mac and ARM M1 chips, compilling needed
#!pip3 install pytesseract


# In[ ]:


#!pip3 list


# In[ ]:


# Importing Dependencies

# Enables in-game captured frame processing
import cv2

# Transformational framework
import numpy as np

# File path managements
import os

# For sending in commands
#import pydirectinput # For Windows

# On macOS, will request accessibility access from terminal
# Security Preferences -> Security & Privacy -> Privacy -> Accessibility -> Allow Terminal
import pyautogui

# For extracting the game over text component via OCR
import pytesseract

# For pausing
import time

# Environment components
from gym import Env
from gym.spaces import Box
from gym.spaces import Discrete

# For visualizing captured in game screen frames
from matplotlib import pyplot as plt

# For screen capturing, faster than opencv
from mss import mss

from stable_baselines3 import DQN

# For checking whether a valid environment is up and running whilst training the AI model
from stable_baselines3.common import env_checker

# For saving models
from stable_baselines3.common.callbacks import BaseCallback

# Many video games utilize Direct X, which essentially bypasses the OS and will not register virtual key press
# inputs, instead requiring direct key press inputs, meaning PyAutoGui might not work inside the game window
#from pynput.keyboard import Key
#from pynput.keyboard import Controller


# In[ ]:


# Building A Custom Game Environment
class WebGame(Env):
    
    # Sets up the environment action and observation shapes
    def __init__(self):
        # Subclasses the model
        super().__init__()
        
        # Sets up the observation spaces
        self.observation_space = Box(low = 0, high = 255, shape = (1, 83, 100), dtype = np.uint8)
        
        # 3 Different Possible Actions: Jump, Duck & No Action
        self.action_space = Discrete(3)
        
        # Defines the extraction parameters for the game
        self.screen_capture = mss()
        self.game_location = {'top': 300, 'left': 0, 'width': 600, 'height': 500}
        self.game_over_location = {'top': 405, 'left': 630, 'width': 660, 'height': 70}
    
    # Carries out an action and actually does something within the game
    def step(self, action):
        # Action Keys:
        # 0 - Spacebar (Jump)
        # 1 - Down (Duck)
        # 2 - No Action
        action_map = {
            0: 'space',
            1: 'down',
            2: 'no_operation'
        }
        
        if(action != 2):
            #pydirectinput.press(action_map[action]) # For Windows
            pyautogui.press(action_map[action]) # For Non-Windows
        
        # Checks whether the game has ended
        game_over, game_over_screen_capture = self.get_done()
        
        # Retrieves the new observation
        new_observation = self.get_observation()
        
        # Rewards the AI model's agent
        # As the game in this context is Dino Game, a reward is given for every frame the game has not reached
        # the game over status, therefore the longer the Dino survives and the game runs, the greater the reward
        reward = 1
        
        # Information Dictionary
        # Required by stable baselines, even if unnecessary for the context and left empty
        information_dictionary = {}
        
        # Return order very important and must not change
        return new_observation, reward, game_over, information_dictionary
    
    # Visualizes the game
    def render(self):
        # Displays the pre-processed in game frame via screen capture
        cv2.imshow('Game', np.array(self.screen_capture.grab(self.game_location))[:, :, :3])
        
        # Giving ourselves time for the image to actually render
        # If the 'q' key is pressed on the keyboard then the frame will be closed
        if(cv2.waitKey(1) & 0xFF == ord('q')):
           self.close()
    
    # Restarts the game
    def reset(self):
        
        # 'Dino Game' restarts after a game over with a simple click anywhere on the screen
        
        # For Windows
        #pydirectinput.click(x = 150 , y = 150)
        #pydirectinput.press('space')
        
         # For Non-Windows
        pyautogui.click(x = 150 , y = 150)
        pyautogui.press('space')
        
        return self.get_observation()
    
    # Closes down the observation
    def close(self):
        
        # Will close the render function's windows
        cv2.destroyAllWindows()
    
    # Retrieves the desired portion of the observation
    def get_observation(self):
        
        # Computer Vision Pre-processing
        # Grabs a raw initial screen capture of the game using mss, wrapping it into an array via numpy to
        # obtain the pixel values, and finally choosing only the 1st 3 color channels
        raw_screen_capture = np.array(self.screen_capture.grab(self.game_location))[:, :, :3].astype(np.uint8)
        
        # Gravyscaling
        grayscaled_screen_capture = cv2.cvtColor(raw_screen_capture, cv2.COLOR_BGR2GRAY)
        
        # Resizing
        resized_grayscaled_screen_capture = cv2.resize(grayscaled_screen_capture, (100, 83))
        
        # Reformts the shape of the array to have channels first, as this is how PyTorch and stable baselines
        # needs said format to be in to properly function
        preprocessed_screen_capture = np.reshape(resized_grayscaled_screen_capture, (1, 83, 100))
        
        return preprocessed_screen_capture
    
    # Retrieves the game over text using OCR
    def get_done(self):
        
        # Retrieves the game over screen
        game_over_screen_capture = np.array(self.screen_capture.grab(self.game_over_location))[:, :, :3].astype(np.uint8)
        
        # Valid 'Game Over' text
        # Utilizes 'GAME' and 'GAHE' as sometimes it mixes up the two, but should still be able to recognize
        # that the 'player' in this situation has lost and now must either quit or restart
        game_over_strings = ['GAME', 'GAHE']
        
        game_over = False
        
        # Carries out the actual optical character recognition, trying to extract any text from the given
        # observation space into a string, and then comparing said string to whether or not it is the word 'GAME'
        # Could otherwise use image recognition if the 'Game Over' portion does not change
        #result = pytesseract.image_to_string(game_over_screen_capture)[ : 4] # Google OCR not yet installed
        
        # Temporary Setup
        result = 'GAME'
        
        # If True, then the training episode for the AI model will end
        if(result in game_over_strings):
            game_over = True
        
        return game_over, game_over_screen_capture
    


# In[ ]:


game_environment = WebGame()


# In[ ]:


# Rendering
# Opens a new window displaying what the computer is looking at and playing
# May have to check differences on Windows, Ubuntu and macOS due to how new windows are handled
game_environment.render()


# In[ ]:


# Resetting
game_environment.reset()


# In[ ]:


game_environment.close()


# In[ ]:


# For checking and testing purposes

# Observation Space

# Screen capture of the entire game
#plt.imshow(game_environment.observation_space.sample()[0])
#game_environment.get_observation()
#plt.imshow(game_environment.get_observation())
#plt.imshow(game_environment.get_observation()[0])
#plt.imshow(cv2.cvtColor(game_environment.get_observation()[0], cv2.COLOR_BGR2RGB))
# game_environment.get_observation().shape

# Screen capture of the game over text within the game
#plt.imshow(game_environment.get_done())
#np.array(game_environment.get_done()).shape

# Action Space
#game_environment.action_space.sample()
#game_environment.observation_space.sample()

# Game Over
#game_over, game_over_screen_capture = game_environment.get_done()
#plt.imshow(game_over)
#print(game_over)


# In[ ]:


# Testing The Environment


# In[ ]:


testing_environment = WebGame()


# In[ ]:


testing_observation = testing_environment.get_observation()


# In[ ]:


plt.imshow(cv2.cvtColor(testing_observation[0], cv2.COLOR_BGR2RGB))


# In[ ]:


game_over, game_over_screen_capture = testing_environment.get_done()
plt.imshow(game_over)
print(game_over)


# In[ ]:


# What pytesseract reads from the 'GAME OVER' screen capture observation
#pytesseract.image_to_string(came_over_screen_capture)
#pytesseract.image_to_string(came_over_screen_capture)[:4]


# In[ ]:


# Plays 10 games whilst taking random actions
for episode in range(10):
    observation = game_environment.reset()
    game_over = False
    total_reward = 0
    
    while(not game_over):
        observation, reward, game_over, information = game_environment.step(game_environment.action_space.sample())
        total_reward += reward
        
    print(f'Total Reward: {total_reward} - Episode: {episode}')


# In[ ]:


# Training AI Model


# In[ ]:


# Checks whether or not the environment is valid
env_checker.check_env(game_environment)


# In[ ]:


class TrainAndLoggingCallback(BaseCallback):
    
    def __init__(self, check_freq, save_path, verbose = 1):
        super(TrainAndLoggingCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.save_path = save_path
        
    def _init_callback(self):
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok = True)
            
    def _on_step(self):
        if self.n_calls % self.check_freq == 0:
            model_path = os.path.join(self.save_path, 'best_model_{}'.format(self.n_calls))
            self.model.save(model_path)
        
        return True


# In[ ]:


CHECKPOINT_DIRECTORY = '../Models/Training/'
LOG_DIRECTORY = '../Models/Logs/'


# In[ ]:


# Saves the model every 'x' number of steps / frames
callback = TrainAndLoggingCallback(check_freq = 1000, save_path = CHECKPOINT_DIRECTORY)


# In[ ]:


# Building & Training The Deep Q-Network (DQN) Model


# In[ ]:


# 'buffer_size': Adjust accordignly to how much RAM is available upon the machine running the training
# 'learning_start': Start learning after the first 1000 frames / steps
dino_game_model = DQN('CnnPolicy', game_environment, tensorboard_log = LOG_DIRECTORY, 
                      verbose = 1, buffer_size = 1200000, learning_starts = 1000)


# In[ ]:


# Kicks off the model AI's training
# 'total_timesteps': Equivalent to epochs in tensforflow, how low to train essentially
# 88,000 steps takes all night
dino_game_model.learn(total_timesteps = 5000, callback = callback)


# In[ ]:


# Testing The Model


# In[ ]:


dino_game_model = DQN.load('../Models/dinoGameModel')


# In[ ]:


# Plays 10 games utilizing the model's predictions
for episode in range(10):
    observation = game_environment.reset()
    game_over = False
    total_reward = 0
    
    while(not game_over):
        action, _ = dino_game_model.predict(observation)
        observation, reward, game_over, information = game_environment.step(int(action))
        total_reward += reward
        
    print(f'Total Reward: {total_reward} - Episode: {episode}')

