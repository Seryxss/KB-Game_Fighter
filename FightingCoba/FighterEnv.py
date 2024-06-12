import numpy as np
import pygame
from FighterAIRLvAIBT import FighterAIRLvAIBT
import gym
from gym import spaces

class FighterEnv(gym.Env):
    def __init__(self, you, enemy):
        self.you = you
        self.enemy = enemy
        self.last_you_hp = you.health
        self.last_enemy_hp = enemy.health
        # Define the observation space
        # Assuming the observation is a stack of 4 grayscale frames of size 84x84
        # ini cuma buat literal liat secara visual btwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwww
        # self.observation_space = spaces.Box(low=0, high=255, shape=(4, 84, 84), dtype=np.uint8)

        # Define the action space
        # Assuming a discrete action space with 6 possible actions
        # self.action_space = spaces.Discrete(6)

        # Initialize the game state
        self.observation_space = spaces.Dict(
            {
                "you_rect": spaces.Box(low=np.array([0,0]), high=np.array(1000,600), shape=(2,), dtype=int),
                "enemy_rect": spaces.Box(low=np.array([0,0]), high=np.array(1000,600), shape=(2,), dtype=int),
                "you_state": spaces.Discrete(),
                "enemy_state": spaces.Discrete(),
            }
        )
        
        self.action_space = spaces.Discrete(banyak_move)
        
        ''' enable this kalo mau input shit di sini
        self.action_to_move = {
            0: # kiri
            1: # kanan
            2: # lompat
            3: # nunduk
            4: # move
        }
        '''
        
    def get_obs(self):
        return{ 
            "you_rect": ,
            "enemy_rect": ,
            "you_state": ,
            "enemy_state": ,      
        }
        
    def get_info(self):
        you_hp_diff = last_you_hp - you.health
        enemy_hp_diff = last_enemy_hp - enemy.health
        last_you_hp = you.health
        last_enemy_hp = enemy.health
        return{
            "you_hp_diff": you_hp_diff,
            "enemy_hp_diff" : enemy_hp_diff,
        }
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        observation = self.get_obs()
        info = self.get_info()
        
        return observation, info
    
    def step(self, action):        
        observation = self.get_obs()
        info = self.get_info()
        
        reward = 0
        reward +=    0 if info["you_hp_diff"] == 0   # neutral
                else -10 if info["you_hp_diff"] > 7  # hit
                else 1                               # block
                
        reward +=    0 if info["enemy_hp_diff"] == 0   # neutral
                else 10 if info["enemy_hp_diff"] > 7  # hit
                else 1                               # block
        
        reward +=     150 if enemy.health == 0
                else -150 if you.health == 0
        
        terminated = True if you.health == 0 or enemy.health == 0 else False
            
        
        return observation, reward, terminated, False, info
        
        

    