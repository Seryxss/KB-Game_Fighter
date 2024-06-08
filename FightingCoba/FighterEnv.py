import gym
from gym import spaces
import numpy as np
import pygame
from FighterAIRLvAIBT import FighterAIRLvAIBT

class FighterEnv(gym.Env):
    def __init__(self, warrior_data, warrior_sheet, warrior_animation_steps, warrior_sound, screen_width, warrior2_data, warrior2_sheet, warrior2_animation_steps, warrior2_sound, screen_height, screen):
        self.warrior_data = warrior_data
        self.warrior_sheet = warrior_sheet
        self.warrior_animation_steps = warrior_animation_steps
        self.warrior_sound = warrior_sound
        self.screen_width = screen_width
        self.warrior2_data = warrior2_data
        self.warrior2_sheet = warrior2_sheet
        self.warrior2_animation_steps = warrior2_animation_steps
        self.warrior2_sound = warrior2_sound
        self.screen = screen
        self.screen_height = screen_height
        # Define the observation space
        # Assuming the observation is a stack of 4 grayscale frames of size 84x84
        self.observation_space = spaces.Box(low=0, high=255, shape=(4, 84, 84), dtype=np.uint8)

        # Define the action space
        # Assuming a discrete action space with 6 possible actions
        self.action_space = spaces.Discrete(6)

        # Initialize the game state
        self.game_state = None
        self.player1 = None
        self.player2 = None

    def reset(self):
        # Reset the game state
        self.game_state = initialize_game_state()
        self.player1 = FighterAIRLvAIBT(1, 300, 330, False, self.warrior_data, self.warrior_sheet, self.warrior_animation_steps, self.warrior_sound, self.screen_width, self.screen_height, self.screen)  # Initialize player 1 (AIRL agent)
        self.player2 = FighterAIRLvAIBT(2, 650, 330, True, self.warrior2_data, self.warrior2_sheet, self.warrior2_animation_steps, self.warrior2_sound, self.screen_width, self.screen_height, self.screen)  # Initialize player 2 (AIBT opponent)
        # Return the initial observation
        return self.get_observation()

    def step(self, action):
        # Execute the action for player 1 (AIRL agent)
        self.player1.move(self.player2, action)

        # Execute the AIBT opponent's move
        self.player2.move(self.player1, action)

        # Update the game state
        self.game_state = update_game_state(self.game_state, self.player1, self.player2)

        # Calculate the reward
        reward = calculate_reward(self.game_state, self.player1, self.player2)

        # Check if the episode is done
        done = is_episode_done(self.game_state)

        # Get the next observation
        observation = self.get_observation()

        return observation, reward, done, {}

    def get_observation(self):
        # Preprocess the game state to get the observation
        # (e.g., stack consecutive frames, convert to grayscale, resize)
        observation = preprocess_game_state(self.game_state)
        return observation

    def render(self, mode='human'):
        # Render the game state (optional)
        pass

def initialize_game_state():
    # Initialize the game state
    # You can define the initial state of the game here
    # For example, you can set the initial positions and health of the players
    initial_state = {
        'player1_pos': (300, 330),
        'player2_pos': (650, 330),
        'player1_health': 144,
        'player2_health': 144,
        # Add any other relevant state variables
    }
    return initial_state

def update_game_state(game_state, player1, player2):
    # Update the game state based on player actions
    # You can update the positions, health, and other state variables based on the player actions
    new_state = game_state.copy()
    new_state['player1_pos'] = player1.rect.topleft
    new_state['player2_pos'] = player2.rect.topleft
    new_state['player1_health'] = player1.health
    new_state['player2_health'] = player2.health
    # Update any other relevant state variables
    return new_state

def calculate_reward(game_state, player1, player2):
    # Calculate the reward for the AIRL agent
    # You can define the reward function based on the game state and player actions
    # For example, you can use the health difference between the players
    player1_health = game_state['player1_health']
    player2_health = game_state['player2_health']
    reward = player2_health - player1_health
    return reward

def is_episode_done(game_state):
    # Check if the episode is done (e.g., one player's health reaches 0)
    player1_health = game_state['player1_health']
    player2_health = game_state['player2_health']
    if player1_health <= 0 or player2_health <= 0:
        return True
    else:
        return False

def preprocess_game_state(game_state):
    # Preprocess the game state to get the observation
    # (e.g., stack consecutive frames, convert to grayscale, resize)
    # You can implement the preprocessing steps here
    # For simplicity, let's assume the observation is the game state itself
    observation = game_state
    return observation
