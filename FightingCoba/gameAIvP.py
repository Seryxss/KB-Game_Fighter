import pygame, sys
from pygame import mixer
from FighterPvP import FighterPvP
from FighterPvAI import FighterPvAI
from FighterPvAIRL import FighterPvAIRL
from FighterAIRLvPlayer import FighterAIRLvPlayer
from button import Button
import random
from enum import Enum
from collections import namedtuple
import numpy as np

mixer.init()
pygame.init()


FPS = 80

#define colours
RED = (255, 0, 0)
YELLOW = (255, 255, 0)  
WHITE = (255, 255, 255)

   
# class ActionLists(Enum):
#   IDLE = 0
#   LEFT = 1
#   RIGHT = 2
#   JUMP = 3
#   CROUCH = 4
#   LP = 5
#   HP = 6
#   LK = 7
#   HK = 8
#   CLP = 9
#   CHP = 10
#   CLK = 11
#   CHK = 12
#   CRLP = 13
#   CRHP = 14
#   CRLK = 15
#   CRHK = 16
#   JLP = 17
#   JHP = 18
#   JLK = 19
#   JHK = 20
#   S1 = 21
#   S2 = 22

class ActionLists(Enum):
  IDLE = 0
  W = 1
  A = 2
  S = 3
  D = 4
  R = 5
  T = 6
  F = 7
  G = 8
  C = 9
  V = 10
  WR = 11
  WT = 12
  WF = 13
  WG = 14
  SR = 15
  ST = 16
  SF = 17
  SG = 18


Position = namedtuple('Position', 'x, y, x2, y2')

# class ActionLists(list):
   

class fighterGameAIvP:
   
  def __init__(self) -> None:
    self.SCREEN_WIDTH = 1000
    self.SCREEN_HEIGHT = 600
    self.screen = pygame.display.set_mode((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
    pygame.display.set_caption("Brawler")
    
    #create game window
    self.screen_info = pygame.display.Info()
    self.screen_width = self.screen_info.current_w

    #set framerate
    self.clock = pygame.time.Clock()

    #define game variables
    self.last_count_update = pygame.time.get_ticks()
    self.round_over = False
    self.ROUND_OVER_COOLDOWN = 4000

    #define fighter variablesksssssssssssssssssssssssssssssssssssssssssssssssssssssssssss
    self.WARRIOR_SIZE = 250
    self.WARRIOR_SCALE = 2
    self.WARRIOR_OFFSET = [110, 120]
    self.WARRIOR_DATA = [self.WARRIOR_SIZE, self.WARRIOR_SCALE, self.WARRIOR_OFFSET]
    self.WARRIOR2_SIZE = 250
    self.WARRIOR2_SCALE = 2
    self.WARRIOR2_OFFSET = [110, 120]
    self.WARRIOR2_DATA = [self.WARRIOR2_SIZE, self.WARRIOR2_SCALE, self.WARRIOR2_OFFSET]

    #load music and sounds
    pygame.mixer.music.load("assets/audio/music.mp3")
    pygame.mixer.music.set_volume(0.5)
    pygame.mixer.music.play(-1, 0.0, 5000)
    self.sword_fx = pygame.mixer.Sound("assets/audio/sword.wav")
    self.sword_fx.set_volume(0.5)
    self.magic_fx = pygame.mixer.Sound("assets/audio/magic.wav")
    self.magic_fx.set_volume(0.75)

    #load aaaaaaaaaround image
    self.bg_image = pygame.image.load("assets/images/background/background.jpg").convert_alpha()

    self.victory_img = pygame.image.load("assets/images/icons/victory.png").convert_alpha()

    #load spritesheets
    # warrior_sheet = pygame.image.load("assets/images/warrior/Sprites/ryu.png").convert_alpha()
    # warrior2_sheet = pygame.image.load("assets/images/warrior/Sprites/ryu.png").convert_alpha()
    self.warrior_sheet = pygame.image.load("assets/images/warrior/Sprites/spriteRyuBiru.png").convert_alpha()
    self.warrior2_sheet = pygame.image.load("assets/images/warrior/Sprites/spriteRyuMerah.png").convert_alpha()
    # wizard_sheet = pygame.image.load("assets/images/wizard/Sprites/wizard.png").convert_alpha()

    #load vicory image
    self.victory_img = pygame.image.load("assets/images/icons/victory.png").convert_alpha()

    #define number of steps in each animation
    # WARRIOR_ANIMATION_STEPS = [1,1,1,11,14,9,11,1]
    # WARRIOR2_ANIMATION_STEPS = [1,1,1,11,14,9,13,1]

    self.WARRIOR_ANIMATION_STEPS = [20,25,12,18,15,1,10,33,18,31,12,34,15,34,11,37,11,34,8,14,20,26,46,33,25,1,1,22,13]
    self.WARRIOR2_ANIMATION_STEPS = [20,25,12,18,15,1,10,33,18,31,12,34,15,34,11,37,11,34,8,14,20,26,46,33,25,1,1,22,13]


    #define font
    self.count_font = pygame.font.Font("assets/fonts/turok.ttf", 80)
    self.score_font = pygame.font.Font("assets/fonts/turok.ttf", 30)

    self.resetGame()

  def resetGame(self):
    self.round_over = False
    self.action = ActionLists.IDLE

    self.fighter_1 = FighterAIRLvPlayer(1, 300, 330, False, self.WARRIOR_DATA, self.warrior_sheet, self.WARRIOR_ANIMATION_STEPS, self.sword_fx, self.screen_width)
    self.fighter_2 = FighterAIRLvPlayer(2, 650, 330, True, self.WARRIOR2_DATA, self.warrior2_sheet, self.WARRIOR2_ANIMATION_STEPS, self.magic_fx, self.screen_width)
    self.fighter_1_last_health = self.fighter_1.health
    self.fighter_2_last_health = self.fighter_2.health

    self.positionSelf = Position(self.fighter_1.rect.x, self.fighter_1.rect.y, self.fighter_1.rect.width, self.fighter_1.rect.height)
    self.positionEnemy = Position(self.fighter_2.rect.x, self.fighter_2.rect.y, self.fighter_2.rect.width, self.fighter_2.rect.height)

    self.score = 0

  def play_step(self, action):
    self.clock.tick(FPS)
    # self.draw_text("P1: " + str(score[0]), self.score_font, RED, 20, 60)
    # self.draw_text("P2: " + str(score[1]), self.score_font, RED, 580, 60)

    # 1. collect user input
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            quit()
    
    self.action = action
    # print(self.action)
    
    self.fighter_1.move(self.SCREEN_WIDTH, self.SCREEN_HEIGHT, self.screen, self.fighter_2, self.round_over, self.action)
    self.fighter_2.move(self.SCREEN_WIDTH, self.SCREEN_HEIGHT, self.screen, self.fighter_1, self.round_over, None)

     #update fighters
    self.fighter_1.update()
    self.fighter_2.update()

    self.positionSelf = Position(self.fighter_1.rect.x, self.fighter_1.rect.y, self.fighter_1.rect.width, self.fighter_1.rect.height)
    self.positionEnemy = Position(self.fighter_2.rect.x, self.fighter_2.rect.y, self.fighter_2.rect.width, self.fighter_2.rect.height)
    
    reward = 0
    game_over = False

    if self.round_over == False:
      if self.fighter_1.alive == False:
        game_over = True
        self.round_over = True
        self.round_over_time = pygame.time.get_ticks()
        reward -= 300
        self.score -= 300
        return reward, game_over, self.score
      elif self.fighter_2.alive == False:
        game_over = True
        self.round_over = True
        self.round_over_time = pygame.time.get_ticks()
        reward += 300
        self.score += 300
    else:
      self.screen.blit(self.victory_img, (360, 150))
      if pygame.time.get_ticks() - self.round_over_time > self.ROUND_OVER_COOLDOWN:
        self.round_over = False
        # self.intro_count = 3
    
    if self.fighter_1.health < self.fighter_1_last_health:
       temp = self.fighter_1_last_health - self.fighter_1.health
       reward = reward + 3 if temp < 5 else reward - temp # blocking 
       self.score = self.score + 3 if temp < 5 else self.score - temp 
       self.fighter_1_last_health = self.fighter_1.health

    if self.fighter_2.health < self.fighter_2_last_health:
       reward += self.fighter_2_last_health - self.fighter_2.health
       self.score += self.fighter_2_last_health - self.fighter_2.health
       self.fighter_2_last_health = self.fighter_2.health

    self.draw_bg()

    #show player stats
    self.draw_health_bar(self.fighter_1.health, 20, 20)
    self.draw_health_bar(self.fighter_2.health, 580, 20)
    #draw fighters
    self.fighter_1.draw(self.screen)
    self.fighter_2.draw(self.screen)
    
    pygame.display.update()

    return reward, game_over, self.score

  def draw_text(self,text, font, text_col, x, y):
    self.img = font.render(text, True, text_col)
    self.screen.blit(self.img, (x, y))

  #function for drawing background
  def draw_bg(self):
    self.scaled_bg = pygame.transform.scale(self.bg_image, (self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
    self.screen.blit(self.scaled_bg, (0, 0))

  #function for drawing fighter health bars
  def draw_health_bar(self, health, x, y):
    self.ratio = health / 144
    pygame.draw.rect(self.screen, WHITE, (x - 2, y - 2, 404, 34))
    pygame.draw.rect(self.screen, RED, (x, y, 400, 30))
    pygame.draw.rect(self.screen, YELLOW, (x, y, 400 * self.ratio, 30))
