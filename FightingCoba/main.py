import pygame, sys
from pygame import mixer
from fighter import Fighter
from button import Button

mixer.init()
pygame.init()
SCREEN_WIDTH = 1000
SCREEN_HEIGHT = 600

screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Brawler")

def play():
  #create game window

  #set framerate
  clock = pygame.time.Clock()
  FPS = 40

  #define colours
  RED = (255, 0, 0)
  YELLOW = (255, 255, 0)  
  WHITE = (255, 255, 255)

  #define game variables
  intro_count = 2
  last_count_update = pygame.time.get_ticks()
  round_over = False
  ROUND_OVER_COOLDOWN = 2000

  #define fighter variablesksssssssssssssssssssssssssssssssssssssssssssssssssssssssssss
  WARRIOR_SIZE = 250
  WARRIOR_SCALE = 2
  WARRIOR_OFFSET = [110, 120]
  WARRIOR_DATA = [WARRIOR_SIZE, WARRIOR_SCALE, WARRIOR_OFFSET]
  WARRIOR2_SIZE = 250
  WARRIOR2_SCALE = 2
  WARRIOR2_OFFSET = [110, 120]
  WARRIOR2_DATA = [WARRIOR2_SIZE, WARRIOR2_SCALE, WARRIOR2_OFFSET]

  #load music and sounds
  pygame.mixer.music.load("assets/audio/music.mp3")
  pygame.mixer.music.set_volume(0.5)
  pygame.mixer.music.play(-1, 0.0, 5000)
  sword_fx = pygame.mixer.Sound("assets/audio/sword.wav")
  sword_fx.set_volume(0.5)
  magic_fx = pygame.mixer.Sound("assets/audio/magic.wav")
  magic_fx.set_volume(0.75)

  #load aaaaaaaaaround image
  bg_image = pygame.image.load("assets/images/background/background.jpg").convert_alpha()

  #load spritesheets
  # warrior_sheet = pygame.image.load("assets/images/warrior/Sprites/ryu.png").convert_alpha()
  # warrior2_sheet = pygame.image.load("assets/images/warrior/Sprites/ryu.png").convert_alpha()
  warrior_sheet = pygame.image.load("assets/images/warrior/Sprites/spriteRyuBiru.png").convert_alpha()
  warrior2_sheet = pygame.image.load("assets/images/warrior/Sprites/spriteRyuMerah.png").convert_alpha()
  # wizard_sheet = pygame.image.load("assets/images/wizard/Sprites/wizard.png").convert_alpha()

  #load vicory image
  victory_img = pygame.image.load("assets/images/icons/victory.png").convert_alpha()

  #define number of steps in each animation
  # WARRIOR_ANIMATION_STEPS = [1,1,1,11,14,9,11,1]
  # WARRIOR2_ANIMATION_STEPS = [1,1,1,11,14,9,13,1]

  WARRIOR_ANIMATION_STEPS = [20,25,12,18,15,1,10,33,18,31,12,34,15,34,11,37,11,34,8,14,20,26,46,33,25,1,1,22,13]
  WARRIOR2_ANIMATION_STEPS = [20,25,12,18,15,1,10,33,18,31,12,34,15,34,11,37,11,34,8,14,20,26,46,33,25,1,1,22,13]


  #define font
  count_font = pygame.font.Font("assets/fonts/turok.ttf", 80)
  score_font = pygame.font.Font("assets/fonts/turok.ttf", 30)

  #function for drawing text
  def draw_text(text, font, text_col, x, y):
    img = font.render(text, True, text_col)
    screen.blit(img, (x, y))

  #function for drawing background
  def draw_bg():
    scaled_bg = pygame.transform.scale(bg_image, (SCREEN_WIDTH, SCREEN_HEIGHT))
    screen.blit(scaled_bg, (0, 0))

  #function for drawing fighter health bars
  def draw_health_bar(health, x, y):
    ratio = health / 144
    pygame.draw.rect(screen, WHITE, (x - 2, y - 2, 404, 34))
    pygame.draw.rect(screen, RED, (x, y, 400, 30))
    pygame.draw.rect(screen, YELLOW, (x, y, 400 * ratio, 30))


  #create two instances of fighters
  fighter_1 = Fighter(1, 200, 330, False, WARRIOR_DATA, warrior_sheet, WARRIOR_ANIMATION_STEPS, sword_fx)
  fighter_2 = Fighter(2, 700, 330, True, WARRIOR2_DATA, warrior2_sheet, WARRIOR2_ANIMATION_STEPS, magic_fx)

  global plays
  global score
  while plays:
    clock.tick(FPS)
    #draw background
    draw_bg()

    #show player stats
    draw_health_bar(fighter_1.health, 20, 20)
    draw_health_bar(fighter_2.health, 580, 20)
    draw_text("P1: " + str(score[0]), score_font, RED, 20, 60)
    draw_text("P2: " + str(score[1]), score_font, RED, 580, 60)

    #update countdown
    if intro_count <= 0:  
      #move fighters
      fighter_1.move(SCREEN_WIDTH, SCREEN_HEIGHT, screen, fighter_2, round_over)
      fighter_2.move(SCREEN_WIDTH, SCREEN_HEIGHT, screen, fighter_1, round_over)
    else:
      #display count timer
      draw_text(str(intro_count), count_font, RED, SCREEN_WIDTH / 2, SCREEN_HEIGHT / 3)
      #update count timer
      if (pygame.time.get_ticks() - last_count_update) >= 1000:
        intro_count -= 1
        last_count_update = pygame.time.get_ticks()

    #update fighters
    fighter_1.update()
    fighter_2.update()

    #draw fighters
    fighter_1.draw(screen)
    fighter_2.draw(screen)

    #check for player defeat
    if round_over == False:
      if fighter_1.alive == False:
        score[1] += 1
        round_over = True
        round_over_time = pygame.time.get_ticks()
      elif fighter_2.alive == False:
        score[0] += 1
        round_over = True
        round_over_time = pygame.time.get_ticks()
    else:
      #display victory image
      screen.blit(victory_img, (360, 150))
      if pygame.time.get_ticks() - round_over_time > ROUND_OVER_COOLDOWN:
        round_over = False
        intro_count = 3
        fighter_1 = Fighter(1, 200, 330, False, WARRIOR_DATA, warrior_sheet, WARRIOR_ANIMATION_STEPS, sword_fx)
        fighter_2 = Fighter(2, 700, 330, True, WARRIOR2_DATA, warrior2_sheet, WARRIOR2_ANIMATION_STEPS, magic_fx)

    # print(score)
    if score[0] == 2 or score[1] == 2:
      plays = True
      runs = True
      score = [0,0]
      main_menu()
  
    #event handler
    for event in pygame.event.get():
      if event.type == pygame.QUIT:
        plays = False
        runs = False
        sys.exit()


    #update display
    pygame.display.update()

def get_font(size): # Returns Press-Start-2P in the desired size
    return pygame.font.Font("assets/font.ttf", size)

BG = pygame.image.load("assets/Background.png")
def main_menu():
    global runs
    while runs:
        screen.blit(BG, (0, 0))

        MENU_MOUSE_POS = pygame.mouse.get_pos()

        MENU_TEXT = get_font(100).render("PUNCH EM", True, "#b68f40")
        MENU_RECT = MENU_TEXT.get_rect(center=(510, 150))

        PLAY_BUTTON = Button(image=pygame.image.load("assets/Play Rect.png"), pos=(500, 330), 
                            text_input="PLAY", font=get_font(75), base_color="#d7fcd4", hovering_color="Black")
        # OPTIONS_BUTTON = Button(image=pygame.image.load("assets/Options Rect.png"), pos=(640, 400), 
        #                     text_input="OPTIONS", font=get_font(75), base_color="#d7fcd4", hovering_color="White")
        QUIT_BUTTON = Button(image=pygame.image.load("assets/Quit Rect.png"), pos=(500, 480), 
                            text_input="QUIT", font=get_font(75), base_color="#d7fcd4", hovering_color="Black")

        screen.blit(MENU_TEXT, MENU_RECT)

        for button in [PLAY_BUTTON, QUIT_BUTTON]:
            button.changeColor(MENU_MOUSE_POS)
            button.update(screen)
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                runs = False
                pygame.quit()
                sys.exit()
            if event.type == pygame.MOUSEBUTTONDOWN:
                if PLAY_BUTTON.checkForInput(MENU_MOUSE_POS):
                    play()
                # if OPTIONS_BUTTON.checkForInput(MENU_MOUSE_POS):
                #     options()
                if QUIT_BUTTON.checkForInput(MENU_MOUSE_POS):
                    pygame.quit()
                    sys.exit()

        pygame.display.update()

runs = True
plays = True
score = [0, 0]#player scores. [P1, P2]
# #game loop
# while runs and plays:
#   #display intro screen
#   for event in pygame.event.get():
#       if event.type == pygame.QUIT:
#         runs = False
#   play()

main_menu()

#exit pygame
pygame.quit()