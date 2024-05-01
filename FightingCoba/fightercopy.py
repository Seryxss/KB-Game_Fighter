import pygame
import math

class Fighter():
  def __init__(self, player, x, y, flip, data, sprite_sheet, animation_steps, sound):
    self.player = player
    self.size = data[0]
    self.image_scale = data[1]
    self.offset = data[2]
    self.flip = flip
    self.animation_list = self.load_images(sprite_sheet, animation_steps)
    self.action = 0 
    self.frame_index = 0
    self.image = self.animation_list[self.action][self.frame_index]
    self.update_time = pygame.time.get_ticks()
    self.rect = pygame.Rect((x, y, 60, 180))
    self.vel_y = 0
    self.running = False
    self.backUp = False
    self.jump = False
    self.crouch = False
    self.attacking = False
    self.attack_type = 0
    self.distance = 0
    self.attack_cooldown = 0
    self.attack_sound = sound
    self.hit = False
    self.health = 144
    self.alive = True
    self.target = None


  def load_images(self, sprite_sheet, animation_steps):
    #extract images from spritesheet
    animation_list = []
    for y, animation in enumerate(animation_steps):
      temp_img_list = []
      for x in range(animation):
        temp_img = sprite_sheet.subsurface(x * self.size, y * self.size, self.size, self.size)
        temp_img_list.append(pygame.transform.scale(temp_img, (self.size * self.image_scale, self.size * self.image_scale)))
      animation_list.append(temp_img_list)
    return animation_list


  def move(self, screen_width, screen_height, surface, target, round_over):
    if self.target == None:
      self.target = target

    SPEED = 5
    GRAVITY = 2
    dx = 0
    dy = 0
    self.running = False
    self.backUp = False
    self.crouch = False
    self.attack_type = 0

    #get keypresses
    key = pygame.key.get_pressed()

    #can only perform other actions if not currently attacking
    if self.attacking == False and self.alive == True and round_over == False:
      distance = math.sqrt((self.rect.centerx - target.rect.centerx)**2 + (self.rect.centery - target.rect.centery)**2)
      #check player 1 controls
      if self.player == 1:
        #crouch
        if key[pygame.K_s] and self.crouch == False:
            self.crouch = True
        #movement
        if self.flip == False:
          if key[pygame.K_a] and self.crouch == False:
            dx = -SPEED
            self.backUp = True
          if key[pygame.K_d] and self.crouch == False:
            dx = SPEED
            self.running = True
        else:
          if key[pygame.K_a] and self.crouch == False:
            dx = -SPEED
            self.running = True
          if key[pygame.K_d] and self.crouch == False:
            dx = SPEED
            self.backUp = True
        #jump
        if key[pygame.K_w] and self.jump == False and self.crouch == False:
          self.vel_y = -30
          self.jump = True
        #attack punch
        if (key[pygame.K_r] or key[pygame.K_t]) and self.jump == False and self.crouch == False:
          # self.attack(target, surface)
          #determine which attack type was used
          if distance < 100:
            if key[pygame.K_r]:
              self.attack_type = 13
              self.attacking = True
              damage = 8
              self.attack(target, surface, self.attack_type, damage)
            if key[pygame.K_t]:
              self.attack_type = 14
              self.attacking = True
              damage = 28
              self.attack(target, surface, self.attack_type, damage)
          else :  
            if key[pygame.K_r]:
              self.attack_type = 1
              self.attacking = True
              damage = 8
              dx += SPEED
              self.attack(target, surface, self.attack_type, damage)
              # print(self.attack)
            if key[pygame.K_t]:
              self.attack_type = 2
              self.attacking = True
              damage = 28
              self.attack(target, surface, self.attack_type, damage)
              # print(self.attack)
          
          # self.attack(target, surface)
        #attack while jumping
        if (key[pygame.K_r] or key[pygame.K_t]) and self.jump == True:
          #self.attack(target, surface)
          #determine which attack type was used
          if key[pygame.K_r]:
            self.attack_type = 9
            self.attacking = True
            damage = 12
            self.attack(target, surface, self.attack_type, damage)
          if key[pygame.K_t]:
            self.attack_type = 10
            self.attacking = True
            damage = 26
            self.attack(target, surface, self.attack_type, damage)
        #attack while crouching
        if (key[pygame.K_r] or key[pygame.K_t]) and self.crouch == True:
          #self.attack(target, surface)
          #determine which attack type was used
          if key[pygame.K_r]:
            self.attack_type = 5
            self.attacking = True
            damage = 8
            self.attack(target, surface, self.attack_type, damage)
          if key[pygame.K_t]:
            self.attack_type = 6  
            self.attacking = True
            damage = 28
            self.attack(target, surface, self.attack_type, damage)  
        # special attack
        if key[pygame.K_c]:
          #self.attack(target, surface)
          self.attack_type = 17
          self.attacking = True
          damage = 24
          self.attack(target, surface, self.attack_type, damage)
        if key[pygame.K_v]:
          #self.attack(target, surface)
          self.attack_type = 18
          self.attacking = True
          damage = 32
          self.attack(target, surface, self.attack_type, damage)
        #attack kick
        if (key[pygame.K_f] or key[pygame.K_g]) and self.jump == False and self.crouch == False:
          #self.attack(target, surface)
          #determine which attack type was used
          if distance < 100:
            if key[pygame.K_f]:
              self.attack_type = 15
              self.attacking = True
              damage = 12
              self.attack(target, surface, self.attack_type, damage)
            if key[pygame.K_g]:
              self.attack_type = 16
              self.attacking = True
              damage = 28
              self.attack(target, surface, self.attack_type, damage)
          else : 
            if key[pygame.K_f]:
              self.attack_type = 3
              self.attacking = True
              damage = 14
              self.attack(target, surface, self.attack_type, damage)
            if key[pygame.K_g]:
              self.attack_type = 4
              self.attacking = True
              damage = 30
              self.attack(target, surface, self.attack_type, damage)
          #attack while jumping
        if (key[pygame.K_f] or key[pygame.K_g]) and self.jump == True:
          #self.attack(target, surface)
          #determine which attack type was used
          if key[pygame.K_f]:
            self.attack_type = 11
            self.attacking = True
            damage = 14
            self.attack(target, surface, self.attack_type, damage)
          if key[pygame.K_g]:
            self.attack_type = 12
            self.attacking = True
            damage = 30
            self.attack(target, surface, self.attack_type, damage)
        #attack while crouching
        if (key[pygame.K_f] or key[pygame.K_g]) and self.crouch == True:
          #self.attack(target, surface)
          #determine which attack type was used
          if key[pygame.K_f]:
            self.attack_type = 7
            self.attacking = True
            damage = 8
            self.attack(target, surface, self.attack_type, damage)
          if key[pygame.K_g]:
            self.attack_type = 8  
            self.attacking = True  
            damage = 26
            self.attack(target, surface, self.attack_type, damage)    
    
      #check player 2 controls
      if self.player == 2:
        distance = math.sqrt((self.rect.centerx - target.rect.centerx)**2 + (self.rect.centery - target.rect.centery)**2)
        #crouch
        if key[pygame.K_j] and self.crouch == False:
            self.crouch = True
        #movement
        if self.flip == True:
          if key[pygame.K_h] and self.crouch == False:
            dx = -SPEED
            self.running = True
          if key[pygame.K_k] and self.crouch == False:
            dx = SPEED
            self.backUp = True
        else:
          if key[pygame.K_h] and self.crouch == False:
            dx = -SPEED
            self.backUp = True
          if key[pygame.K_k] and self.crouch == False:
            dx = SPEED
            self.running = True
        #jump
        if key[pygame.K_u] and self.jump == False:
          self.vel_y = -30
          self.jump = True
        #attack punch
        if (key[pygame.K_o] or key[pygame.K_p]) and self.jump == False and self.crouch == False:
          #self.attack(target, surface)
          #determine which attack type was used
          if distance < 100:
            if key[pygame.K_o]:
              self.attack_type = 13
              self.attacking = True
              damage = 8
              self.attack(target, surface, self.attack_type, damage)
            if key[pygame.K_p]:
              self.attack_type = 14
              self.attacking = True
              damage = 28
              self.attack(target, surface, self.attack_type, damage)
          else :  
            if key[pygame.K_o]:
              self.attack_type = 1
              self.attacking = True
              damage = 12
              self.attack(target, surface, self.attack_type, damage)
            if key[pygame.K_p]:
              self.attack_type = 2
              self.attacking = True
              damage = 28
              self.attack(target, surface, self.attack_type, damage)
        #attack while jumping
        if (key[pygame.K_o] or key[pygame.K_p]) and self.jump == True:
          #self.attack(target, surface)
          #determine which attack type was used
          if key[pygame.K_o]:
            self.attack_type = 9
            self.attacking = True
            damage = 12
            self.attack(target, surface, self.attack_type, damage)
          if key[pygame.K_p]:
            self.attack_type = 10
            self.attacking = True
            damage = 26
            self.attack(target, surface, self.attack_type, damage)
        #attack while crouching
        if (key[pygame.K_o] or key[pygame.K_p]) and self.crouch == True:
          #self.attack(target, surface)
          #determine which attack type was used
          if key[pygame.K_o]:
            self.attack_type = 5
            self.attacking = True
            damage = 8  
            self.attack(target, surface, self.attack_type, damage)
          if key[pygame.K_p]:
            self.attack_type = 6  
            self.attacking = True
            damage = 28
            self.attack(target, surface, self.attack_type, damage)  
        # special attack
        if key[pygame.K_COMMA]:
          #self.attack(target, surface)
          self.attack_type = 17
          self.attacking = True
          damage = 24
          self.attack(target, surface, self.attack_type, damage)
        if key[pygame.K_PERIOD]:
          #self.attack(target, surface)
          self.attack_type = 18
          self.attacking = True
          damage = 32
          self.attack(target, surface, self.attack_type, damage)
        #attack kick
        if (key[pygame.K_l] or key[pygame.K_SEMICOLON]) and self.jump == False and self.crouch == False:
          #self.attack(target, surface)
          #determine which attack type was used
          if distance < 100:
            if key[pygame.K_l]:
              self.attack_type = 15
              self.attacking = True
              damage = 12
              self.attack(target, surface, self.attack_type, damage)
            if key[pygame.K_SEMICOLON]:
              self.attack_type = 16
              self.attacking = True
              damage = 28
              self.attack(target, surface, self.attack_type, damage)
          else : 
            if key[pygame.K_l]:
              self.attack_type = 3
              self.attacking = True
              damage = 14
              self.attack(target, surface, self.attack_type, damage)
            if key[pygame.K_SEMICOLON]:
              self.attack_type = 4
              self.attacking = True
              damage = 30
              self.attack(target, surface, self.attack_type, damage)
          #attack while jumping
        if (key[pygame.K_l] or key[pygame.K_SEMICOLON]) and self.jump == True:
          #self.attack(target, surface)
          #determine which attack type was used
          if key[pygame.K_l]:
            self.attack_type = 11
            self.attacking = True
            damage = 14
            self.attack(target, surface, self.attack_type, damage)
          if key[pygame.K_SEMICOLON]:
            self.attack_type = 12
            self.attacking = True
            damage = 30
            self.attack(target, surface, self.attack_type, damage)
        #attack while crouching
        if (key[pygame.K_l] or key[pygame.K_SEMICOLON]) and self.crouch == True:
          #self.attack(target, surface)
          #determine which attack type was used
          if key[pygame.K_l]:
            self.attack_type = 7
            self.attacking = True
            damage = 8
            self.attack(target, surface, self.attack_type, damage)
          if key[pygame.K_SEMICOLON]:
            self.attack_type = 8
            self.attacking = True
            damage = 26
            self.attack(target, surface, self.attack_type, damage)

    #apply gravity
    self.vel_y += GRAVITY
    dy += self.vel_y

    #ensure player stays on screen
    if self.rect.left + dx < 0:
      dx = -self.rect.left
    if self.rect.right + dx > screen_width:
      dx = screen_width - self.rect.right
    if self.rect.bottom + dy > screen_height - 110:
      self.vel_y = 0
      self.jump = False
      dy = screen_height - 110 - self.rect.bottom

    #ensure players face each other
    if target.rect.centerx > self.rect.centerx:
      self.flip = False
    else:
      self.flip = True

    #apply attack cooldown
    if self.attack_cooldown > 0:
      self.attack_cooldown -= 1

    #update player position
    self.rect.x += dx
    self.rect.y += dy


  #handle animation updates
  def update(self):
    #check what action the player is performing
    if self.health <= 0:
      self.health = 0
      self.alive = False
      self.update_action(1)#1:death
      self.target.update_action(24) # victor
    elif self.hit == True:
      self.update_action(2)#2:hit
    elif self.attacking == True:
      if self.attack_type == 1:
        self.update_action(6)#6:lp
      elif self.attack_type == 2:
        self.update_action(7)#7:hp
      elif self.attack_type == 3:
        self.update_action(8)#8:lk
      elif self.attack_type == 4:
        self.update_action(9)#9:hk
      elif self.attack_type == 5:
        self.update_action(14)#14:crouch lp
      elif self.attack_type == 6:
        self.update_action(15)#15:crouch hp
      elif self.attack_type == 7:
        self.update_action(16)#16:crouch lk
      elif self.attack_type == 8:
        self.update_action(17)#17:crouch hk
      elif self.attack_type == 9:
        self.update_action(18)#18:jump lp
      elif self.attack_type == 10:
        self.update_action(19)#19:jump hp
      elif self.attack_type == 11:
        self.update_action(20)#20:jump lk
      elif self.attack_type == 12:
        self.update_action(21)#21:jump hk
      elif self.attack_type == 13:
        self.update_action(10)#10:close lp
      elif self.attack_type == 14:
        self.update_action(11)#11:close hp
      elif self.attack_type == 15:
        self.update_action(12)#12:close lk
      elif self.attack_type == 16:
        self.update_action(13)#13:close hk
      elif self.attack_type == 17:
        self.update_action(22)#22:special 1
      elif self.attack_type == 18:
        self.update_action(23)#23:special 2
    elif self.jump == True:
      self.update_action(3)#3:jump
    elif self.running == True:
      self.update_action(4)#4:run
    elif self.backUp == True:
      self.update_action(25)#4:backUp
    elif self.crouch == True:
      self.update_action(5)#5:crouch
    else:
      self.update_action(0)#0:idle

    # if(self.action != 0):
    #   print("action:  ", self.action)
    #   print("frame_index:  ", self.frame_index)

    animation_cooldown = 5
    #update image
    self.image = self.animation_list[self.action][self.frame_index]
    #check if enough time has passed since the last update
    if pygame.time.get_ticks() - self.update_time > animation_cooldown:
      self.frame_index += 1
      self.update_time = pygame.time.get_ticks()
    #check if the animation has finished
    if self.frame_index >= len(self.animation_list[self.action]):
      #if the player is dead then end the animation
      if self.alive == False:
        self.frame_index = len(self.animation_list[self.action]) - 1
      else:
        self.frame_index = 0
        #check if an attack was executed
        if self.action == 6 or self.action == 7 or self.action == 8 or self.action == 9 or self.action == 10 or self.action == 11 or self.action == 12 or self.action == 13 or self.action == 14 or self.action == 15 or self.action == 16 or self.action == 17 or self.action == 18 or self.action == 19 or self.action == 20 or self.action == 21 or self.action == 22 or self.action == 23 or self.action == 27 or self.action == 28:
          self.attacking = False
          self.attack_cooldown = 0
        #check if damage was taken
        else:
          self.hit = False
          #if the player was in the middle of an attack, then the attack is stopped
          self.attacking = False
          self.attack_cooldown = 0


  def attack(self, target, surface, atk_type, damage):
    # print("attack:  ", self.attack_type)
    # print("cooldown:  ", self.attack_cooldown)
    # print("damage:  ", damage/4)
    if self.attack_cooldown == 0:
      #execute attack
      # self.attacking = True
      self.attack_sound.play()
      #rect attack range (PERLU DIGANTI!!!)
      # print(self.attack_type)
      if atk_type == 1:
        attacking_rect = pygame.Rect(self.rect.centerx - (2 * self.rect.width * self.flip), self.rect.y, 2 * self.rect.width, self.rect.height) #lp 
      elif atk_type == 2:
        attacking_rect = pygame.Rect(self.rect.centerx - (1.25 * self.rect.width * self.flip), self.rect.y, 1.25 * self.rect.width, 1.25 * self.rect.height) #hp
      else:
        attacking_rect = pygame.Rect(self.rect.centerx - (2 * self.rect.width * self.flip), self.rect.y, 5 * self.rect.width, 5 * self.rect.height) 
      
      pygame.draw.rect(surface, (255,0,0, 128), attacking_rect)
      
      # # Create a separate surface with the SRCALPHA flag for per-pixel alpha
      # atk_surface = pygame.Surface((attacking_rect.width, attacking_rect.height), pygame.SRCALPHA)
      # # Draw the rectangle on the atk_surface with your desired color and alpha
      # pygame.draw.rect(atk_surface, (255, 0, 255, 180), atk_surface.get_rect())
      # # Blit the atk_surface onto your main surface at the position of attacking_rect
      # surface.blit(atk_surface, attacking_rect.topleft)

      
      #check if the attacking player is the player to attack
      if attacking_rect.colliderect(target.rect):
        target.hit = True
        if target.backUp or target.crouch:
          target.health -= 1
        else:
          target.health -= damage/2



  def update_action(self, new_action):
    #check if the new action is different to the previous one
    if new_action != self.action:
      self.action = new_action
      #update the animation settings
      self.frame_index = 0
      self.update_time = pygame.time.get_ticks()

  def draw(self, surface):
    img = pygame.transform.flip(self.image, self.flip, False)
    surface.blit(img, (self.rect.x - (self.offset[0] * self.image_scale), self.rect.y - (self.offset[1] * self.image_scale)))
    # pygame.draw.rect(surface, (255,0,0, 128), self.rect)
    # Create a separate surface with per-pixel alpha
    bodyRect = pygame.Surface((self.rect.size), pygame.SRCALPHA)
    bodyRect.fill((255, 0, 0, 100))
    surface.blit(bodyRect, self.rect.topleft)