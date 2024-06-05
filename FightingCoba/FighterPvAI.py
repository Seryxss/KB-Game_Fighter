import pygame
import math
import random

class FighterPvAI():
  def __init__(self, player, x, y, flip, data, sprite_sheet, animation_steps, sound, screen_width):
    self.upward_force = 20
    self.floating_duration = 40
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
    self.rect = pygame.Rect((x, y, 60, 160))
    self.collision_rect = pygame.Rect(self.rect.x, self.rect.y, self.rect.width, self.rect.height)
    self.vel_y = 0
    self.running = False
    self.backUp = False
    self.knockback_frames = 0
    self.jump = False
    self.crouch = False
    self.attacking = False
    self.attack_type = 0
    self.distance = 0
    self.attack_cooldown = 0
    self.attack_sound = sound
    self.hit = False
    self.health = 144.0
    self.alive = True
    self.target = None
    self.surface = None
    self.damage = 0
    self.knockback = 3
    self.block = False
    self.screen_width = screen_width
    self.offsetStand = data[2]
    self.offsetCrouch=[110,150]
    self.pauseHurtBox = 0
    self.jump_hit = False
    
    self.reaction_frame = 1000/60 #ini dari ticks nya idk translasinya ke fps :V (jadi tiap 1600ticks, blablabla)
    self.last_count_update = pygame.time.get_ticks()
    self.keys={"W" : False,
               "A" : False,
               "S" : False,
               "D" : False,
               "R" : False,
               "T" : False,
               "F" : False,
               "G" : False,
               "C" : False,
               "V" : False
               # aku mikir buat nambahi combi specialnya skalian kek "ST" = False, dst, enak gimana?
               }
    self.next_action = 8

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
    if self.target == None and self.surface == None:
      self.target = target
      self.surface = surface

    self.collision_rect.x = self.rect.x
    self.collision_rect.y = self.rect.y
    self.SPEED = 3
    self.GRAVITY = 2
    self.dx = 0
    self.dy = 0
    self.running = False
    self.backUp = False
    self.crouch = False
    self.attack_type = 0

    #get keypresses
    key = pygame.key.get_pressed()

    # Check for collision with the target player
    if self.collision_rect.colliderect(target.collision_rect):
    # Resolve collision by separating the players
      if self.rect.centerx < target.rect.centerx:
        self.rect.right = target.rect.left
        self.collision_rect.right = target.collision_rect.left
      else:
        self.rect.left = target.rect.right
        self.collision_rect.left = target.collision_rect.right

    #can only perform other actions if not currently attacking
    if self.attacking == False and self.alive == True and round_over == False and self.attack_cooldown == 0:
      distance = math.sqrt((self.rect.centerx - target.rect.centerx)**2 + (self.rect.centery - target.rect.centery)**2)
      #check player 1 controls
      if self.player == 1 and self.action != 2:
        #crouch
        if key[pygame.K_s] and self.crouch == False and self.jump == False:
            self.crouch = True
            self.intialCrouch = True
            self.dx = 0  # Set dx to 0 when crouching
            if self.flip == False:
                if key[pygame.K_a]:  # Check if moving backward and not already backing up
                    self.backUp = True
                else:
                    self.backUp = False
            else:
                if key[pygame.K_d]:  # Check if moving backward and not already backing up
                    self.backUp = True
                else:
                    self.backUp = False
        #movement
        if not self.crouch:  # Only allow movement when not crouching
            if self.flip == False:
                if key[pygame.K_a]:
                    self.dx = -self.SPEED
                    self.backUp = True
                if key[pygame.K_d]:
                    self.dx = self.SPEED
                    self.running = True
                    self.backUp = False
            else:
                if key[pygame.K_a]:
                    self.dx = -self.SPEED
                    self.running = True
                    self.backUp = False
                if key[pygame.K_d]:
                    self.dx = self.SPEED
                    self.backUp = True
        #jump
        if key[pygame.K_w] and self.jump == False and self.crouch == False:
          self.vel_y = -30
          self.jump = True
          self.initial_flip = self.flip

        #attack punch
        if (key[pygame.K_r] or key[pygame.K_t]) and self.jump == False and self.crouch == False:
          #determine which attack type was used
          if distance < 80:
            if key[pygame.K_r]:
              self.attack_type = 13
              self.attacking = True
              self.damage = 8
              
            if key[pygame.K_t]:
              self.attack_type = 14
              self.attacking = True
              self.damage = 28
              
          else :  
            if key[pygame.K_r]:
              self.attack_type = 1
              self.attacking = True
              self.damage = 8
              
              # print(self.attack)
            if key[pygame.K_t]:
              self.attack_type = 2
              self.attacking = True
              self.damage = 28
              
              # print(self.attack)
          
        #attack while jumping
        if (key[pygame.K_r] or key[pygame.K_t]) and self.jump == True:
          
          #determine which attack type was used
          if key[pygame.K_r]:
            self.attack_type = 9
            self.attacking = True
            self.damage = 12
            
          if key[pygame.K_t]:
            self.attack_type = 10
            self.attacking = True
            self.damage = 26
            
        #attack while crouching
        if (key[pygame.K_r] or key[pygame.K_t]) and self.crouch == True:
          
          #determine which attack type was used
          if key[pygame.K_r]:
            self.attack_type = 5
            self.attacking = True
            self.damage = 8
            
          if key[pygame.K_t]:
            self.attack_type = 6  
            self.attacking = True
            self.damage = 28
              
        # special attack
        if key[pygame.K_c] and self.jump == False and self.crouch == False :
          
          self.attack_type = 17
          self.attacking = True
          self.damage = 24
          
        if key[pygame.K_v]  and self.jump == False and self.crouch == False:
          
          self.attack_type = 18
          self.attacking = True
          self.damage = 32
          
        #attack kick
        if (key[pygame.K_f] or key[pygame.K_g]) and self.jump == False and self.crouch == False:
          
          #determine which attack type was used
          if distance < 80:
            if key[pygame.K_f]:
              self.attack_type = 15
              self.attacking = True
              self.damage = 12
              
            if key[pygame.K_g]:
              self.attack_type = 16
              self.attacking = True
              self.damage = 28
              
          else : 
            if key[pygame.K_f]:
              self.attack_type = 3
              self.attacking = True
              self.damage = 14
              
            if key[pygame.K_g]:
              self.attack_type = 4
              self.attacking = True
              self.damage = 30
              
          #attack while jumping
        if (key[pygame.K_f] or key[pygame.K_g]) and self.jump == True:
          
          #determine which attack type was used
          if key[pygame.K_f]:
            self.attack_type = 11
            self.attacking = True
            self.damage = 14
            
          if key[pygame.K_g]:
            self.attack_type = 12
            self.attacking = True
            self.damage = 30
            
        #attack while crouching
        if (key[pygame.K_f] or key[pygame.K_g]) and self.crouch == True:
          
          #determine which attack type was used
          if key[pygame.K_f]:
            self.attack_type = 7
            self.attacking = True
            self.damage = 8
            
          if key[pygame.K_g]:
            self.attack_type = 8  
            self.attacking = True  
            self.damage = 26
      
      ###############TEMPAT AI BEHA#######################
      if (pygame.time.get_ticks() - self.last_count_update) >= self.next_action*self.reaction_frame:
        self.next_action = random.randint(6,10)
        choice = random.randint(1,1000)
        distance = abs(self.rect.centerx - target.rect.centerx)
        #print(distance-self.rect.width, self.rect.width)
        if self.flip == 0:
          print(distance)
        self.keys = dict.fromkeys(self.keys,False)
        self.last_count_update = pygame.time.get_ticks()
        if self.jump:
          if target.jump:
            if choice in range(1, 240):
              self.keys["R"] = True
            elif choice in range(241, 480):
              self.keys["T"] = True
            elif choice in range(481, 720):
              self.keys["F"] = True
            elif choice in range(721, 960):
              self.keys["G"] = True
            elif choice in range(961, 1000):
              if self.flip:
                self.keys["D"] = True
              else:
                self.keys["A"] = True
              if choice in range(981,1000):
                self.keys["S"] = True
          else:
            if distance < 180: # 4-6 kali cek dalam 1 jump
              if choice in range(1, 120):
                self.keys["R"] = True
              elif choice in range(121, 300):
                self.keys["T"] = True
              elif choice in range(301, 420):
                self.keys["F"] = True
              elif choice in range(421, 600):
                self.keys["G"] = True
              elif choice in range(601, 960):
                if self.flip:
                  self.keys["D"] = True
                else:
                  self.keys["A"] = True
                if choice in range(721,960):
                  self.keys["S"] = True
            else: 
                if self.flip:
                  self.keys["A"] = True
                else:
                  self.keys["D"] = True
              
            
        
        elif distance > 180: # maju kalo jauh
          if self.flip:
            if choice in range (1, 100):
              self.keys["D"] = True
              self.keys["S"] = True
            else:
              self.keys["A"] = True
          else:
            if choice in range (1, 100):
              self.keys["A"] = True
              self.keys["S"] = True
            else:
              self.keys["D"] = True
        else:
          if target.jump:
            if choice in range(1,150):
              self.keys["C"] = True
            elif choice in range(151,270):
              self.keys["S"] = True
              self.keys["T"] = True
            elif choice in range(271,390):
              if self.flip:
                self.keys["D"] = True
              else:
                self.keys["A"] = True
            elif choice in range(391,480):
              self.keys["F"] = True
            elif choice in range(481,570):
              self.keys["G"] = True
            elif choice in range(571,630):
              self.keys["R"] = True
            elif choice in range(631,690):
              self.keys["T"] = True
            elif choice in range(691,750):
              self.keys["S"] = True
              if self.flip:
                self.keys["D"] = True
              else:
                self.keys["A"] = True
            elif choice in range(751,780):
              self.keys["S"] = True
              self.keys["R"] = True
            elif choice in range(781,810):
              self.keys["S"] = True
              self.keys["F"] = True
            elif choice in range(811,840):
              self.keys["S"] = True
              self.keys["G"] = True
            elif choice in range(841,870):
              self.keys["V"] = True
            elif choice in range(871,990):
              self.keys["W"] = True
          # formula attack hit: distance = total range + 30
          else:
            if distance < 100: # close range
              if choice in range(1,50):
                if self.flip:
                  self.keys["A"]=True
                else:
                  self.keys["D"]=True
              elif choice in range(51,450):
                if self.flip:
                  self.keys["A"]=True
                else:
                  self.keys["D"]=True
                if choice in range(51,250):
                  self.keys["S"]=True
              elif choice in range(451,560):
                self.keys["R"]=True
              elif choice in range(561,615):
                self.keys["T"]=True
              elif choice in range(616,725):
                self.keys["F"]=True
              elif choice in range(726,780):
                self.keys["G"]=True
              elif choice in range(781,890):
                self.keys["S"]=True
                self.keys["R"]=True
              elif choice in range(891,900):
                self.keys["C"]=True
            
            elif distance < 130: # mid range
              if choice in range(1,200):
                if self.flip:
                  self.keys["A"]=True
                else:
                  self.keys["D"]=True
              elif choice in range(201,450):
                if self.flip:
                  self.keys["A"]=True
                else:
                  self.keys["D"]=True
                if choice in range(21,350):
                  self.keys["S"]=True
              elif choice in range(351,530):
                self.keys["T"]=True
              elif choice in range(531,610):
                self.keys["S"]=True
                self.keys["R"]=True
              elif choice in range(611,690):
                self.keys["S"]=True
                self.keys["F"]=True
              elif choice in range(691,770):
                self.keys["S"]=True
                self.keys["G"]=True
              elif choice in range(771,800):
                self.keys["R"]=True
              elif choice in range(801,830):
                self.keys["G"]=True
              elif choice in range(831,860):
                self.keys["S"]=True
                self.keys["T"]=True
              elif choice in range(861,890):
                self.keys["W"]=True
              elif choice in range(891,920):
                self.keys["C"]=True
              elif choice in range(921,950):
                self.keys["V"]=True
                
            else: # long range 
              if choice in range(1,600):
                if self.flip:
                  self.keys["A"]=True
                else:
                  self.keys["D"]=True
              elif choice in range(601,800):
                if self.flip:
                  self.keys["A"]=True
                else:
                  self.keys["D"]=True
                if choice in range(601,700):
                  self.keys["S"]=True
              elif choice in range(701,760):
                self.keys["S"]=True
                self.keys["F"]=True
              elif choice in range(761,820):
                self.keys["S"]=True
                self.keys["G"]=True
              elif choice in range(821,880):
                self.keys["C"]=True
              elif choice in range(881,940):
                self.keys["V"]=True

      
      ####################################check player 2 controls################################################################

      if self.player == 2 and self.action != 2:
        distance = math.sqrt((self.rect.centerx - target.rect.centerx)**2 + (self.rect.centery - target.rect.centery)**2)
        #crouch
        if self.keys["S"] and self.crouch == False and self.jump == False:
            self.crouch = True
            self.intialCrouch = True
            self.dx = 0  # Set dx to 0 when crouching
            if self.flip == False:
                if self.keys["A"]:  # Check if moving backward and not already backing up
                    self.backUp = True
                else:
                    self.backUp = False
            else:
                if self.keys["D"]:  # Check if moving backward and not already backing up
                    self.backUp = True
                else:
                    self.backUp = False
        #movement
        if not self.crouch:  # Only allow movement when not crouching
            if self.flip == False:
                if self.keys["A"]:
                    self.dx = -self.SPEED
                    self.backUp = True
                if self.keys["D"]:
                    self.dx = self.SPEED
                    self.running = True
                    self.backUp = False
            else:
                if self.keys["A"]:
                    self.dx = -self.SPEED
                    self.running = True
                    self.backUp = False
                if self.keys["D"]:
                    self.dx = self.SPEED
                    self.backUp = True
        #jump            
        if self.keys["W"] and self.jump == False and self.crouch == False:
          self.vel_y = -30
          self.jump = True
          self.initial_flip = self.flip
          
        #attack punch
        if (self.keys["R"] or self.keys["T"]) and self.jump == False and self.crouch == False:
          
          #determine which attack type was used
          if distance < 80:
            if self.keys["R"]:
              self.attack_type = 13
              self.attacking = True
              self.damage = 8
              
            if self.keys["T"]:
              self.attack_type = 14
              self.attacking = True
              self.damage = 28
              
          else :  
            if self.keys["R"]:
              self.attack_type = 1
              self.attacking = True
              self.damage = 12
              
            if self.keys["T"]:
              self.attack_type = 2
              self.attacking = True
              self.damage = 28
              
        #attack while jumping
        if (self.keys["R"] or self.keys["T"]) and self.jump == True:
          
          #determine which attack type was used
          if self.keys["R"]:
            self.attack_type = 9
            self.attacking = True
            self.damage = 12
            
          if self.keys["T"]:
            self.attack_type = 10
            self.attacking = True
            self.damage = 26
            
        #attack while crouching
        if (self.keys["R"] or self.keys["T"]) and self.crouch == True:
          
          #determine which attack type was used
          if self.keys["R"]:
            self.attack_type = 5
            self.attacking = True
            self.damage = 8  
            
          if self.keys["T"]:
            self.attack_type = 6  
            self.attacking = True
            self.damage = 28
              
        # special attack
        if self.keys["C"] and self.jump == False and self.crouch == False:
          
          self.attack_type = 17
          self.attacking = True
          self.damage = 24
          
        if self.keys["V"]  and self.jump == False and self.crouch == False:
          
          self.attack_type = 18
          self.attacking = True
          self.damage = 32
          
        #attack kick
        if (self.keys["F"] or self.keys["G"]) and self.jump == False and self.crouch == False:
          
          #determine which attack type was used
          if distance < 80:
            if self.keys["F"]:
              self.attack_type = 15
              self.attacking = True
              self.damage = 12
              
            if self.keys["G"]:
              self.attack_type = 16
              self.attacking = True
              self.damage = 28
              
          else : 
            if self.keys["F"] :
              self.attack_type = 3
              self.attacking = True
              self.damage = 14
              
            if self.keys["G"]:
              self.attack_type = 4
              self.attacking = True
              self.damage = 30
              
          #attack while jumping
        if (self.keys["F"] or self.keys["G"]) and self.jump == True:
          
          #determine which attack type was used
          if self.keys["F"]:
            self.attack_type = 11
            self.attacking = True
            self.damage = 14
            
          if self.keys["G"]:
            self.attack_type = 12
            self.attacking = True
            self.damage = 30
            
        #attack while crouching
        if (self.keys["F"] or self.keys["G"]) and self.crouch == True:
          
          #determine which attack type was used
          if self.keys["F"]:
            self.attack_type = 7
            self.attacking = True
            self.damage = 8
            
          if self.keys["G"]:
            self.attack_type = 8
            self.attacking = True
            self.damage = 26
            
    ###########################################################################################################################

    #apply gravity
    self.vel_y += self.GRAVITY
    self.dy += self.vel_y

    #ensure player stays on screen
    if self.rect.left + self.dx < 0:
      self.dx = -self.rect.left
    if self.rect.right + self.dx > screen_width:
      self.dx = screen_width - self.rect.right
    if self.rect.bottom + self.dy > screen_height - 110:
      self.vel_y = 0
      self.jump = False
      self.dy = screen_height - 110 - self.rect.bottom

    if self.jump and self.action == 3:
        if self.initial_flip == False:
            self.dx = self.SPEED * 2
        else:
            self.dx = -self.SPEED * 2
    
    if not (self.jump and self.action == 3):
        if target.rect.centerx > self.rect.centerx:
            self.flip = False
        else:
            self.flip = True

    # Apply gravity and downward movement for the target if it was hit while jumping
    if target.jump_hit:
        target.vel_y += target.GRAVITY
        target.dy += target.vel_y

        # Reset the jump_hit flag when the target lands
        if target.rect.bottom + target.dy > screen_height - 110:
            target.vel_y = 0
            target.jump = False
            target.dy = screen_height - 110 - target.rect.bottom
            target.jump_hit = False
    
    if self.crouch == True:
      #self.vel_y = 500
      #print(self.rect.y)
      self.rect.y=390

    #ensure players face each other
    if target.rect.centerx > self.rect.centerx:
      self.flip = False
    else:
      self.flip = True

    #apply attack cooldown
    if self.attack_cooldown > 0:
      self.attack_cooldown -= 1

    if self.action == 3 and self.frame_index == 0:
        self.dx = 0

    if self.knockback_frames > 0:
      knockback_direction = -1 if self.flip else 1
      self.dx = -knockback_direction * self.knockback
      self.knockback_frames -= 1

    #update player position
    self.rect.x += self.dx
    self.rect.y += self.dy


  #handle animation update
  def update(self):
    #check what action the player is performing
    if self.action == 5 or self.action == 14 or self.action == 16 or self.action == 17:
      self.collision_rect.height = 110
    else:
      self.collision_rect.height = 160

    if self.health <= 0:
        self.health = 0
        self.alive = False
        self.update_action(1)  # 1:death
        self.target.update_action(24)  # victor
    elif self.hit == True:
        if self.flip == True:
            if self.player == 1:
                if self.crouch == True:
                    self.update_action(26)
                elif self.backUp == True:
                    self.block = True
                    self.update_action(25)
                elif self.jump == True:
                    self.dx = 0
                    self.rect.x += self.knockback * 4
                else:
                    self.rect.x += self.knockback
                    self.update_action(2)  # 2:hit
            elif self.player == 2:
                if self.crouch == True:
                    self.update_action(26)
                elif self.backUp == True:
                    self.block = True
                    self.update_action(25)
                elif self.jump == True:
                    self.dx = 0
                    self.rect.x += self.knockback * 4
                else:
                    self.rect.x += self.knockback
                    self.update_action(2)  # 2:hit
        else:
            if self.player == 1:
                if self.crouch == True:
                    self.update_action(26)
                elif self.backUp == True:
                    self.block = True
                    self.update_action(25)
                elif self.jump == True:
                    self.dx = 0
                    self.rect.x -= self.knockback * 4
                else:
                    self.rect.x -= self.knockback
                    self.update_action(2)  # 2:hit
            elif self.player == 2:
                if self.crouch == True:
                    self.update_action(26)
                elif self.backUp == True:
                    self.block = True
                    self.update_action(25)
                elif self.jump == True:
                    self.dx = 0
                    self.rect.x -= self.knockback * 4
                else:
                    self.rect.x -= self.knockback
                    self.update_action(2)  # 2:hit
    elif self.hit:
        self.hit = False
        self.dx = 0
    elif self.attacking == True and self.attack_cooldown == 0:
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
        self.update_action(3)  # 3:jump
    elif self.running == True:
        self.update_action(4)  # 4:run
    elif self.backUp == True and not self.crouch:
        self.update_action(4)  # 4:backUp
    elif self.crouch == True:
        self.update_action(5)  # 5:crouch
    else:
        self.intialCrouch = False
        self.crouch = False
        self.jump = False
        self.update_action(0)  # 0:idle
        self.rect.y = 330

    # if(self.action != 0):
    #   print("damage: ", self.damage)
      # print("action:  ", self.action)
      # print("frame_index:  ", self.frame_index)
  
    # if self.attack_cooldown != 0:
    #   print( )
    # stunEnemy berdasarkan collision, nanti custom tambah berdasarkan frame kenanya
    if (self.action == 6): ############################ lp 96
      if(self.frame_index == 0 ):
        self.attack_sound.play()
      if(self.frame_index >= 2 and self.frame_index < 6):
        attacking_rect = pygame.Rect(self.rect.centerx - (0.1 * self.rect.width * 2*(self.flip-0.5)) - (1 * self.flip*self.rect.width), self.rect.y+25, 1 * self.rect.width, self.rect.height*0.17)
        self.attack(self.target, self.surface, self.damage, attacking_rect, stunEnemy=11-self.frame_index+6, cooldownSelf=0)
    elif (self.action == 7): ############################ hp 138
      if(self.frame_index == 0 ):
        self.attack_sound.play()
      if(self.frame_index >= 5 and self.frame_index < 11):
        attacking_rect = pygame.Rect(self.rect.centerx - (0.4 * self.rect.width * 2*(self.flip-0.5)) - (1.4 * self.flip*self.rect.width), self.rect.y+12, 1.4 * self.rect.width, self.rect.height*0.17)
        self.attack(self.target, self.surface, self.damage, attacking_rect, stunEnemy=34-self.frame_index-6, cooldownSelf=0)
    elif (self.action == 8): ########################### lk 90
      if(self.frame_index == 0 ):
        self.attack_sound.play()
      if(self.frame_index >= 6 and self.frame_index < 14):
        attacking_rect = pygame.Rect(self.rect.centerx - (-0.3 * self.rect.width * 2*(self.flip-0.5)) - (1.3 * self.flip*self.rect.width), self.rect.y-5, 1.3 * self.rect.width, self.rect.height*0.3)
        self.attack(self.target, self.surface, self.damage, attacking_rect, stunEnemy=19-self.frame_index+3, cooldownSelf=0)
    elif (self.action == 9): ########################### hk 108
      if(self.frame_index == 0 ):
        self.attack_sound.play()
      if(self.frame_index >= 2 and self.frame_index < 6): 
        attacking_rect = pygame.Rect(self.rect.centerx - (0.3 * self.rect.width * 2*(self.flip-0.5)) - (0.8 * self.flip*self.rect.width), self.rect.y-8, 0.8 * self.rect.width, self.rect.height*0.35)
        self.attack(self.target, self.surface, self.damage, attacking_rect, stunEnemy=31-self.frame_index-2, cooldownSelf=0)
      if(self.frame_index >= 6 and self.frame_index < 14):
        attacking_rect = pygame.Rect(self.rect.centerx - (0.3 * self.rect.width * 2*(self.flip-0.5)) - (1.3 * self.flip*self.rect.width), self.rect.y-5, 1.3 * self.rect.width, self.rect.height*0.25)
        self.attack(self.target, self.surface, self.damage, attacking_rect, stunEnemy=31-self.frame_index-2, cooldownSelf=0)
    elif (self.action == 10): ########################### close lp xx
      if(self.frame_index == 0 ):
        self.attack_sound.play()
      if(self.frame_index >= 3 and self.frame_index < 7):
        attacking_rect = pygame.Rect(self.rect.centerx - (0.15 * self.rect.width * 2*(self.flip-0.5)) - (0.9 * self.flip*self.rect.width), self.rect.y, 0.9 * self.rect.width, self.rect.height*0.15)
        self.attack(self.target, self.surface, self.damage, attacking_rect, stunEnemy=11-self.frame_index+6, cooldownSelf=0)
    elif (self.action == 11): ########################### close hp xx
      if(self.frame_index == 0 ):
        self.attack_sound.play()
      if(self.frame_index >= 3 and self.frame_index < 6):
        attacking_rect = pygame.Rect(self.rect.centerx - (0.2 * self.rect.width * 2*(self.flip-0.5)) - (1.3 * self.flip*self.rect.width), self.rect.y+25, 1.3 * self.rect.width, self.rect.height*0.15)
        self.attack(self.target, self.surface, self.damage, attacking_rect, stunEnemy=34-self.frame_index-8, cooldownSelf=0)
      if(self.frame_index >= 6 and self.frame_index < 13):
        attacking_rect = pygame.Rect(self.rect.centerx - (0.15 * self.rect.width * 2*(self.flip-0.5)) - (1.5 * self.flip*self.rect.width), self.rect.y-30, 1.5 * self.rect.width, self.rect.height*0.5)
        self.attack(self.target, self.surface, self.damage, attacking_rect, stunEnemy=34-self.frame_index-8, cooldownSelf=0)
    elif (self.action == 12): ########################### close lk xx
      if(self.frame_index == 0 ):
        self.attack_sound.play()
      if(self.frame_index >= 5 and self.frame_index < 11):
        attacking_rect = pygame.Rect(self.rect.centerx - (0.3 * self.rect.width * 2*(self.flip-0.5)) - (1.4 * self.flip*self.rect.width), self.rect.y+95, 1.4 * self.rect.width, self.rect.height*0.4)
        self.attack(self.target, self.surface, self.damage, attacking_rect, stunEnemy=15-self.frame_index+5, cooldownSelf=0)
    elif (self.action == 13): ########################### close hk xx
      if(self.frame_index == 0 ):
        self.attack_sound.play()
      if(self.frame_index >= 9 and self.frame_index < 18):
        attacking_rect = pygame.Rect(self.rect.centerx - (0.35 * self.rect.width * 2*(self.flip-0.5)) - (0.9 * self.flip*self.rect.width), self.rect.y-50, 0.9 * self.rect.width, self.rect.height*0.72)
        self.attack(self.target, self.surface, self.damage, attacking_rect, stunEnemy=30-self.frame_index+8, cooldownSelf=0)
      if(self.frame_index >= 18 and self.frame_index < 21):
        attacking_rect = pygame.Rect(self.rect.centerx - (0.5 * self.rect.width * 2*(self.flip-0.5)) - (1.35 * self.flip*self.rect.width), self.rect.y+25, 1.35 * self.rect.width, self.rect.height*0.4)
        self.attack(self.target, self.surface, self.damage, attacking_rect, stunEnemy=30-self.frame_index+8, cooldownSelf=0)
    elif (self.action == 14): ############################ nunduk lp 108
      if(self.frame_index == 0 ):
        self.attack_sound.play()
      if(self.frame_index >= 2 and self.frame_index < 6):
        attacking_rect = pygame.Rect(self.rect.centerx - (0.1 * self.rect.width * 2*(self.flip-0.5)) - (1.2 * self.flip*self.rect.width), self.rect.y+25, 1.2 * self.rect.width, self.rect.height*0.17)
        self.attack(self.target, self.surface, self.damage, attacking_rect, stunEnemy=11-self.frame_index+6, cooldownSelf=0)
    elif (self.action == 15): ########################### nundk hp 108
      if(self.frame_index == 0 ):
        self.rect.y = 330
        self.attack_sound.play()
      if(self.frame_index >= 3 and self.frame_index < 7):
        attacking_rect = pygame.Rect(self.rect.centerx - (0.1 * self.rect.width * 2*(self.flip-0.5)) - (1.2 * self.flip*self.rect.width), self.rect.y+5, 1.2 * self.rect.width, self.rect.height*0.36)
        self.attack(self.target, self.surface, self.damage, attacking_rect, stunEnemy=37-self.frame_index-11, cooldownSelf=0)
      if(self.frame_index >= 7 and self.frame_index < 16):
        attacking_rect = pygame.Rect(self.rect.centerx - (0.2 * self.rect.width * 2*(self.flip-0.5)) - (1.1 * self.flip*self.rect.width), self.rect.y-40, 1.1 * self.rect.width, self.rect.height*0.68)
        self.attack(self.target, self.surface, self.damage, attacking_rect, stunEnemy=37-self.frame_index-11, cooldownSelf=0)
    elif (self.action == 16): ########################### nunduk lk 126
      if(self.frame_index == 0 ):
        self.attack_sound.play()
      if(self.frame_index >= 2 and self.frame_index < 7):
        attacking_rect = pygame.Rect(self.rect.centerx - (0.4 * self.rect.width * 2*(self.flip-0.5)) - (1.2 * self.flip*self.rect.width), self.rect.y+80, 1.2 * self.rect.width, self.rect.height*0.29)
        self.attack(self.target, self.surface, self.damage, attacking_rect, stunEnemy=11-self.frame_index+6, cooldownSelf=0)
    elif (self.action == 17): ########################### nunduk hk 162
      if(self.frame_index == 0 ):
        self.attack_sound.play()
      if(self.frame_index >= 3 and self.frame_index < 10):
        attacking_rect = pygame.Rect(self.rect.centerx - (0.3 * self.rect.width * 2*(self.flip-0.5)) - (1.9 * self.flip*self.rect.width), self.rect.y+80, 1.9 * self.rect.width, self.rect.height*0.29)
        self.attack(self.target, self.surface, self.damage, attacking_rect, stunEnemy=34-self.frame_index+2, cooldownSelf=0)
    elif (self.action == 18): ########################### lompat lp 81+jump
      if(self.frame_index == 0 ):
        self.attack_sound.play()
      if(self.frame_index >= 2):
        attacking_rect = pygame.Rect(self.rect.centerx - (0.1 * self.rect.width * 2*(self.flip-0.5)) - (0.75 * self.flip*self.rect.width), self.rect.y+10, 0.75 * self.rect.width, self.rect.height*0.3)
        self.attack(self.target, self.surface, self.damage, attacking_rect, stunEnemy=10, cooldownSelf=0)
    elif (self.action == 19): ########################### lompat hp 93+jump
      if(self.frame_index == 0 ):
        self.attack_sound.play()
      if(self.frame_index >= 4 and self.frame_index < 12):
        attacking_rect = pygame.Rect(self.rect.centerx - (0.05 * self.rect.width * 2*(self.flip-0.5)) - (1 * self.flip*self.rect.width), self.rect.y+50, 1 * self.rect.width, self.rect.height*0.15)
        self.attack(self.target, self.surface, self.damage, attacking_rect, stunEnemy=10, cooldownSelf=0)
    elif (self.action == 20): ########################### lompat lk 84+jump
      if(self.frame_index == 0 ):
        self.attack_sound.play()
      if(self.frame_index >= 3):
        attacking_rect = pygame.Rect(self.rect.centerx - (-0.8 * self.rect.width * 2*(self.flip-0.5)) - (1.7 * self.flip*self.rect.width), self.rect.y+5, 1.7 * self.rect.width, self.rect.height*0.4)
        self.attack(self.target, self.surface, self.damage, attacking_rect, stunEnemy=10, cooldownSelf=0)
    elif (self.action == 21): ########################### lompat hk 72,117 +jump
      if(self.frame_index == 0):
        self.attack_sound.play()
      if(self.frame_index >= 3 and self.frame_index < 8):
        attacking_rect = pygame.Rect(self.rect.centerx - (-0.5 * self.rect.width * 2*(self.flip-0.5)) - (1.2 * self.flip*self.rect.width), self.rect.y+5, 1.2 * self.rect.width, self.rect.height*0.4)
        self.attack(self.target, self.surface, self.damage, attacking_rect, stunEnemy=10, cooldownSelf=0)
      if(self.frame_index >= 8 and self.frame_index < 15):
        attacking_rect = pygame.Rect(self.rect.centerx - (-0.5 * self.rect.width * 2*(self.flip-0.5)) - (1.95 * self.flip*self.rect.width), self.rect.y+20, 1.95 * self.rect.width, self.rect.height*0.5)
        self.attack(self.target, self.surface, self.damage, attacking_rect, stunEnemy=10, cooldownSelf=0)
    elif (self.action == 22): ########################### shoryuken move 20 total 140
      if(self.frame_index == 5):
        self.gravity = 0.2
        if self.floating_duration > 0:
            self.vel_y -= self.upward_force
            self.floating_duration -= 1
        else:
            self.gravity = 2
            self.floating_duration = 40 
            self.action = 0

        # Update position based on velocity
        self.rect.y += self.vel_y/2
        # hilangin hitbox
        self.attack_sound.play()
      if(self.frame_index >= 4 and self.frame_index < 8):
        if not self.flip:
          self.rect.x += 5
        else: 
          self.rect.x -= 5 
        attacking_rect = pygame.Rect(self.rect.centerx - (0 * self.rect.width * 2*(self.flip-0.5)) - (1.5 * self.flip*self.rect.width), self.rect.y+30, 1.5 * self.rect.width, self.rect.height*0.5)
        self.attack(self.target, self.surface, self.damage, attacking_rect, stunEnemy=55-self.frame_index-26, cooldownSelf=0)
      if(self.frame_index >= 8 and self.frame_index < 22):
        attacking_rect = pygame.Rect(self.rect.centerx - (0 * self.rect.width * 2*(self.flip-0.5)) - (1.25 * self.flip*self.rect.width), self.rect.y-50, 1.25 * self.rect.width, self.rect.height*0.7)
        self.attack(self.target, self.surface, self.damage, attacking_rect, stunEnemy=55-self.frame_index-26, cooldownSelf=0)
      # if(self.frame_index >= 22):
      # tambahin hurtbox balik
    elif (self.action == 23): ########################### tatsumaki senpukyaku move 170 first hit 126+65=191
      if not self.flip:  # Facing right
          self.rect.x += 5
      else:  # Facing left
          self.rect.x -= 5 
      if(self.frame_index == 0):
        self.attack_sound.play()
      if(self.frame_index >= 11 and self.frame_index < 14):
        attacking_rect = pygame.Rect(self.rect.centerx - (0.2 * self.rect.width * 2*(self.flip-0.5)) - (1.4 * self.flip*self.rect.width), self.rect.y-15, 1.4 * self.rect.width, self.rect.height*0.25)
        self.attack(self.target, self.surface, self.damage, attacking_rect, stunEnemy=35-self.frame_index-1, cooldownSelf=0)
      if(self.frame_index >= 17 and self.frame_index < 20):
        attacking_rect = pygame.Rect(self.rect.centerx - (-1.2 * self.rect.width * 2*(self.flip-0.5)) - (1.4 * self.flip*self.rect.width), self.rect.y-15, 1.4 * self.rect.width, self.rect.height*0.25)
        self.attack(self.target, self.surface, self.damage, attacking_rect, stunEnemy=35-self.frame_index-1, cooldownSelf=150)
    else:
      self.damage = 0
      # if(self.frame_index == 0 ):
      #   self.attack_sound.play()

    if (self.action == 5 or self.action == 26 or self.action == 14 or self.action == 16 or self.action == 17): ########################### crouch
      self.rect = pygame.Rect((self.rect.x, self.rect.y, 60, 100))
      self.offset = self.offsetCrouch
    else:
      self.rect = pygame.Rect((self.rect.x, self.rect.y, 60, 160))
      self.offset = self.offsetStand
  
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
          #self.attack_cooldown = 10
        #check if damage was taken
        else:
          self.hit = False
          #if the player was in the middle of an attack, then the attack is stopped
          self.attacking = False
          # self.attack_cooldown = 2
  def attack(self, target, surface, damage, attacking_rect, stunEnemy, cooldownSelf):
    if self.attack_cooldown == 0:
        pygame.draw.rect(surface, (255, 0, 0, 128), attacking_rect)

        if attacking_rect.colliderect(target.rect):
            self.attacking = True
            target.hit = True
            knockback_direction = -1 if self.flip else 1

            if self.intialCrouch:
              print("player Crouch")
              #Cek Edge
              target_at_edge = False
              if target.rect.left <= 0 and knockback_direction == -1:
                  target_at_edge = True
              elif target.rect.right >= self.screen_width and knockback_direction == 1:
                target_at_edge = True
              if target.crouch:
                print("target Crouch")
                  #Cek Backup
                if target.backUp:         
                    # Target backing up (blocking)
                    self.knockback_frames = 10  # Set knockback frames for the attacking player
                elif target_at_edge:
                    # Target at the edge
                    self.knockback_frames = 10
                    target.health -= damage / 20
                    target.attack_cooldown = stunEnemy
                else:             
                    # Target not backing up (hit)
                    target.dx = -knockback_direction * self.knockback
                    target.health -= damage / 20
                    target.attack_cooldown = stunEnemy
                
              else: #Target Standing
                if target.backUp:
                   target.dx = -knockback_direction * self.knockback
                elif target_at_edge:
                  print("target at edge")
                  # Target at the edge
                  self.knockback_frames = 10
                  target.health -= damage / 20
                  target.attack_cooldown = stunEnemy
                else:
                  print ("target standing")
                  target.dx = -knockback_direction * self.knockback
                  target.health -= damage / 20
                  target.attack_cooldown = stunEnemy

            if target.jump:
                print("target jump")
                # Apply upward force to the target
                target.vel_y = -20  # Adjust the value to control the upward force
                target.jump_hit = True
                target.dx = 0

            if not self.intialCrouch:
                if not target.crouch:
                    if not target.backUp:
                        print("target jump")
                        target.dx = -knockback_direction * self.knockback
                        target.health -= damage / 20
                        target.attack_cooldown = stunEnemy
                        
            if not self.intialCrouch: #PlayerStanding
              print ("player standing")
              #Cek Edge
              target_at_edge = False
              if target.rect.left <= 0 and knockback_direction == -1:
                  target_at_edge = True
              elif target.rect.right >= self.screen_width and knockback_direction == 1:
                  target_at_edge = True
              if target.crouch: #Target Crouching
                print ("target crouching")
                if target.backUp:         
                    print("target back up")
                    # Target backing up (blocking)
                    self.knockback_frames = 10  # Set knockback frames for the attacking player
                elif target_at_edge:
                    print("target at edge")
                    # Target at the edge
                    self.knockback_frames = 10
                    target.health -= damage / 20
                    target.attack_cooldown = stunEnemy
                else:             
                    print("target not back up")
                    # Target not backing up (hit)
                    target.dx = -knockback_direction * self.knockback
                    target.health -= damage / 20
                    target.attack_cooldown = stunEnemy
              
              else: #Target Standing
                if target.backUp:
                    print("target back up")
                    # Target backing up (blocking)
                    self.knockback_frames = 10  # Set knockback frames for the attacking player
                elif target_at_edge:
                    print("target at edge")
                    # Target at the edge
                    self.knockback_frames = 10
                    target.health -= damage / 20
                    target.attack_cooldown = stunEnemy
                else:
                    print("target not back up")
                    # Target not backing up (hit)
                    target.dx = -knockback_direction * self.knockback
                    target.health -= damage / 20
                    target.attack_cooldown = stunEnemy
    
  def update_action(self, new_action):
    #check if the new action is different to the previous one
    if new_action != self.action:
      self.action = new_action
      #update the animation settings
      self.frame_index = 0
      self.update_time = pygame.time.get_ticks()
    # print("test update action", self.update_time)

  def draw(self, surface):
    img = pygame.transform.flip(self.image, self.flip, False)
    surface.blit(img, (self.rect.x - (self.offset[0] * self.image_scale), self.rect.y - (self.offset[1] * self.image_scale)))
    # pygame.draw.rect(surface, (255,0,0, 128), self.rect)
    # Create a separate surface with per-pixel alpha
    bodyRect = pygame.Surface((self.rect.size), pygame.SRCALPHA)
    bodyRect.fill((255, 0, 0, 100))
    surface.blit(bodyRect, self.rect.topleft)