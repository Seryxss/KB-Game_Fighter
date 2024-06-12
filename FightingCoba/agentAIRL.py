import torch
import random
import numpy as np
from collections import deque

from gameAI import fighterGameAITraining, ActionLists, Position
from model import Linear_QNet, QTrainer
from helper import plot

MAX_MEMORY = 100_000_000
BATCH_SIZE = 1000000
LR = 0.001

class Agent:

    def __init__(self):
        self.n_games = 0
        self.epsilon = 0 # randomness
        self.gamma = 0.9 # discount rate
        self.memory = deque(maxlen=MAX_MEMORY) # popleft()
        self.model = Linear_QNet(23, 256, 19) # (input neurons, I decide the intermediate neurons, output results)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)

    def get_state(self, game):
        positionSelf = game.positionSelf
        positionEnemy = game.positionSelf

        act_0 = game.action == ActionLists.IDLE
        act_1 = game.action == ActionLists.W
        act_2 = game.action == ActionLists.A
        act_3 = game.action == ActionLists.S
        act_4 = game.action == ActionLists.D
        act_5 = game.action == ActionLists.R
        act_6 = game.action == ActionLists.T
        act_7 = game.action == ActionLists.F
        act_8 = game.action == ActionLists.G
        act_9 = game.action == ActionLists.C
        act_10 = game.action == ActionLists.V
        act_11 = game.action == ActionLists.WR
        act_12 = game.action == ActionLists.WT
        act_13 = game.action == ActionLists.WF
        act_14 = game.action == ActionLists.WG
        act_15 = game.action == ActionLists.SR
        act_16 = game.action == ActionLists.ST
        act_17 = game.action == ActionLists.SF
        act_18 = game.action == ActionLists.SG

        state = [
            act_0, #1
            act_1, 
            act_2, 
            act_3,
            act_4,
            act_5,
            act_6,
            act_7,
            act_8,
            act_9,
            act_10,
            act_11,
            act_12,
            act_13,
            act_14,
            act_15,
            act_16,
            act_17,
            act_18, #19

            abs(positionSelf.x - positionEnemy.x) < 180, #20
            abs(positionSelf.y - positionEnemy.y) < 220,
            positionSelf.x2 == positionEnemy.x2,
            positionSelf.y2 == positionEnemy.y2 #23
        ]

        return np.array(state, dtype = int)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done)) # pop left if MAX_MEM is reached

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE) # list of tuples
            
        else:
            mini_sample = self.memory
        
        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)
        for state, action, reward, next_state, done in mini_sample:
            self.trainer.train_step(state, action, reward, next_state, done)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        self.epsilon = 100 - self.n_games #100 ni kek disuru explor sek 100 game e
        final_move = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 18)
            final_move[move] = 1
        
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1
            
        return final_move


def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent = Agent()
    game = fighterGameAITraining()
    
    agent.model.load() # checks and loads the previous trained model if it exists
    
    while True:
        # get old state 
        state_old = agent.get_state(game)
        
        # get move
        final_move = agent.get_action(state_old)
        
        # Perform move and get new state
        reward, done, score = game.play_step(final_move)
        state_new = agent.get_state(game)
        
        # train short memory
        agent.train_short_memory(state_old, final_move, reward, state_new, done)
        
        # Remember 
        agent.remember(state_old, final_move, reward, state_new, done)
        
        if done:
            # Train long memory, plot result
            game.resetGame()
            agent.n_games += 1
            agent.train_long_memory()
            
            if score > record:
                record = score
                agent.model.save()
                
            print('Game: %s \nScore: %s \nRecord: %s' % (agent.n_games, score, record))
            
            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)
            plot(plot_scores, plot_mean_scores)
            

if __name__ == "__main__":
    train()