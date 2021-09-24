#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MountainCar Project

@author: deronsart
"""

import gym
import random
import numpy as np

from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam



env = gym.make('MountainCar-v0')
env.reset()

goal_steps = 200            # -199 -> loose the game
score_requirement = -198    # success -> -198 min 
initial_games = 10000



def model_data_preparation():
    training_data = []
    accepted_scores = []
    
    for game_index in range(initial_games):
        score = 0
        game_memory = []
        previous_observation = []
        
        for step_index in range(goal_steps):
            action = random.randrange(0, 3)
            observation, reward, done, info = env.step(action)
            
            if len(previous_observation)>0:
                game_memory.append([previous_observation, action])
            
            previous_observation = observation
            
            if observation[0]>-0.2:
                reward = 1
            
            score+=reward
            if done:
                break
        
        if score>=score_requirement:
            accepted_scores.append(score)
            for data in game_memory:
                if data[1]==0:
                    output = [1, 0, 0]
                elif data[1]==1:
                    output = [0, 1, 0]
                elif data[1]==2:
                    output = [0, 0, 1]
                training_data.append([data[0], output])
        
        env.reset()
    
    print(accepted_scores)
    return training_data



training_data = model_data_preparation()



def build_model(input_size, output_size):
    model = Sequential()
    model.add(Dense(128, input_dim=input_size, activation='relu'))
    model.add(Dense(52, activation='relu'))
    model.add(Dense(output_size, activation='linear'))
    
    model.compile(loss='mse', optimizer=Adam())
    return model



def train_model(training_data):
    x = np.array([i[0] for i in training_data]).reshape(-1, len(training_data[0][0]))
    y = np.array([i[1] for i in training_data]).reshape(-1, len(training_data[0][1]))
    
    model = build_model(len(x[0]), len(y[0]))
    model.fit(x, y, epochs=10)
    return model



trained_model = train_model(training_data)



scores = []
choices = []
for each_game in range(10):
    score = 0
    game_memory = []
    prev_obs = []
    
    done = False
    while not done:
        
        env.render()
        
        if len(prev_obs)==0:
            action = random.randrange(0, 2)
        else:
            action = np.argmax(trained_model.predict(prev_obs.reshape(-1, len(prev_obs)))[0])
        choices.append(action)
        
        new_observation, reward, done, info = env.step(action)
        prev_obs = new_observation
        score+=reward
    
    env.reset()
    scores.append(score)

print('\nScores :')
print(scores)
print('\nAverage Score: ', sum(scores)/len(scores))
print('Choice 0: {} \nChoice 1: {} \nChoice 2: {}'.format(choices.count(0)/len(choices), choices.count(1)/len(choices), choices.count(2)/len(choices)))



env.close()


