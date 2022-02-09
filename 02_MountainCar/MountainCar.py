#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MountainCar Project

A car is on a one-dimensional track, positioned between two "mountains". 
The goal is to drive up the mountain on the right; however, the car's engine
is not strong enough to scale the mountain in a single pass. Therefore, 
the only way to succeed is to drive back and forth to build up momentum.

@author: deronsart
"""

import gym

import random

from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

import numpy as np

import time


def data_preparation(env, number_games, number_steps, score_requirement):
    '''
        Prepare the data for the training part with random actions on a number 
        of games
        Param :
            The environment
            The number of games
            The number of steps per game
            The minimum score require to win the game
        Return:
            The data which will be used to train the model
    '''
    print ("Creation of the training data ...")
    training_data = []
    
    for game in range(number_games):
        print("\r"+str(int(game*100/number_games))+" %", end="")
        score = 0
        game_memory = []
        previous_observation = []
        
        for step in range(number_steps):
            action = random.randrange(3)
            observation, reward, done, info = env.step(action)
            
            if (step != 0):
                game_memory.append([previous_observation, action])
            previous_observation = observation
            
            if (observation[0] > -0.2):
                reward = 1
            score += reward
            
            if done:
                break
        
        if (score >= score_requirement):
            for data in game_memory:
                if data[1] == 0:
                    output = [1, 0, 0]
                elif data[1] == 1:
                    output = [0, 1, 0]
                elif data[1] == 2:
                    output = [0, 0, 1]
                training_data.append([data[0], output])
                
        env.reset()
        
    print("\r100 %")
    return training_data


def build_model(input_size, output_size):
    '''
        Create the Deep Learning model to play the game
        Param :
            The input size
            The output size
        Return:
            The model
    '''
    
    model = Sequential()
    model.add(Dense(128, input_dim=input_size, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(output_size, activation='linear'))
    
    model.compile(loss='mse', optimizer=Adam())
    
    return model
 

def train_model(training_data):
    '''
        Create and train the Deep Learning model to play the game
        Param :
            The data which will be used to train the model
        Return:
            The trained model
    '''
    x = np.array([i[0] for i in training_data]).reshape(
                                                -1, len(training_data[0][0])) 
    y = np.array([i[1] for i in training_data]).reshape(
                                                -1, len(training_data[0][1]))
    
    model = build_model(len(x[0]), len(y[0]))
    model.fit(x, y, epochs=10)
    
    return model



if __name__ == '__main__':
    env = gym.make('MountainCar-v0')
    env.reset()
    
    training_data = data_preparation(env = env,
                                     number_games = 10000, 
                                     number_steps = 200, 
                                     score_requirement = -198)
    
    model = train_model(training_data)
    
    number_games = 10
    scores = []
    for i in range(1, number_games+1):
        time.sleep(1)
        observation = env.reset()
        
        score = 0
        step = 0
        done = False
        last_observation = []
        while (not done):
            env.render()
            
            if len(last_observation)==0:
                action = random.randrange(0, 2)
            else:
                action = np.argmax(model.predict(
                    last_observation.reshape(-1, len(last_observation)))[0])
            
            observation, reward, done, info = env.step(action)
            last_observation = observation
            
            score += reward
            step += 1;
            
        scores.append(score)
        print ("Game " + str(i) + " : " + str(scores[i-1]))
        
    
    print("Average Score = ", sum(scores)/len(scores))
    
    env.close()


