#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CartPole AI Project

A pole is attached by an un-actuated joint to a cart, which moves along a 
frictionless track. 
The system is controlled by applying a force of +1 or -1 to the cart. 
The pendulum starts upright, and the goal is to prevent it from falling over. 
A reward of +1 is provided for every timestep that the pole remains upright. 
The episode ends when the pole is more than 15 degrees from vertical, or the
cart moves more than 2.4 units from the center.

@author: deronsart
"""

import gym
import random
import numpy as np

from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

import time


def model_data_preparation(number_games, number_steps, score_requirement):
    '''
        Prepare the data for the training part with random actions on a number 
        of games
        Param :
            The number of games
            The number of steps per game
            The score require 
        Return:
            The data which will be used to train the model
    '''
    
    training_data = []
    
    for _ in range(number_games):
        score = 0
        game_memory = []
        previous_observation = []
        
        for step in range(number_steps):
            action = random.randint(0, 1)
            observation, reward, done, info = env.step(action)
            
            if step!=0:
                game_memory.append([previous_observation, action])
            previous_observation = observation
            
            score+=reward
            
            if done:
                break
        
        if score >= score_requirement:
            for data in game_memory:
                if data[1] == 1:
                    output = [0, 1]
                elif data[1] == 0:
                    output = [1, 0]
                training_data.append([data[0], output])
        
        env.reset()
        
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
    model.add(Dense(32, input_dim=input_size, activation='relu'))
    model.add(Dense(128, activation='relu'))
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
    env = gym.make('CartPole-v1')
    env.reset()

    training_data = model_data_preparation(number_games = 10000, 
                                           number_steps = 500, 
                                           score_requirement = 60)

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


