#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tests of the gym environment MountainCar.

A car is on a one-dimensional track, positioned between two "mountains". 
The goal is to drive up the mountain on the right; however, the car's engine
is not strong enough to scale the mountain in a single pass. Therefore, 
the only way to succeed is to drive back and forth to build up momentum.

@author: deronsart
"""

import time
import gym


def random_game(number_games):
    '''
        Create n random games
        The details of the games are printed
        Param :
            The number of games
    '''
    
    for i in range(1, number_games+1):
        print("\n\n\nGame " + str(i) + ":")
        time.sleep(1)
        observation = env.reset()
        
        step = 0
        done = False
        while (not done):
            env.render()
            action = env.action_space.sample()
            observation, reward, done, info = env.step(action)
            print("\n\nStep " + str(step))
            print("action: " + str(action))
            print("observation: " + str(observation))
            print("reward: " + str(reward))
            print("done: " + str(done))
            print("info: " + str(info))
            step += 1;



if __name__ == '__main__':
    env = gym.make('MountainCar-v0')
    env.reset()
    
    random_game(5)
    
    env.close()


