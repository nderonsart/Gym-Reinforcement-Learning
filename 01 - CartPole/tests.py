#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tests of the gym environment CartPole.

A pole is attached by an un-actuated joint to a cart, which moves along a 
frictionless track. 
The system is controlled by applying a force of +1 or -1 to the cart. 
The pendulum starts upright, and the goal is to prevent it from falling over. 
A reward of +1 is provided for every timestep that the pole remains upright. 
The episode ends when the pole is more than 15 degrees from vertical, or the
cart moves more than 2.4 units from the center.

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
        time.sleep(2)
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
            time.sleep(0.05)
            step += 1;



if __name__ == '__main__':
    env = gym.make('CartPole-v1')
    env.reset()
    
    random_game(5)
    
    env.close()


