# INITIALIZATION: libraries, parameters, network...

from keras.models import Sequential      # One layer after the other
from keras.layers import Dense, Flatten  # Dense layers are fully connected layers, Flatten layers flatten out multidimensional inputs
from collections import deque            # For storing moves 

import numpy as np
import gym      

env = gym.make('MountainCar-v0')          # Choose game (any in the gym should work)

observation = env.reset()
action = 1
done = False

while True:
    env.render()                    # Uncomment to see game running
    c = raw_input()
    if c == 'a':
    	action = 0
    elif c == 's':
    	action = 1
    elif c == 'd':
    	action = 2

    print(c)
    observation, reward, done, info = env.step(action)

print('Game ended! Total reward: {}'.format(reward))
