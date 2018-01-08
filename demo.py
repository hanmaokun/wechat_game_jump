# INITIALIZATION: libraries, parameters, network...

from keras.models import Sequential         # One layer after the other
from keras.layers import Dense, Flatten     # Dense layers are fully connected layers, Flatten layers flatten out multidimensional inputs
from collections import deque               # For storing moves 
from keras.models import model_from_json

import numpy as np
import pickle
import os

from game import JumpGame
env = JumpGame()            

import random                               # For sampling batches from the observations

def model_init(model_file_name):
    # Create network. Input is two consecutive game states, output is Q-values of the possible moves.
    json_model_path = './model/' + model_file_name + '.json'
    weights_model_path = './model/' + model_file_name + '.h5'

    if os.path.isfile(weights_model_path):
        file_stat = os.stat(weights_model_path)
        if file_stat.st_size != 0:
            json_file = open(json_model_path, 'r')
            loaded_model_json = json_file.read()
            json_file.close()
            loaded_model = model_from_json(loaded_model_json)
            # load weights into new model
            loaded_model.load_weights(weights_model_path)

            return loaded_model

    model = Sequential()
    model.add(Dense(20, input_shape=(2,) + (320, 240, 3), init='uniform', activation='relu'))
    model.add(Flatten())                           # Flatten input so as to have no problems with processing
    model.add(Dense(18, init='uniform', activation='relu'))
    model.add(Dense(10, init='uniform', activation='relu'))

    model.add(Dense(20, init='uniform', activation='linear'))    # Same number of outputs as possible actions
    model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])

    return model

# Parameters
observetime = 500                          # Number of timesteps we will be acting on the game and observing results
epsilon = 0.7                              # Probability of doing a random move
gamma = 0.9                                # Discounted future reward. How much we care about steps further in time
mb_size = 50                               # Learning minibatch size

jump_model = model_init('jump_model')
observation = env.reset()                  # Game begins
obs = np.expand_dims(observation, axis=0)  # (Formatting issues) Making the observation the first element of a batch of inputs 
state = np.stack((obs, obs), axis=1)
done = False
for t in range(observetime):
    Q = jump_model.predict(state)          # Q-values predictions
    action = np.argmax(Q)                  # Move with highest Q-value is the chosen one
    print(action)
    observation_new, reward, done, info = env.step(action + 10)     # See state of the game, reward... after performing the action
    obs_new = np.expand_dims(observation_new, axis=0)          # (Formatting issues)
    state_new = np.append(np.expand_dims(obs_new, axis=0), state[:, :1, :], axis=1)     # Update the input with the new state of the game

    state = state_new

print('Demo Finished')