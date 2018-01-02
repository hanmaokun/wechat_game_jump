# INITIALIZATION: libraries, parameters, network...

from keras.models import Sequential      	# One layer after the other
from keras.layers import Dense, Flatten  	# Dense layers are fully connected layers, Flatten layers flatten out multidimensional inputs
from collections import deque            	# For storing moves 

import numpy as np
import pickle
import os

from game import JumpGame
env = JumpGame()			
#import gym                                	# To train our network
#env = gym.make('MountainCar-v0')			# Choose game (any in the gym should work)

import random     							# For sampling batches from the observations

# Create network. Input is two consecutive game states, output is Q-values of the possible moves.
model = Sequential()
model.add(Dense(20, input_shape=(2,) + env.observation_space.shape, init='uniform', activation='relu'))
model.add(Flatten())       					# Flatten input so as to have no problems with processing
model.add(Dense(18, init='uniform', activation='relu'))
model.add(Dense(10, init='uniform', activation='relu'))

#model.add(Dense(env.action_space.n, init='uniform', activation='linear'))    # Same number of outputs as possible actions
#model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])

model.add(Dense(1, init='uniform', activation='softmax'))
model.compile(loss='mean_squared_error', optimizer='adam')

# Parameters
D = deque()                                # Register where the actions will be stored

observetime = 500                          # Number of timesteps we will be acting on the game and observing results
epsilon = 0.7                              # Probability of doing a random move
gamma = 0.9                                # Discounted future reward. How much we care about steps further in time
mb_size = 50                               # Learning minibatch size

# FIRST STEP: Knowing what each action does (Observing)

pkl_file_path = './data/D.pkl'
observation = env.reset()                     # Game begins
obs = np.expand_dims(observation, axis=0)     # (Formatting issues) Making the observation the first element of a batch of inputs 
state = np.stack((obs, obs), axis=1)
done = False
for t in range(observetime):
    if np.random.rand() <= epsilon:
        #action = np.random.randint(0, env.action_space.n, size=1)[0]
        action = random.uniform(0, 1)
    else:
        Q = model.predict(state)            # Q-values predictions
        action = Q[0][0]                    # Move with highest Q-value is the chosen one
    observation_new, reward, done, info = env.step(action)     # See state of the game, reward... after performing the action
    obs_new = np.expand_dims(observation_new, axis=0)          # (Formatting issues)
    state_new = np.append(np.expand_dims(obs_new, axis=0), state[:, :1, :], axis=1)     # Update the input with the new state of the game
    pkl_file_stat = os.stat(pkl_file_path)

    if pkl_file_stat.st_size != 0:
        pkl_file_D = open(pkl_file_path, 'rb')
        D = pickle.load(pkl_file_D)
        pkl_file_D.close()

    D.append((state, action, reward, state_new, done))         # 'Remember' action and consequence

    with open(pkl_file_path, "wb") as f:
        pickle.dump(D, f, pickle.HIGHEST_PROTOCOL)
        f.close()

    state = state_new         # Update state
    if done:
        env.reset()           # Restart game if it's finished
        obs = np.expand_dims(observation, axis=0)     # (Formatting issues) Making the observation the first element of a batch of inputs 
        state = np.stack((obs, obs), axis=1)

        # update model every 20 steps
        if t>20 and t%20==0:
            len_D = len(D)
            mb_size = len_D if len_D < 50 else 50
            minibatch = random.sample(D, mb_size)                              # Sample some moves

            inputs_shape = (mb_size,) + state.shape[1:]
            inputs = np.zeros(inputs_shape)
            targets = np.zeros((mb_size, 1))

            for i in range(0, mb_size):
                state = minibatch[i][0]
                action = minibatch[i][1]
                reward = minibatch[i][2]
                state_new = minibatch[i][3]
                done = minibatch[i][4]
                
                # Build Bellman equation for the Q function
                inputs[i:i+1] = np.expand_dims(state, axis=0)
                target = model.predict(state)
                targets[i] = target[0][0]
                Q_sa = model.predict(state_new)
                
                if done:
                    targets[i, 0] = reward
                else:
                    targets[i, 0] = reward + gamma * Q_sa[0][0]

                # Train network to output the Q function
                model.train_on_batch(inputs, targets)
            print('Learning Finished')

print('Observing Finished')
