# -*- coding:utf-8 -*-
'''
processing with recorded screen on phone, while playing game.
'''

import numpy as np
import imageio
import random
import argparse
import sys
import pylab
import copy
import cv2
import os,sys
import pickle
from rec import MeterValueReader
from collections import deque                # For storing moves 

from keras.models import Sequential          # One layer after the other
from keras.layers import Dense, Flatten      # Dense layers are fully connected layers, Flatten layers flatten out multidimensional inputs

reload(sys)
sys.setdefaultencoding("utf-8")

# specific params related to recorded video
IGNORE_FIRST_FRAMES_N = 50
MAX_VALID_FRAMES = 1040

TAP_REGION_XMIN = 64
TAP_REGION_XMAX = 160
TAP_REGION_YMIN = 0
TAP_REGION_YMAX = 70

SCORE_REGION_XMIN = 18
SCORE_REGION_XMAX = 76
SCORE_REGION_YMIN = 40
SCORE_REGION_YMAX = 66

GAME_SCREEN_XMIN = 0
GAME_SCREEN_XMAX = 240
GAME_SCREEN_YMIN = 64
GAME_SCREEN_YMAX = 384

# 
def check(src_mp4_file):
    vid = imageio.get_reader(src_mp4_file, 'ffmpeg')
    for num, image in enumerate(vid):
        if num < 1040:
            continue
        else:
            #imageio.imwrite('./data/tap_none_.png', image)
            fig = pylab.figure()
            pylab.imshow(image)
            timestamp = float(num)/ vid.get_meta_data()['fps']
            print(timestamp)
            fig.suptitle('image #{}, timestamp={}'.format(num, timestamp), fontsize=20)
            pylab.show()

def match_touch(image):
    template_img = cv2.imread(os.path.join('/home/nlp/bigsur/devel/wechat-games/jump/data/', 'tap_bw1.png'), cv2.CV_LOAD_IMAGE_GRAYSCALE)
    (thresh, template_img_bw) = cv2.threshold(template_img, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    #cv2.imshow('', np.asarray(template_img_bw))
    #cv2.waitKey()

    input_img = image[TAP_REGION_YMIN:TAP_REGION_YMAX, TAP_REGION_XMIN:TAP_REGION_XMAX]
    input_img_gray = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)

    thres = input_img_gray.mean(1).mean(0)
    thres = thres+10 if thres < 254 else 254

    (thresh, im_bw) = cv2.threshold(input_img_gray, thres, 255, cv2.THRESH_BINARY)
    #cv2.imshow('', np.asarray(im_bw))
    #cv2.waitKey()

    meth = 'cv2.TM_CCOEFF_NORMED'
    method = eval(meth)
    res = cv2.matchTemplate(im_bw, template_img_bw, method)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    #print(max_val)
    #print(max_loc)

    #w, h = template_img.shape[::-1]
    #top_left = max_loc
    #bottom_right = (top_left[0] + w, top_left[1] + h)
    #cv2.rectangle(im_bw,top_left, bottom_right, 128, 2)
    #cv2.imshow('', np.asarray(im_bw))
    #cv2.waitKey()

    if max_val > 0.3:
        return True
    else:
        return False

def get_cur_q(image, score_reader):
    input_img = image

    score_img = input_img[SCORE_REGION_YMIN:SCORE_REGION_YMAX, SCORE_REGION_XMIN:SCORE_REGION_XMAX]
    cv2.imwrite('/tmp/score.jpeg', score_img)
    img_bw = cv2.imread(os.path.join('/tmp/score.jpeg'), 0)

    q_reward = int(score_reader.get_value0(img_bw))
    print(q_reward)

    return q_reward, input_img[GAME_SCREEN_YMIN:GAME_SCREEN_YMAX, GAME_SCREEN_XMIN:GAME_SCREEN_XMAX]

def check_finish(num, image_data):
    if num >= 1040:
        return True
    else:
        return False

def recording2traindata(src_mp4_file, score_reader, store_d):
    # Create network. Input is two consecutive game states, output is Q-values of the possible moves.
    model = Sequential()
    model.add(Dense(20, input_shape=(2,) + (320, 240, 3), init='uniform', activation='relu'))
    model.add(Flatten())                           # Flatten input so as to have no problems with processing
    model.add(Dense(18, init='uniform', activation='relu'))
    model.add(Dense(10, init='uniform', activation='relu'))

    model.add(Dense(15, init='uniform', activation='linear'))    # Same number of outputs as possible actions
    model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
    epsilon = 0.7                              # Probability of doing a random move
    gamma = 0.9

    pkl_file_path = './data/D-manul.pkl'
    S_STARTED, S_IDLE, S_TAPPINGDOWN = range(3)
    state_cur = S_STARTED
    state_prev = S_STARTED

    tap_time = 0
    vid = imageio.get_reader(src_mp4_file, 'ffmpeg')
    image_prev = None
    q_state = None
    q_action = 0
    q_reward = 0
    reward_prev = 0
    q_state_new = None
    tap_ctr = 0

    for num, image in enumerate(vid):

        if num < IGNORE_FIRST_FRAMES_N:
            continue

        else:
            is_finished = check_finish(num, copy.deepcopy(image))
            if is_finished:    
                with open(pkl_file_path, "wb") as f:
                    pickle.dump(store_d, f, pickle.HIGHEST_PROTOCOL)
                    f.close()
                break

            tap_down = match_touch(copy.deepcopy(image))

            if tap_down:
                if state_prev == S_IDLE:
                    if tap_time > 0:
                        cv2.imwrite('/tmp/tap_' + str(tap_ctr) + '.jpeg', image)
                        tap_ctr += 1
                        reward_cur, obs_new = get_cur_q(copy.deepcopy(image), score_reader)
                        q_reward = reward_cur - reward_prev
                        q_action = tap_time - 11
                        observation_new = image_prev[GAME_SCREEN_YMIN:GAME_SCREEN_YMAX, GAME_SCREEN_XMIN:GAME_SCREEN_XMAX]

                        # See state of the game, reward... after performing the action
                        obs_new = np.expand_dims(observation_new, axis=0)          # (Formatting issues)
                        q_state_new = np.append(np.expand_dims(obs_new, axis=0), q_state[:, :1, :], axis=1)     # Update the input with the new state of the game

                        store_d.append((q_state, q_action, q_reward, q_state_new, 0))

                        q_state = q_state_new
                        reward_prev = reward_cur

                    state_cur = S_TAPPINGDOWN
                    tap_time = 1

                if state_prev == S_TAPPINGDOWN:
                    tap_time += 1

            elif state_prev == S_TAPPINGDOWN:
                state_cur = S_IDLE

            elif state_prev == S_STARTED:
                observation = image[GAME_SCREEN_YMIN:GAME_SCREEN_YMAX, GAME_SCREEN_XMIN:GAME_SCREEN_XMAX]
                obs = np.expand_dims(observation, axis=0)     # (Formatting issues) Making the observation the first element of a batch of inputs 
                q_state = np.stack((obs, obs), axis=1)
                state_cur = S_IDLE

            state_prev = state_cur
            image_prev = copy.deepcopy(image)

    print('training data recorded to ' + pkl_file_path)

    len_D = len(store_d)
    mb_size = len_D if len_D < 20 else 20
    minibatch = random.sample(store_d, mb_size)                              # Sample some moves

    state = minibatch[0][0]
    inputs_shape = (mb_size,) + state.shape[1:]
    inputs = np.zeros(inputs_shape)
    targets = np.zeros((mb_size, 15))

    for i in range(0, mb_size):
        state = minibatch[i][0]
        action = minibatch[i][1]
        reward = minibatch[i][2]
        state_new = minibatch[i][3]
        done = minibatch[i][4]
        
        # Build Bellman equation for the Q function
        inputs[i:i+1] = np.expand_dims(state, axis=0)
        #target = model.predict(state)
        #targets[i] = target[0][0]
        targets[i] = model.predict(state)
        Q_sa = model.predict(state_new)
        
        if done:
            targets[i, action] = reward
        else:
            targets[i, action] = reward + gamma * np.max(Q_sa)
            #targets[i, action] = reward + gamma * Q_sa[0][0]

        # Train network to output the Q function
        model.train_on_batch(inputs, targets)

    print('Learning Finished')

if __name__ == '__main__':
    random.seed(200)
    parser = argparse.ArgumentParser(description="Clean annotation data.")
    parser.add_argument("--src", default="./data/2018_01_03_07_28_41.mp4")
    args = parser.parse_args()
    src_file = args.src
    score_reader = MeterValueReader()
    D = deque()

    #match_touch(cv2.imread('/home/nlp/bigsur/devel/wechat-games/jump/data/tap_down0.png'))
    #get_cur_q('/home/nlp/bigsur/devel/wechat-games/jump/data/tap_none.png', score_reader)
    #check("./data/2018_01_03_07_28_41.mp4")
    recording2traindata("./data/2018_01_03_07_28_41.mp4", score_reader, D)
