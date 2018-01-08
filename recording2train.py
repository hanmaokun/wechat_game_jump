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
import fnmatch
import xml.etree.cElementTree as ET
from rec import MeterValueReader
from collections import deque                # For storing moves 

import keras
from keras.models import Model
from keras.models import Input
from keras.models import Sequential          # One layer after the other
from keras.layers import Dense, Flatten      # Dense layers are fully connected layers, Flatten layers flatten out multidimensional inputs
from keras.models import model_from_json

reload(sys)
sys.setdefaultencoding("utf-8")

# specific params related to recorded video
IGNORE_FIRST_FRAMES_N = 50
MAX_VALID_FRAMES = 1040

TAP_REGION_XMIN = 64
TAP_REGION_XMAX = 160
TAP_REGION_YMIN = 0
TAP_REGION_YMAX = 70
# TAP_REGION_XMIN = 50
# TAP_REGION_XMAX = 190
# TAP_REGION_YMIN = 360
# TAP_REGION_YMAX = 426

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

    score_str = score_reader.get_value0(img_bw)
    if score_str == '':
        q_reward = -1
    else:
        q_reward = int(score_str)

    return q_reward, input_img[GAME_SCREEN_YMIN:GAME_SCREEN_YMAX, GAME_SCREEN_XMIN:GAME_SCREEN_XMAX]

def check_finish(num, image_data):
    img_check_finish = image_data[TAP_REGION_YMIN:TAP_REGION_YMAX, TAP_REGION_XMIN:TAP_REGION_XMAX]
    mean_val = img_check_finish.mean(1).mean(0)
    rgb_mean = (np.mean(mean_val, dtype=np.float64))

    if rgb_mean < 80:
        return True
    else:
        return False

def check_started(image_data, score_reader):
    img_score = image_data[SCORE_REGION_YMIN:SCORE_REGION_YMAX, SCORE_REGION_XMIN:SCORE_REGION_XMAX]
    cv2.imwrite('/tmp/score_start.jpeg', img_score)
    img_bw = cv2.imread(os.path.join('/tmp/score_start.jpeg'), 0)

    #q_reward = int(score_reader.get_value0(img_bw))
    score_str = score_reader.get_value0(img_bw)
    if len(score_str):
        print('not started yet')
        if int(score_str) == 0:
            return True
    
    return False

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
            loaded_model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])

            return loaded_model

    # a naive sequential model
    # model = Sequential()
    # model.add(Dense(20, input_shape=(2,) + (320, 240, 3), init='uniform', activation='relu'))
    # model.add(Flatten())                           # Flatten input so as to have no problems with processing
    # model.add(Dense(20, init='uniform', activation='relu'))
    # model.add(Dense(10, init='uniform', activation='relu'))

    # model.add(Dense(20, init='uniform', activation='linear'))    # Same number of outputs as possible actions
    # model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])

    # finetuned vgg16 model
    input_tensor = Input(shape=(320, 240, 3))
    base_model = keras.applications.vgg16.VGG16(include_top=False, weights='imagenet', input_tensor=input_tensor, pooling=None, classes=1000)
    for layer in base_model.layers:
    	layer.trainable = False
    top_model = Sequential()
    top_model.add(Flatten(input_shape=base_model.output_shape[1:]))
    top_model.add(Dense(20, init='uniform', activation='relu'))
    top_model.add(Dense(20, init='uniform', activation='sigmoid'))
    model = Model(inputs= base_model.input, outputs= top_model(base_model.output))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model

def recording2traindata(src_mp4_file, score_reader):
    store_d = deque()
    pkl_file_path = './data/' + src_mp4_file.split('/')[-1].split('.')[0] + '.pkl'
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
    is_stared = False

    for num, image in enumerate(vid):

        if not is_stared:
            is_stared = check_started(copy.deepcopy(image), score_reader)
            continue

        else:
            is_finished = check_finish(num, copy.deepcopy(image))
            if is_finished:    
                q_action = tap_time - 11
                q_reward = -10
                store_d.append((q_state, q_action, q_reward, q_state, 1))
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
                        if reward_cur == -1:
                            reward_cur = reward_prev
                        q_reward = reward_cur - reward_prev
                        q_action = tap_time
                        observation_new = image_prev[GAME_SCREEN_YMIN:GAME_SCREEN_YMAX, GAME_SCREEN_XMIN:GAME_SCREEN_XMAX]
                        print('tap time: ' + str(tap_time))
                        # See state of the game, reward... after performing the action
                        #obs_new = np.expand_dims(observation_new, axis=0)          # (Formatting issues)
                        #q_state_new = np.append(np.expand_dims(obs_new, axis=0), q_state[:, :1, :], axis=1)     # Update the input with the new state of the game
                        q_state_new = observation_new
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
                #obs = np.expand_dims(observation, axis=0)     # (Formatting issues) Making the observation the first element of a batch of inputs 
                #q_state = np.stack((obs, obs), axis=1)
                q_state = observation
                state_cur = S_IDLE

            state_prev = state_cur
            image_prev = copy.deepcopy(image)
            #cv2.imwrite('/tmp/image_prev.jpeg', image)

    print('training data recorded to ' + pkl_file_path)
    vid.close()

    return pkl_file_path, len(store_d)

def train(store_d_file, model):
    pkl_file_D = open(store_d_file, 'rb')
    store_d = pickle.load(pkl_file_D)
    len_D = len(store_d)
    mb_size = len_D if len_D < 20 else 20
    minibatch = random.sample(store_d, mb_size)                              # Sample some moves

    epsilon = 0.7                              # Probability of doing a random move
    gamma = 0

    state = minibatch[0][0]

    # inputs_shape = (mb_size,) + state.shape[1:]
    # inputs = np.zeros(inputs_shape)
    # targets = np.zeros((mb_size, 20))

    inputs_shape = (mb_size,) + state.shape
    inputs = np.zeros(inputs_shape)
    targets = np.zeros((mb_size, 20))

    for i in range(0, mb_size):
        state = minibatch[i][0]
        action = minibatch[i][1]
        reward = minibatch[i][2]
        state_new = minibatch[i][3]
        done = minibatch[i][4]
        
        if action > 10 and action < 30:
            action -= 10
            # Build Bellman equation for the Q function
            #inputs[i:i+1] = np.expand_dims(state, axis=0)
            inputs[i:i+1] = np.expand_dims(state, axis=0)
            #target = model.predict(state)
            #targets[i] = target[0][0]
            target = model.predict(inputs[i:i+1])
            targets[i] = target[0]
            Q_sa = model.predict(np.expand_dims(state_new, axis=0))
            print(np.argmax(Q_sa))
            if done:
                targets[i, action] = reward
            else:
                targets[i, action] = reward + gamma * np.max(Q_sa)
                #targets[i, action] = reward + gamma * Q_sa[0][0]

            # Train network to output the Q function
    model.train_on_batch(inputs, targets)

    pkl_file_D.close()
    print('Learning Finished')

if __name__ == '__main__':
    random.seed(200)
    parser = argparse.ArgumentParser(description="Clean annotation data.")
    parser.add_argument("--src", default="./data/2018_01_03_07_28_41.mp4")
    args = parser.parse_args()
    src_file = args.src
    score_reader = MeterValueReader()

    #match_touch(cv2.imread('/home/nlp/bigsur/devel/wechat-games/jump/data/tap_down0.png'))
    #get_cur_q('/home/nlp/bigsur/devel/wechat-games/jump/data/tap_none.png', score_reader)
    #check("./data/2018_01_03_07_28_41.mp4")

    save_model_name = 'jump_model'

    model = model_init(save_model_name)
    samples_ctr = 0
    for root, dir_names, file_names in os.walk('./data/test/'):
        for mp4_file_name in fnmatch.filter(file_names, '*.mp4'):
            mp4_file_path = os.path.join(root, mp4_file_name)
            print('processing with ' + mp4_file_path)
            stored_d, num_samples = recording2traindata(mp4_file_path, score_reader)
            train(stored_d, model)
            samples_ctr += num_samples

    # serialize model to JSON
    model_json = model.to_json()
    with open('./model/' + save_model_name + '.json', "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights('./model/' + save_model_name + '.h5')

    print('training finished with all ' + str(samples_ctr) + ' samples, saved to ' + save_model_name)
