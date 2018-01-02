# -*- coding:utf-8 -*-
'''
interactions with wechat game, running on phone.
'''
import os
import subprocess
from PIL import Image
import numpy as np
import cv2
import copy
from rec import MeterValueReader
import time

def sh(command):
    p = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    p.wait()
    #print p.stdout.read()
    return p.returncode

class ObservationSpace:
	def __init__(self, shape):
		self.shape = shape

class ActionSpace:
	def __init__(self, n):
		self.n = n

class JumpGame:
	def __init__(self):
		self.score = 0
		self.obs_space_shape = (135, 240, 3)
		self.observation_space = ObservationSpace(self.obs_space_shape)
		self.action2time = ['200', '300', '400', '500', '600', '700', '800', '900']
		self.action_space = ActionSpace(len(self.action2time))
		self.game_start_btn_coord = (530, 1580)
		self.data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
		self.score_reader = MeterValueReader()
		if not os.path.exists(self.data_dir):
			os.makedirs(self.data_dir)

	def get_score(self, image):
		img = np.array(image, dtype=np.float32)
		img = np.delete(img, 3, axis=2)
		score_img = img[180:300, 80:240, :]
		cv2.imwrite('/tmp/score.jpeg', score_img)
		img_bw = cv2.imread(os.path.join('/tmp/score.jpeg'), 0)

		return int(self.score_reader.get_value(img_bw))

	def check_finish(self, image):
		w, h, _ = self.obs_space_shape
		image.thumbnail((w,h), Image.ANTIALIAS)
		#im.save('/tmp/test_resize.jpeg', "JPEG")
		in_ = np.array(image, dtype=np.float32)
		observation = np.delete(in_, 3, axis=2)

		is_finished = False
		im_mean = observation.mean(1).mean(0)
		if(im_mean[0] < 100) and (im_mean[1] < 100) and (im_mean[2] < 100):
			is_finished = True

		observation = np.transpose(observation, (1,0,2))

		return is_finished, observation

	def get_state(self):
		return_code = sh('adb shell screencap -p /sdcard/wechat-game-jump-state.png')
		if return_code != 0:
			print('game screenshot failed.')
			return -1
		else:
			print('executed screenshoting')

		return_code =sh('adb pull /sdcard/wechat-game-jump-state.png ' + self.data_dir)
		if return_code != 0:
			print('retrieve game screen png failed.')
			return -1
		else:
			print('fetched screenshot img')

		im = Image.open(os.path.join(self.data_dir, 'wechat-game-jump-state.png'))
		im_copy = copy.deepcopy(im)
		is_finished, observation = self.check_finish(im)

		score = -10
		if not is_finished:
			score = self.get_score(im_copy)

		return observation, score, is_finished

	def reset(self):
		self.score = 0
		# at this moment, make sure game is on the 'ready to start' screen.
		return_code = subprocess.call('adb shell input tap ' + \
			str(self.game_start_btn_coord[0]) + ' ' + str(self.game_start_btn_coord[1]), shell=True)
		if return_code == 0:
			print('game is reset')
		else:
			print('game reset failed.')
			return -1

		# wait till it takes effect
		time.sleep(1)

		# next grab the screenshot as the initial observation(state).
		obs, _, _ = self.get_state()

		return obs

	def step(self, action):
		# take action
		return_code = sh('adb shell input swipe 530 1580 530 1580 ' + self.action2time[action])
		if return_code != 0:
			print('failed while step forward.')
			return -1
		else:
			print('stepped forward ' + self.action2time[action] + 'ms')

		# wait till it takes effect
		time.sleep(3)

		# get state
		obs, score, is_finished = self.get_state()
		if score == -10:
			self.score = -10
		else:
			score -= self.score
			self.score += score

		# add info to align with atari gym API
		info = None
		return obs, score, is_finished, info

	def render(self):
		print('render')

if __name__ == '__main__':
	game = JumpGame()
	game.reset()