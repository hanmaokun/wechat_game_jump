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

GAME_SCREEN_XMIN = 0
GAME_SCREEN_XMAX = 240
GAME_SCREEN_YMIN = 64
GAME_SCREEN_YMAX = 384

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
		self.best_score = 0
		self.average_round = 0
		self.obs_space_shape = (135, 240, 3)
		self.observation_space = ObservationSpace(self.obs_space_shape)
		self.action_space = ActionSpace(20)
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

		return int(self.score_reader.get_value0(img_bw))

	def check_finish(self, image):
		#w, h, _ = self.obs_space_shape
		w = 240
		h = 426
		image.thumbnail((w,h), Image.ANTIALIAS)
		#im.save('/tmp/test_resize.jpeg', "JPEG")
		in_ = np.array(image, dtype=np.float32)
		observation = np.delete(in_, 3, axis=2)
		observation_ = copy.deepcopy(observation)

		is_finished = False
		observation = observation[0:h/3, :, :]
		im_mean = observation.mean(1).mean(0)
		if(im_mean[0] < 100) and (im_mean[1] < 100) and (im_mean[2] < 100):
			is_finished = True

		#observation_ = np.transpose(observation_, (1,0,2))
		observation_ = observation_[GAME_SCREEN_YMIN:GAME_SCREEN_YMAX, GAME_SCREEN_XMIN:GAME_SCREEN_XMAX]

		return is_finished, observation_

	def get_state(self):
		return_code = sh('adb shell screencap -p /sdcard/wechat-game-jump-state.png')
		if return_code != 0:
			print('game screenshot failed.')
			return -1
		#else:
		#	print('executed screenshoting')

		return_code =sh('adb pull /sdcard/wechat-game-jump-state.png ' + self.data_dir)
		if return_code != 0:
			print('retrieve game screen png failed.')
			return -1
		#else:
		#	print('fetched screenshot img')

		im = Image.open(os.path.join(self.data_dir, 'wechat-game-jump-state.png'))
		im_copy = copy.deepcopy(im)
		is_finished, observation = self.check_finish(im)

		score = -10
		if not is_finished:
			score = self.get_score(im_copy)

		return observation, score, is_finished

	def reset(self):
		print('BEST SCORE: ' + str(self.best_score))
		self.score = 0
		# at this moment, make sure game is on the 'ready to start' screen.
		return_code = subprocess.call('adb shell input tap ' + \
			str(self.game_start_btn_coord[0]) + ' ' + str(self.game_start_btn_coord[1]), shell=True)
		if return_code == 0:
			print('####################GAME IS RESET####################')
		else:
			print('game reset failed.')
			return -1

		# wait till it takes effect
		time.sleep(1)

		# next grab the screenshot as the initial observation(state).
		obs, _, _ = self.get_state()

		return obs

	def action_2_time(self, act):
		return int(300 + 600*act)

	def step(self, action):
		# take action
		return_code = sh('adb shell input swipe 530 1580 530 1580 ' + str(int(action*40)))
		#return_code = sh('adb shell input swipe 530 1580 530 1580 ' + str(self.action_2_time(action)))
		if return_code != 0:
			print('failed while step forward.')
			return -1
		else:
			print('stepped forward ' + str(int(action*40)) + 'ms')
			#print('stepped forward ' + str(self.action_2_time(action)) + 'ms')

		# wait till it takes effect
		time.sleep(3)

		# get state
		obs, score, is_finished = self.get_state()
		if score == -10:
			self.score = -10
		else:
			if score > self.best_score:
				self.best_score = score
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