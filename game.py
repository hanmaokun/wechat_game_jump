# -*- coding:utf-8 -*-
'''
interactions with wechat game, running on phone.
'''
import os
import subprocess
from PIL import Image
import numpy as np

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
		self.obs_space_shape = (135, 240)
		self.observation_space = ObservationSpace(self.obs_space_shape)
		self.action_space = ActionSpace(8)
		self.game_start_btn_coord = (530, 1580)
		self.data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
		if not os.path.exists(self.data_dir):
			os.makedirs(self.data_dir)

	def reset(self):
		# at this moment, make sure game is on the 'ready to start' screen.
		return_code = subprocess.call('adb shell input tap ' + \
			str(self.game_start_btn_coord[0]) + ' ' + str(self.game_start_btn_coord[1]), shell=True)
		if return_code == 0:
			print('game is reset')
		else:
			print('game reset failed.')
			return -1

		# next grab the screenshot as the initial observation(state).
		#return_code = subprocess.call('adb shell screencap -p /sdcard/wechat-game-jump-state.png')
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
		im.thumbnail(self.obs_space_shape, Image.ANTIALIAS)
		#im.save('/tmp/test_resize.jpeg', "JPEG")
		in_ = np.array(im, dtype=np.float32)
		in_ = np.delete(in_, 3, axis=2)

		return in_

	def step(self, action):
		print('step forward')

	def render(self):
		print('render')

if __name__ == '__main__':
	game = JumpGame()
	game.reset()