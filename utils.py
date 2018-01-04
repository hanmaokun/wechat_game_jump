# -*- coding:utf-8 -*-

import os, sys
import fnmatch
import xml.etree.cElementTree as ET
import argparse
import cv2
import numpy as np
from copy import deepcopy
import random
from matplotlib import pyplot as plt
import shutil
import csv
import math
import pickle

reload(sys)
sys.setdefaultencoding("utf-8")

# Deal with cmdline argument first
parser = argparse.ArgumentParser(description="Clean annotation data.")

def check_pkl(src_file):
    pkl_file = open(src_file, 'rb')
    content = pickle.load(pkl_file)
    for per_d in content:
        print(per_d[1])

def seg_pig(src_img):
    import numpy as np
    from PIL import Image
    import matplotlib.pyplot as plt
    import caffe

    caffe.set_mode_gpu()
    caffe.set_device(1)

    # load image, switch to BGR, subtract mean, and make dims C x H x W for Caffe
    #im = Image.open('/home/nlp/bigsur/data/jd_pig/output/demo/jd_pig_demo_2.png')
    im = Image.open(src_img)
    in_ = np.array(im, dtype=np.float32)
    in_ = in_[:,:,::-1]
    in_ = np.delete(in_, 0, axis=2)
    #in_ -= np.array((104.00698793,116.66876762,122.67891434))
    #in_ -= np.array((87.76231951,95.93502808,113.34667416))
    in_ = in_.transpose((2,0,1))

    # load net
    net = caffe.Net('/home/nlp/bigsur/devel/fcn.berkeleyvision.org/voc-fcn8s/deploy.prototxt', '/home/nlp/Downloads/fcn8s-heavy-pascal.caffemodel', caffe.TEST)
    # shape for input (data blob is N x C x H x W), set data
    net.blobs['data'].reshape(1, *in_.shape)
    net.blobs['data'].data[...] = in_
    # run net and take argmax for prediction
    net.forward()
    out = net.blobs['score'].data[0].argmax(axis=0)

    z_ = np.zeros_like(out)
    o_ = np.ones_like(out)

    #out_ = [(x if c else y) for x,y,c in zip(z_.all(), o_.all(), out.all())]

    plt.imshow(out,cmap='gray');
    plt.axis('off')

    plt.savefig(src_img.split('/')[-1].split('.')[0] + '_seg.png')

if __name__ == '__main__':
    random.seed(200)
    parser.add_argument("--src", default="/home/nlp/bigsur/devel/wechat-games/jump/1.png")
    args = parser.parse_args()
    SRC = args.src
    #seg_pig(SRC)
    check_pkl('/home/nlp/bigsur/devel/wechat-games/jump/data/D-manul.pkl')