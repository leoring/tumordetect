# -*- coding: utf-8 -*-
""" 
This source code is created by Le Ning (lening@sjtu.edu.cn) on August 30, 2017.
This is training code based on residual net.
"""

from __future__ import division, print_function, absolute_import
import tensorflow as tf													
import tflearn
from tflearn.data_utils import shuffle, to_categorical
from tflearn.layers.core import input_data, dropout, flatten
from tflearn.layers.conv import conv_2d, max_pool_2d, avg_pool_2d
from tflearn.layers.estimator import regression
import pickle
import numpy as np 
from PIL import Image
import os.path

#define paramaters
tf.app.flags.DEFINE_string('file_path','../data/postsamples/',"directory which include original cancer or no-cancer pictures")
tf.app.flags.DEFINE_integer('sample_size', 64, "sample size")
tf.app.flags.DEFINE_integer('nepoch', 100, "epoch number")
tf.app.flags.DEFINE_integer('batchsize', 128, "batch size number")
FLAGS = tf.app.flags.FLAGS

def load_image(img_path):
    img = Image.open(img_path)
    return img

def resize_image(in_image, new_width, new_height, out_image=None,
                 resize_mode=Image.ANTIALIAS):
    img = in_image.resize((new_width, new_height), resize_mode)
    if out_image:
        img.save(out_image)
    return img

def pil_to_nparray(pil_image):
    pil_image.load()
    return np.asarray(pil_image, dtype="float32")

def load_data(datafile, num_clss, save=False, save_path='dataset.pkl', normalize=True):
    train_list = open(datafile,'r')
    labels = []
    images = []
    for line in train_list:
        tmp = line.strip().split(' ')
        fpath = tmp[0]
        print(fpath)
        img = load_image(fpath)
        img = resize_image(img,FLAGS.sample_size,FLAGS.sample_size)
        np_img = pil_to_nparray(img)
        
        if(normalize==True):
             np_img = np_img/255.0
        images.append(np_img)

        index = int(tmp[1])
        label = np.zeros(num_clss)
        label[index] = 1
        labels.append(label)
    if save:
        pickle.dump((images, labels), open(save_path, 'wb'))
    return images, labels

# Data loading
# Load path/class_id image file:
dataset_file = FLAGS.file_path + 'filelist.txt'

# Build the preloader array, resize images to sample size
num_classes = 2
X, Y = load_data(dataset_file, num_classes)

# preprocessing
img_prep = tflearn.ImagePreprocessing()

# Real-time data augmentation
img_aug = tflearn.ImageAugmentation()
img_aug.add_random_flip_leftright()

# Building 'residual network'
n = 5
net = tflearn.input_data(shape=[None, FLAGS.sample_size, FLAGS.sample_size, 3],
                         data_preprocessing=img_prep,
                         data_augmentation=img_aug)

net = tflearn.conv_2d(net, 16, 3, regularizer='L2', weight_decay=0.0001)
net = tflearn.residual_block(net, n, 16)
net = tflearn.residual_block(net, 1, 32, downsample=True)
net = tflearn.residual_block(net, n-1, 32)
net = tflearn.residual_block(net, 1, 64, downsample=True)
net = tflearn.residual_block(net, n-1, 64)
net = tflearn.batch_normalization(net)
net = tflearn.activation(net, 'relu')
net = tflearn.global_avg_pool(net)

# Regression
net = tflearn.fully_connected(net, num_classes, activation='softmax')
mom = tflearn.Momentum(0.1, lr_decay=0.1, decay_step=30000, staircase=True)
net = tflearn.regression(net, optimizer=mom, loss='categorical_crossentropy')

# Training
model = tflearn.DNN(net, checkpoint_path='model_resnet_tumor',
                    max_checkpoints=10, tensorboard_verbose=0,
                    clip_gradients=0.)

model.load('model_resnet_tumor-208800')
model.fit(X, Y, n_epoch=FLAGS.nepoch,validation_set=0.1,
          snapshot_epoch=False, snapshot_step=200,
          show_metric=True, batch_size=FLAGS.batchsize, shuffle=True,
          run_id='resnet_tumor')

model.save('residual_net')

