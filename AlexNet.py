# -*- coding: utf-8 -*-
""" 
This source code is created by Le Ning (lening@sjtu.edu.cn) on August 30, 2017.
This is training code based on alexnet.
"""

from __future__ import division, print_function, absolute_import
import tensorflow as tf														
import tflearn

'''
from tflearn.data_utils import shuffle, to_categorical
from tflearn.layers.core import input_data, dropout, flatten
from tflearn.layers.conv import conv_2d, max_pool_2d, avg_pool_2d
from tflearn.layers.estimator import regression
'''

#define paramaters
tf.app.flags.DEFINE_string('train_path','../data/postsamples/',"directory which include training cancer or no-cancer pictures")
tf.app.flags.DEFINE_string('test_path','../data/testsamples/',"directory which include testing cancer or no-cancer pictures")
tf.app.flags.DEFINE_integer('sample_size', 64, "sample size")
tf.app.flags.DEFINE_integer('nepoch', 100, "epoch number")
tf.app.flags.DEFINE_integer('batchsize', 256, "batch size number")
FLAGS = tf.app.flags.FLAGS

# Data loading
# Load path/class_id image file:
train_file = FLAGS.train_path + 'filelist.txt'
test_file = FLAGS.test_path + 'filelist.txt'

# Build the preloader array, resize images to sample size
from tflearn.data_utils import image_preloader
X, Y = image_preloader(train_file, image_shape=(FLAGS.sample_size, FLAGS.sample_size),
                       mode='file', categorical_labels=True,
                       normalize=True)

testX, testY = image_preloader(test_file, image_shape=(FLAGS.sample_size, FLAGS.sample_size),
                       mode='file', categorical_labels=True,
                       normalize=True)

num_classes = 2

# preprocessing
img_prep = tflearn.ImagePreprocessing()

# Real-time data augmentation
img_aug = tflearn.ImageAugmentation()
img_aug.add_random_flip_leftright()

# Building 'alexnet network'
network = tflearn.input_data(shape=[None, FLAGS.sample_size, FLAGS.sample_size, 3],
                         data_preprocessing=img_prep,
                         data_augmentation=img_aug)
network = tflearn.conv_2d(network, 96, 11, strides=4, activation='relu')
network = tflearn.max_pool_2d(network, 3, strides=2)
network = tflearn.local_response_normalization(network)
network = tflearn.conv_2d(network, 256, 5, activation='relu')
network = tflearn.max_pool_2d(network, 3, strides=2)
network = tflearn.local_response_normalization(network)
network = tflearn.conv_2d(network, 384, 3, activation='relu')
network = tflearn.conv_2d(network, 384, 3, activation='relu')
network = tflearn.conv_2d(network, 256, 3, activation='relu')
network = tflearn.max_pool_2d(network, 3, strides=2)
network = tflearn.local_response_normalization(network)
network = tflearn.fully_connected(network, 4096, activation='tanh')
network = tflearn.dropout(network, 0.8)
network = tflearn.fully_connected(network, 4096, activation='tanh')
network = tflearn.dropout(network, 0.8)
network = tflearn.fully_connected(network, num_classes, activation='softmax')
network = tflearn.regression(network, optimizer='momentum',
                         loss='categorical_crossentropy',
                         learning_rate=0.0001)

# Training
model = tflearn.DNN(network, checkpoint_path='model_alexnet_tumor',
                    max_checkpoints=10, tensorboard_verbose=0,
                    clip_gradients=0.)

model.load('model_alexnet_tumor-102000')
model.fit(X, Y, n_epoch=FLAGS.nepoch,validation_set=(testX,testY),
          snapshot_epoch=False, snapshot_step=1000,
          show_metric=True, batch_size=FLAGS.batchsize, shuffle=True,
          run_id='alexnet_tumor')

model.save('fine_alex_net01')
