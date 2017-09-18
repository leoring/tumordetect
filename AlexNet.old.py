# -*- coding: utf-8 -*-
""" 
This source code is created by Le Ning (lening@sjtu.edu.cn) on August 30, 2017.
This is training code based on alexnet.
"""

from __future__ import division, print_function, absolute_import
import tensorflow as tf														
import tflearn


from tflearn.data_utils import shuffle, to_categorical
'''
from tflearn.layers.core import input_data, dropout, flatten
from tflearn.layers.conv import conv_2d, max_pool_2d, avg_pool_2d
from tflearn.layers.estimator import regression
'''

#define paramaters
tf.app.flags.DEFINE_string('file_path','../data/postsamples/',"directory which include original cancer or no-cancer pictures")
tf.app.flags.DEFINE_integer('sample_size', 64, "sample size")
tf.app.flags.DEFINE_integer('nepoch', 500, "epoch number")
tf.app.flags.DEFINE_integer('batchsize', 256, "batch size number")
FLAGS = tf.app.flags.FLAGS

# Data loading
# Load path/class_id image file:
dataset_file = FLAGS.file_path + 'filelist.txt'

# Build the preloader array, resize images to sample size
from tflearn.data_utils import image_preloader
X, Y = image_preloader(dataset_file, image_shape=(FLAGS.sample_size, FLAGS.sample_size),
                       mode='file', categorical_labels=True,
                       normalize=True)

num_classes = 2

#X, Y = shuffle(X, Y)

# Real-time data preprocessing
img_prep = tflearn.ImagePreprocessing()
img_prep.add_featurewise_zero_center()
#img_prep.add_featurewise_stdnorm()

# Real-time data augmentation
img_aug = tflearn.ImageAugmentation()
img_aug.add_random_flip_leftright()
img_aug.add_random_rotation(max_angle=25.)

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
network = tflearn.dropout(network, 0.5)
network = tflearn.fully_connected(network, 4096, activation='tanh')
network = tflearn.dropout(network, 0.5)
network = tflearn.fully_connected(network, num_classes, activation='softmax')
network = tflearn.regression(network, optimizer='momentum',
                         loss='categorical_crossentropy',
                         learning_rate=0.0001)

# Training
model = tflearn.DNN(network, checkpoint_path='model_alexnet_tumor',
                    max_checkpoints=10, tensorboard_verbose=0,
                    clip_gradients=0.)

model.load('model_alexnet_tumor-246000')
model.fit(X, Y, n_epoch=FLAGS.nepoch,validation_set=0.1,
          snapshot_epoch=False, snapshot_step=1000,
          show_metric=True, batch_size=FLAGS.batchsize, shuffle=True,
          run_id='alexnet_tumor')

model.save('fine_alex_net03')
