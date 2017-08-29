# -*- coding: utf-8 -*-

""" 
    Connection Area Detection
    2017.8.28
"""

from __future__ import division, print_function, absolute_import
import matplotlib.pyplot as plt
from skimage import data,color,morphology

import tensorflow as tf

# Added by Le Ning on July 27,2017
tf.app.flags.DEFINE_string('load_pic','m1.png',"marked picture")
tf.app.flags.DEFINE_string('save_pic','c1.jpg',"connected picture")
FLAGS = tf.app.flags.FLAGS

import matplotlib.pyplot as plt
from skimage import data,color,morphology,feature

#load target image
from skimage import io
img = io.imread(FLAGS.load_pic)
#io.imshow(img)

#Detect canny edge & generate binary image
img = color.rgb2gray(img)
edgs = feature.canny(img, sigma = 3) 
chull = morphology.convex_hull_object(edgs)

'''
#draw outlines
fig, axes = plt.subplots(1,2,figsize=(8,8))
ax0, ax1 = axes.ravel()
ax0.imshow(edgs,plt.cm.gray)
ax0.set_title('many objects')
ax1.imshow(chull,plt.cm.gray)
ax1.set_title('convex_hull image')
plt.show()
'''

#save results;
img = 255 * chull
io.imsave(FLAGS.save_pic,img)
