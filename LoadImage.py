# -*- coding: utf-8 -*-
""" 
This source code is created by Le Ning (lening@sjtu.edu.cn) on September 5, 2017.
This is image prediction code based on alexnet.
"""

from __future__ import division, print_function, absolute_import
import tensorflow as tf														
import tflearn
from PIL import Image
import numpy as np
import os

#define paramaters
tf.app.flags.DEFINE_string('targetfile','predict.tiff',"Target picture")
tf.app.flags.DEFINE_string('savepath','../data/blocks/',"Target picture")
tf.app.flags.DEFINE_string('modelfile','fine_alex_net01',"model file")
tf.app.flags.DEFINE_integer('sample_size', 64, "sample size")
tf.app.flags.DEFINE_integer('block_width', 64, "block width")
tf.app.flags.DEFINE_integer('block_height', 64, "block height")
tf.app.flags.DEFINE_integer('stepX', 64, "step in horizontal direction")
tf.app.flags.DEFINE_integer('stepY', 64, "step in vertical direction")
tf.app.flags.DEFINE_float('IOU_threshold', 0.3, "step in vertical direction")
tf.app.flags.DEFINE_integer('offsetX', 0, "step in horizontal direction")
tf.app.flags.DEFINE_integer('offsetY', 0, "step in vertical direction")
tf.app.flags.DEFINE_boolean('blockgenerate', True, "generate blocks")
FLAGS = tf.app.flags.FLAGS

#Directly read data from images;
def DirectReadImg(filename, Imgheight=FLAGS.sample_size, Imgwidth=FLAGS.sample_size, normlized=True):

    #Load image from disk
    img=np.array(Image.open(filename).resize((Imgheight,Imgwidth)))
    
    if(normlized == True):
        img_normed = img/255.0
        return img_normed.reshape(1,Imgheight,Imgwidth,3)
    else:
        return img.reshape(1,Imgheight,Imgwidth,3)

def create_alexnet(num_classes):

    # Building 'AlexNet'
    network = tflearn.input_data(shape=[None, FLAGS.sample_size, FLAGS.sample_size, 3])
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
                         learning_rate=0.001)
    return network

# Generate Image Blocks;
def BlockGenerate(FileName=FLAGS.targetfile):

    #split orginal image to blocks 
    pil_im = Image.open(FileName)
    xRange = int(pil_im.width / FLAGS.stepX) 
    yRange = int(pil_im.height / FLAGS.stepY)

    left = FLAGS.offsetX
    right = FLAGS.block_width + FLAGS.offsetX
    upper = FLAGS.offsetY
    lower = FLAGS.block_height + FLAGS.offsetY

    count = 0
    boxList = []
    for i in range(0, yRange):
        for j in range(0, xRange):

            #generate image blocks;
            box = (left, upper, right, lower)
            boxList.append(box)

            region = pil_im.crop(box)
            
            temfile1 = ' '
            if(count>999):
                temfile1='%d'%count
            elif(count>99):
                temfile1='0%d'%count
            elif(count>9):
                temfile1='00%d'%count
            else:
                temfile1='000%d'%count
            Targetfile = FileName[:-4] + temfile1 +'.jpg' 
            region.save(FLAGS.savepath + Targetfile)
         
            left = left + FLAGS.stepX
            right = right + FLAGS.stepX
            count = count + 1
            
            if(right > pil_im.width):
                break

        left = FLAGS.offsetX
        right = FLAGS.block_width + FLAGS.offsetX
        upper = upper + FLAGS.stepY
        lower = lower + FLAGS.stepY
        
        if(lower > pil_im.height):
            break

    return boxList

def IsSubString(SubStrList,Str):
    
    flag=True
    for substr in SubStrList:
        if not(substr in Str):
            flag=False
    return flag

def RecognizeImgList(FindPath=FLAGS.savepath,FlagStr=[]):
    
    #load model
    network = create_alexnet(2)
    model = tflearn.DNN(network)
    model.load(FLAGS.modelfile)

    #Generate result image
    msize = 2048
    toImage = Image.new('RGBA', (msize, msize))

    FileList=[]
    FileNames=os.listdir(FindPath)
    
    xcount = 0
    ycount = 0
    if (len(FileNames)>0):
        FileNames.sort()
        for fn in FileNames:
            if (len(FlagStr)>0):
                if (IsSubString(FlagStr,fn)):
                    X = DirectReadImg(FindPath + fn)
                    pred = model.predict([X[0]])

                    if(pred[0][1]>pred[0][0]):
                        fromImage = Image.open(FindPath + fn)
                        toImage.paste(fromImage,(xcount * FLAGS.sample_size, ycount * FLAGS.sample_size))
                    else: 
                        fromImage = Image.open('blackblock.jpg')
                        toImage.paste(fromImage,(xcount * FLAGS.sample_size, ycount * FLAGS.sample_size))
 
                    FileList.append(fn)
                else:
                    X = DirectReadImg(FindPath + fn)
                    pred = model.predict([X[0]])

                    if(pred[0][1]>pred[0][0]):
                        fromImage = Image.open(FindPath + fn)
                        toImage.paste(fromImage,(xcount * FLAGS.sample_size, ycount * FLAGS.sample_size))
                    else: 
                        fromImage = Image.open('blackblock.jpg')
                        toImage.paste(fromImage,(xcount * FLAGS.sample_size, ycount * FLAGS.sample_size))
 
                    FileList.append(fn)

            xcount = xcount + 1
            if(xcount == 32):
                xcount = 0
                ycount = ycount + 1

    toImage.save('result.jpg')

    return FileList

if __name__ == '__main__':
    
    if(FLAGS.blockgenerate):
        print(BlockGenerate())
    else:    
        print(RecognizeImgList(FLAGS.savepath, 'jpg'))       

