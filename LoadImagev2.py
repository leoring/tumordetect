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
tf.app.flags.DEFINE_string('targetfile','2017-06-10_14.36.03.ndpi.16.27261_16508.2048x2048.tiff',"Target picture")
tf.app.flags.DEFINE_string('savepath','../data/blocks/',"Target picture")
tf.app.flags.DEFINE_string('modelfile','model_alexnet_tumor-246000',"model file")
tf.app.flags.DEFINE_integer('sample_size', 64, "sample size")
tf.app.flags.DEFINE_integer('block_width', 64, "block width")
tf.app.flags.DEFINE_integer('block_height', 64, "block height")
tf.app.flags.DEFINE_integer('stepX', 64, "step in horizontal direction")
tf.app.flags.DEFINE_integer('stepY', 64, "step in vertical direction")
tf.app.flags.DEFINE_float('IOU_threshold', 0.3, "step in vertical direction")
tf.app.flags.DEFINE_integer('offsetX', 0, "step in horizontal direction")
tf.app.flags.DEFINE_integer('offsetY', 0, "step in vertical direction")
tf.app.flags.DEFINE_boolean('blockgenerate', False, "generate blocks")
tf.app.flags.DEFINE_boolean('direction8', True, "8-neighborhood")
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

def create_cnn(num_classes):
    # Build 'cnn'
    network = tflearn.input_data(shape=[None, FLAGS.sample_size, FLAGS.sample_size, 3])
    network = tflearn.conv_2d(network, 20, 5, strides=1, activation='relu')
    network = tflearn.max_pool_2d(network, 2, strides=2)
    network = tflearn.conv_2d(network, 50, 5, strides=1, activation='relu')
    network = tflearn.max_pool_2d(network, 2, strides=2)
    network = tflearn.fully_connected(network, 500, activation='relu')
    #network = tflearn.dropout(network, 0.5)
    network = tflearn.fully_connected(network, num_classes, activation='softmax')
    network = tflearn.regression(network, optimizer='adam',
                         loss='categorical_crossentropy',
                         learning_rate=0.001)
    return network

def create_vgg(num_classes):
    # Building 'VGG Network'
    network = tflearn.input_data(shape=[None, FLAGS.sample_size, FLAGS.sample_size, 3])
    network = tflearn.conv_2d(network, 64, 3, activation='relu')
    network = tflearn.conv_2d(network, 64, 3, activation='relu')
    network = tflearn.max_pool_2d(network, 2, strides=2)

    network = tflearn.conv_2d(network, 128, 3, activation='relu')
    network = tflearn.conv_2d(network, 128, 3, activation='relu')
    network = tflearn.max_pool_2d(network, 2, strides=2)

    network = tflearn.conv_2d(network, 256, 3, activation='relu')
    network = tflearn.conv_2d(network, 256, 3, activation='relu')
    network = tflearn.conv_2d(network, 256, 3, activation='relu')
    network = tflearn.max_pool_2d(network, 2, strides=2)

    network = tflearn.conv_2d(network, 512, 3, activation='relu')
    network = tflearn.conv_2d(network, 512, 3, activation='relu')
    network = tflearn.conv_2d(network, 512, 3, activation='relu')
    network = tflearn.max_pool_2d(network, 2, strides=2)

    network = tflearn.conv_2d(network, 512, 3, activation='relu')
    network = tflearn.conv_2d(network, 512, 3, activation='relu')
    network = tflearn.conv_2d(network, 512, 3, activation='relu')
    network = tflearn.max_pool_2d(network, 2, strides=2)

    network = tflearn.fully_connected(network, 4096, activation='relu')
    network = tflearn.dropout(network, 0.5)
    network = tflearn.fully_connected(network, 4096, activation='relu')
    network = tflearn.dropout(network, 0.5)
    network = tflearn.fully_connected(network, num_classes, activation='softmax')

    network = tflearn.regression(network, optimizer='rmsprop',
                         loss='categorical_crossentropy',
                         learning_rate=0.0001)
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

def RecognizeImgList(FindPath = FLAGS.savepath,FlagStr=[]):
    
    #load alexnet model
    network = create_alexnet(2)
   
    #load cnn model
    #network = create_cnn(2)
    
    #load vgg model
    #network = create_vgg(2)

    model = tflearn.DNN(network)
    model.load(FLAGS.modelfile)

    tagList = []
    FileNames=os.listdir(FindPath)
   
    if (len(FileNames)>0):
        FileNames.sort()
        for fn in FileNames:
            if (len(FlagStr)>0):
                if (IsSubString(FlagStr,fn)):
                    X = DirectReadImg(FindPath + fn)
                    pred = model.predict([X[0]])

                    tag = 0
                    if(pred[0][1]>pred[0][0]):
                        tag = 1 

                    tagList.append(tag)
                else:
                    X = DirectReadImg(FindPath + fn)
                    pred = model.predict([X[0]])

                    tag = 0
                    if(pred[0][1]>pred[0][0]):
                        tag = 1
                   
                    tagList.append(tag)
    return tagList

def RemoveSingleNoises(tagList):
    
    newtagList = []
    row = int(2048/FLAGS.sample_size) 
    column = int(2048/FLAGS.sample_size) 
 
    if (len(tagList)>0):
        count = 0
        for tag in tagList:  
            i = int(count/row)
            j = int(count%row)

            newtag = 0 
            if(tag == 1):
                newtag = 1
  
                #4 conners
                if(i==0 and j==0):
                    if(FLAGS.direction8):
                        #8-neighborhood
                        if(tagList[(i+1)*row]==0 and tagList[i*row+(j+1)]==0 
                           and tagList[(i+1)*row+(j+1)]==0):
                            newtag = 0
                    else:
                        #4-neighborhood
                        if(tagList[(i+1)*row]==0 and tagList[(i+2)*row]==0 
                           and tagList[i*row+(j+1)]==0 and tagList[i*row+(j+2)]==0):
                            newtag = 0    

                elif(i==0 and j==row-1):
                    if(FLAGS.direction8):
                        #8-neighborhood
                        if(tagList[(i+1)*row]==0 and tagList[i*row+(j-1)]==0 
                           and tagList[(i+1)*row+(j-1)]==0):
                            newtag = 0
                    else:
                        #4-neighborhood
                        if(tagList[(i+1)*row]==0 and tagList[(i+2)*row]==0 
                           and tagList[i*row+(j-1)]==0 and tagList[i*row+(j-2)]==0):
                            newtag = 0  

                elif(i==column-1 and j==0):
                    if(FLAGS.direction8):
                        #8-neighborhood
                        if(tagList[(i-1)*row]==0 and tagList[i*row+(j+1)]==0 
                           and tagList[(i-1)*row+(j+1)]==0):
                            newtag = 0
                    else:
                        #4-neighborhood
                        if(tagList[(i-1)*row]==0 and tagList[(i-2)*row]==0 
                           and tagList[i*row+(j+1)]==0 and tagList[i*row+(j+2)]==0):
                            newtag = 0  

                elif(i==column-1 and j==row-1):
                    if(FLAGS.direction8):
                        #8-neighborhood
                        if(tagList[(i-1)*row]==0 and tagList[i*row+(j-1)]==0 
                           and tagList[(i-1)*row+(j-1)]==0):
                            newtag = 0
                    else:
                        #4-neighborhood
                        if(tagList[(i-1)*row]==0 and tagList[(i-2)*row]==0 
                           and tagList[i*row+(j-1)]==0 and tagList[i*row+(j-2)]==0):
                            newtag = 0

                #4 borders
                elif(i==0 and j>0 and j<row-1):
                    #Top
                    if(FLAGS.direction8):
                        #8-neighborhood
                        if(tagList[i*row+j-1]==0 and tagList[i*row+(j+1)]==0 
                           and tagList[(i+1)*row+j]==0 and tagList[(i+1)*row+(j-1)]==0 
                           and tagList[(i+1)*row+(j+1)]==0):
                            newtag = 0
                    else:
                        #4-neighborhood
                        if(tagList[i*row+j-1]==0 and tagList[i*row+(j+1)]==0 
                           and tagList[i*row+(j+2)]==0 and tagList[(i+1)*row+j]==0 
                           and tagList[(i+2)*row+j]==0):
                            newtag = 0

                elif(i>0 and i<column-1 and j==0):
                    #Left
                    if(FLAGS.direction8):
                        #8-neighborhood
                        if(tagList[(i-1)*row+j]==0 and tagList[(i+1)*row+j]==0 
                           and tagList[i*row+(j+1)]==0 and tagList[(i-1)*row+(j+1)]==0 
                           and tagList[(i+1)*row+(j+1)]==0):
                            newtag = 0
                    else:
                        #4-neighborhood
                        if(tagList[(i-1)*row+j]==0 and tagList[i*row+(j+1)]==0 
                           and tagList[i*row+(j+2)]==0 and tagList[(i+1)*row+j]==0 
                           and tagList[(i+2)*row+j]==0):
                            newtag = 0

                elif(i>0 and i<column-1 and j==row-1):
                    #Right
                    if(FLAGS.direction8):
                        #8-neighborhood
                        if(tagList[(i-1)*row+j]==0 and tagList[(i+1)*row+j]==0 
                           and tagList[i*row+(j-1)]==0 and tagList[(i-1)*row+(j-1)]==0 
                           and tagList[(i+1)*row+(j-1)]==0):
                            newtag = 0
                    else:
                        #4-neighborhood
                        if(tagList[(i-1)*row+j]==0 and tagList[i*row+(j-1)]==0 
                           and tagList[i*row+(j-2)]==0 and tagList[(i+1)*row+j]==0 
                           and tagList[(i+2)*row+j]==0):
                            newtag = 0

                elif(i==column-1 and j>0 and j<column-1):
                    #Bottom
                    if(FLAGS.direction8):
                        #8-neighborhood
                        if(tagList[i*row+(j-1)]==0 and tagList[i*row+(j+1)]==0 
                           and tagList[(i-1)*row+j]==0 and tagList[(i-1)*row+(j-1)]==0 
                           and tagList[(i-1)*row+(j+1)]==0):
                            newtag = 0
                    else:
                        #4-neighborhood
                        if(tagList[i*row+(j-1)]==0 and tagList[(i-1)*row+j]==0 
                           and tagList[(i-1)*row+(j-2)]==0 and tagList[i*row+(j+1)]==0 
                           and tagList[i*row+(j+2)]==0):
                            newtag = 0

                #inside 
                else:
                    if(FLAGS.direction8):
                        #8-neighborhood
                        if(i>=1 and j>=1 and i<=column-1 and j<=row-1):
                            l=0
                            if(tagList[i*row+(j-1)]==1):
                               l=l+1
                            if(tagList[i*row+(j+1)]==1):
                               l=l+1
                            if(tagList[(i-1)*row+j]==1):
                               l=l+1
                            if(tagList[(i+1)*row+j]==1):
                               l=l+1
                            if(tagList[(i-1)*row+(j-1)]==1):
                               l=l+1
                            if(tagList[(i-1)*row+(j+1)]==1):
                               l=l+1
                            if(tagList[(i+1)*row+(j-1)]==1):
                               l=l+1
                            if(tagList[(i+1)*row+(j+1)]==1):
                               l=l+1
                         
                            if(l<=1):   
                               newtag = 0
                    else:
                        #4-neighborhood
                        if(i>=1 and j>=1 and i<=column-1 and j<=row-1):                       
                           if(tagList[i*row+(j-1)]==0 and tagList[i*row+(j+1)]==0 
                              and tagList[(i-1)*row+j]==0 and tagList[(i+1)*row+j]==0):
                              newtag = 0
                                   
            count = count + 1
            newtagList.append(newtag)

    return newtagList

def MergeSplitAreas(tagList,threshold):
    
    newtagList = []
    row = int(2048/FLAGS.sample_size) 
    column = int(2048/FLAGS.sample_size) 
 
    if (len(tagList)>0):
        count = 0
        for tag in tagList:  
            i = int(count/row)
            j = int(count%row)

            newtag = 1 
            if(tag == 0):
                newtag = 0
                l=0
            
                #4 conners
                if(i==0 and j==0):
                    if(FLAGS.direction8):
                        #8-neighborhood
                        if(tagList[(i+1)*row]==1):
                            l=l+1
                        if(tagList[i*row+(j+1)]==1):
                            l=l+1
                        if(tagList[(i+1)*row+(j+1)]==1):
                            l=l+1
                         
                        if(l>1):
                            newtag = 1
                    else:
                        #4-neighborhood
                        if(tagList[(i+1)*row]==1):
                            l=l+1
                        if(tagList[(i+2)*row]==1):
                            l=l+1
                        if(tagList[i*row+(j+1)]==1):
                            l=l+1
                        if(tagList[i*row+(j+2)]==1):
                            l=l+1
                         
                        if(l>2):   
                            newtag = 1    

                elif(i==0 and j==row-1):
                    if(FLAGS.direction8):
                        #8-neighborhood
                        if(tagList[(i+1)*row]==1):
                            l=l+1
                        if(tagList[i*row+(j-1)]==1):
                            l=l+1
                        if(tagList[(i+1)*row+(j-1)]==1):
                            l=l+1
                         
                        if(l>1):   
                            newtag = 1
                    else:
                        #4-neighborhood
                        if(tagList[(i+1)*row]==1):
                            l=l+1
                        if(tagList[(i+2)*row]==1):
                            l=l+1
                        if(tagList[i*row+(j-1)]==1):
                            l=l+1
                        if(tagList[i*row+(j-2)]==1):
                            l=l+1
                         
                        if(l>2):   
                            newtag = 1  

                elif(i==column-1 and j==0):
                    if(FLAGS.direction8):
                        #8-neighborhood
                        if(tagList[(i-1)*row]==1):
                           l=l+1
                        if(tagList[i*row+(j+1)]==1):
                           l=l+1
                        if(tagList[(i-1)*row+(j+1)]==1):
                           l=l+1 
                        
                        if(l>1):
                            newtag = 1
                    else:
                        #4-neighborhood
                        if(tagList[(i-1)*row]==1):
                            l=l+1
                        if(tagList[(i-2)*row]==1):
                            l=l+1
                        if(tagList[i*row+(j+1)]==1):
                            l=l+1
                        if(tagList[i*row+(j+2)]==1):
                            l=l+1
                            
                        if(l>2):
                            newtag = 1  

                elif(i==column-1 and j==row-1):
                    if(FLAGS.direction8):
                        #8-neighborhood
                        if(tagList[(i-1)*row]==1):
                            l=l+1
                        if(tagList[i*row+(j-1)]==1):
                            l=l+1
                        if(tagList[(i-1)*row+(j-1)]==1):
                            l=l+1
                         
                        if(l>1):   
                            newtag = 1
                    else:
                        #4-neighborhood
                        if(tagList[(i-1)*row]==1):
                            l=l+1
                        if(tagList[(i-2)*row]==1):
                            l=l+1
                        if(tagList[i*row+(j-1)]==1):
                            l=l+1
                        if(tagList[i*row+(j-2)]==1):
                            l=l+1
                         
                        if(l>2):   
                            newtag = 1

                #4 borders
                elif(i==0 and j>0 and j<row-1):
                    #Top
                    if(FLAGS.direction8):
                        #8-neighborhood
                        if(tagList[i*row+j-1]==1):
                            l=l+1
                        if(tagList[i*row+(j+1)]==1):
                            l=l+1
                        if(tagList[(i+1)*row+j]==1):
                            l=l+1
                        if(tagList[(i+1)*row+(j-1)]==1):
                            l=l+1
                        if(tagList[(i+1)*row+(j+1)]==1):
                            l=l+1
                        
                        if(l>2):    
                            newtag = 1
                    else:
                        #4-neighborhood
                        if(tagList[i*row+j-1]==1):
                            l=l+1
                        if(tagList[i*row+(j+1)]==1):
                            l=l+1
                        if(tagList[i*row+(j+2)]==1):
                            l=l+1
                        if(tagList[(i+1)*row+j]==1):
                            l=l+1
                        if(tagList[(i+2)*row+j]==1):
                            l=l+1
                        
                        if(l>3):    
                            newtag = 1
                            
                elif(i>0 and i<column-1 and j==0):
                    #Left
                    if(FLAGS.direction8):
                        #8-neighborhood
                        if(tagList[(i-1)*row+j]==1):
                            l=l+1
                        if(tagList[(i+1)*row+j]==1):
                            l=l+1
                        if(tagList[i*row+(j+1)]==1):
                            l=l+1
                        if(tagList[(i-1)*row+(j+1)]==1):
                            l=l+1
                        if(tagList[(i+1)*row+(j+1)]==1):
                            l=l+1
                            
                        if(l>2):    
                            newtag = 1
                    else:
                        #4-neighborhood
                        if(tagList[(i-1)*row+j]==1):
                            l=l+1
                        if(tagList[i*row+(j+1)]==1):
                            l=l+1
                        if(tagList[i*row+(j+2)]==1):
                            l=l+1
                        if(tagList[(i+1)*row+j]==1):
                            l=l+1
                        #if(tagList[(i+2)*row+j]==1):
                        #    l=l+1
                            
                        if(l>2):
                            newtag = 1

                elif(i>0 and i<column-1 and j==row-1):
                    #Right
                    if(FLAGS.direction8):
                        #8-neighborhood
                        if(tagList[(i-1)*row+j]==1):
                            l=l+1
                        if(tagList[(i+1)*row+j]==1):
                            l=l+1
                        if(tagList[i*row+(j-1)]==1):
                            l=l+1
                        if(tagList[(i-1)*row+(j-1)]==1):
                            l=l+1
                        if(tagList[(i+1)*row+(j-1)]==1):
                            l=l+1
                         
                        if(l>2):   
                            newtag = 1
                    else:
                        #4-neighborhood
                        if(tagList[(i-1)*row+j]==1):
                            l=l+1
                        if(tagList[i*row+(j-1)]==1):
                            l=l+1
                        if(tagList[i*row+(j-2)]==1):
                            l=l+1
                        if(tagList[(i+1)*row+j]==1):
                            l=l+1
                        if(tagList[(i+2)*row+j]==1):
                            l=l+1
                        
                        if(l>3):    
                            newtag = 1

                elif(i==column-1 and j>0 and j<column-1):
                    #Bottom
                    if(FLAGS.direction8):
                        #8-neighborhood
                        if(tagList[i*row+(j-1)]==1):
                            l=l+1
                        if(tagList[i*row+(j+1)]==1):
                            l=l+1
                        if(tagList[(i-1)*row+j]==1):
                            l=l+1
                        if(tagList[(i-1)*row+(j-1)]==1):
                            l=l+1
                        if(tagList[(i-1)*row+(j+1)]==1):
                            l=l+1
                            
                        if(l>2):   
                            newtag = 1
                    else:
                        #4-neighborhood
                        if(tagList[i*row+(j-1)]==1):
                            l=l+1
                        if(tagList[(i-1)*row+j]==1):
                            l=l+1
                        if(tagList[(i-1)*row+(j-2)]==1):
                            l=l+1
                        if(tagList[i*row+(j+1)]==1):
                            l=l+1
                        if(tagList[i*row+(j+2)]==1):
                            l=l+1
                            
                        if(l>3):
                            newtag = 1

                #inside 
                else:
                    if(FLAGS.direction8):
                        #8-neighborhood
                        if(i>=1 and j>=1 and i<=column-1 and j<=row-1):
                            if(tagList[i*row+(j-1)]==1):
                               l=l+1
                            if(tagList[i*row+(j+1)]==1):
                               l=l+1
                            if(tagList[(i-1)*row+j]==1):
                               l=l+1
                            if(tagList[(i+1)*row+j]==1):
                               l=l+1
                            if(tagList[(i-1)*row+(j-1)]==1):
                               l=l+1
                            if(tagList[(i-1)*row+(j+1)]==1):
                               l=l+1
                            if(tagList[(i+1)*row+(j-1)]==1):
                               l=l+1
                            if(tagList[(i+1)*row+(j+1)]==1):
                               l=l+1
                         
                            if(l>=threshold):   
                               newtag = 1
                    else:
                        #4-neighborhood
                        if(i>=1 and j>=1 and i<=column-1 and j<=row-1):                       
                            if(tagList[i*row+(j-1)]==1):
                               l=l+1
                            if(tagList[i*row+(j+1)]==1):
                               l=l+1
                            if(tagList[(i-1)*row+j]==1):
                               l=l+1
                            if(tagList[(i+1)*row+j]==1):
                               l=l+1
                            
                            if(l>2):    
                               newtag = 1
                                   
            count = count + 1
            newtagList.append(newtag)
            
    return newtagList 

#Remove noise & connect split areas according to 4-neighborhood or 8-neighborhood
def Postprocessing(tagList):
    
    #Remove single noises
    tagList = RemoveSingleNoises(tagList)
    #tagList = RemoveSingleNoises(tagList)
        
    #Connect split areas
    tagList = MergeSplitAreas(tagList,3)
    #tagList = MergeSplitAreas(tagList,5)   
       
    return tagList

#Generate Result Images according to tag list
def GenerateResImage():

    tagList = RecognizeImgList(FLAGS.savepath, 'jpg')
    NewtagList = Postprocessing(tagList)

    #Generate result image
    msize = 2048
    toImage = Image.new('RGBA', (msize, msize))
    
    row = int(msize / FLAGS.sample_size)

    FindPath = FLAGS.savepath
    FileNames=os.listdir(FindPath)
    FlagStr='jpg'
    
    xcount = 0
    ycount = 0
    if (len(FileNames)>0):
        FileNames.sort()
        for fn in FileNames:
            if (len(FlagStr)>0):
                if (IsSubString(FlagStr,fn)):

                    if(NewtagList[ycount * row + xcount] == 1):
                        fromImage = Image.open(FindPath + fn)
                        toImage.paste(fromImage,(xcount * FLAGS.sample_size, ycount * FLAGS.sample_size))
                    else: 
                        fromImage = Image.open('blackblock.jpg')
                        toImage.paste(fromImage,(xcount * FLAGS.sample_size, ycount * FLAGS.sample_size))
 
                else:
                    if(NewtagList[ycount * row + xcount] == 1):
                        fromImage = Image.open(FindPath + fn)
                        toImage.paste(fromImage,(xcount * FLAGS.sample_size, ycount * FLAGS.sample_size))
                    else: 
                        fromImage = Image.open('blackblock.jpg')
                        toImage.paste(fromImage,(xcount * FLAGS.sample_size, ycount * FLAGS.sample_size))
                    
            xcount = xcount + 1
            if(xcount == row):
                xcount = 0
                ycount = ycount + 1

    toImage.save('result.jpg')
    
    return NewtagList

if __name__ == '__main__':
    
    if(FLAGS.blockgenerate):
        print(BlockGenerate())
    else:
        print(GenerateResImage())           

