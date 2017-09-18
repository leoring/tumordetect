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
tf.app.flags.DEFINE_string('targetfile','2017-06-10_19.46.43.ndpi.16.43169_13551.2048x2048.tiff',"Target picture")
tf.app.flags.DEFINE_string('savepath','../data/blocks/',"Target picture")
tf.app.flags.DEFINE_string('modelfile','model_resnet_tumor-208800',"model file")
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

def create_residualnet(num_classes):

    # Building 'residual network'
    n = 5
    net = tflearn.input_data(shape=[None, FLAGS.sample_size, FLAGS.sample_size, 3])
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

    return net

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
    
    #load model
    network = create_residualnet(2)
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
                    if(pred[0][1]>pred[0][0] and pred[0][1]>0.1):
                        tag = 1 

                    tagList.append(tag)
                else:
                    X = DirectReadImg(FindPath + fn)
                    pred = model.predict([X[0]])

                    tag = 0
                    if(pred[0][1]>pred[0][0] and pred[0][1]>0.1):
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
                        if(tagList[i*row+(j-1)]==0 and tagList[i*row+(j+1)]==0 
                           and tagList[(i-1)*row+j]==0 and tagList[(i+1)*row+j]==0 
                           and tagList[(i-1)*row+(j-1)]==0 and tagList[(i-1)*row+(j+1)]==0
                           and tagList[(i+1)*row+(j-1)]==0 and tagList[(i+1)*row+(j+1)]==0):
                            newtag = 0
                    else:
                        #4-neighborhood
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
    #tagList = RemoveSingleNoises(tagList)

    #Connect split areas
    #tagList = MergeSplitAreas(tagList,3)
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

