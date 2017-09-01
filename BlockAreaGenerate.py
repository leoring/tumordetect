# -*- coding: utf-8 -*-
			
""" 
    Generate Training & Testing Sets
    2017.8.29
"""

from __future__ import division, print_function, absolute_import

import os
import tensorflow as tf
from PIL import Image 

#define paramaters
tf.app.flags.DEFINE_string('load_path','../data/cancer/',"directory which include original cancer or no-cancer pictures")
tf.app.flags.DEFINE_string('cc_path','../data/ConnectArea/',"directory which include connected pictures")
tf.app.flags.DEFINE_string('save_path','../data/presamples/',"directory which save results")
tf.app.flags.DEFINE_string('postive','../data/presamples/postive/',"directory which save postive samples")
tf.app.flags.DEFINE_string('negtive','../data/presamples/negtive/',"directory which save negtive samples")
tf.app.flags.DEFINE_integer('block_width', 64, "block width")
tf.app.flags.DEFINE_integer('block_height', 64, "block height")
tf.app.flags.DEFINE_integer('stepX', 64, "step in horizontal direction")
tf.app.flags.DEFINE_integer('stepY', 64, "step in vertical direction")
tf.app.flags.DEFINE_float('IOU_threshold', 0.3, "step in vertical direction")
tf.app.flags.DEFINE_integer('offsetX', 0, "step in horizontal direction")
tf.app.flags.DEFINE_integer('offsetY', 0, "step in vertical direction")
tf.app.flags.DEFINE_boolean('debug_mode', False, "Debug mode")
FLAGS = tf.app.flags.FLAGS

#Determinate block type (Tumor: 1, Without Tumor:0)
def IOUType(FileName, x_top, y_top, x_bottom, y_bottom):
    IOU = 0

    ccFileName = FileName[:-4] + 'jpg'
    pil_im = Image.open(FLAGS.cc_path + ccFileName)

    count = 0
    for i in range(y_top, y_bottom):
        for j in range(x_top, x_bottom):
            if(pil_im.getpixel((j,i))>100):
                count = count + 1

    coverRatio = count * 1.0 / (FLAGS.block_width * FLAGS.block_height)
    if(coverRatio > FLAGS.IOU_threshold):
        IOU = 1
    
    if(FLAGS.debug_mode):
        print(IOU)

    return IOU

#get file number in selected directory
def filenumber(FilePath):
    count = 0
    for root, dirs, files in os.walk(FilePath):
        fileLength = len(files)
        if fileLength != 0:
            count = count + fileLength
    return count

#Split single picture to blocks with results (0 or 1)
def Split2Blocks(FileName):
    
    #get existed file number in pre-defined directory
    count = 0
    count = filenumber(FLAGS.postive) + filenumber(FLAGS.negtive)
    if(FLAGS.debug_mode == True):
        print ("The number of existed files is: ", count)
        
    #split orginal image to blocks 
    pil_im = Image.open(FLAGS.load_path + FileName)
    xRange = int(pil_im.width / FLAGS.stepX) 
    yRange = int(pil_im.height / FLAGS.stepY)

    x_top = 0 + FLAGS.offsetX
    y_top = 0 + FLAGS.offsetY
    x_bottom = FLAGS.block_width + FLAGS.offsetX
    y_bottom = FLAGS.block_height + FLAGS.offsetY

    all_the_text = ''
    for i in range(0, yRange):
        for j in range(0, xRange):

            #check IOU type;
            IOU = IOUType(FileName, x_top, y_top, x_bottom, y_bottom)

            #generate & save image blocks;
            box = (x_top, y_top, x_bottom, y_bottom)
            region = pil_im.crop(box)
            
            temfile1='%d'%count
            temfile2='%d'%IOU
            
            if(IOU == 1):
                #postive samples
                Targetfile = FileName[:-4] + temfile1 +'.jpg' 
                region.save(FLAGS.postive + Targetfile)
            
                all_the_text = all_the_text + FLAGS.postive + Targetfile + ' ' + temfile2 + '\n'
            else: 
                #negtive samples
                Targetfile = FileName[:-4] + temfile1 +'.jpg' 
                region.save(FLAGS.negtive + Targetfile)
            
                all_the_text = all_the_text + FLAGS.negtive + Targetfile + ' ' + temfile2 + '\n'

            x_top = x_top + FLAGS.stepX
            x_bottom = x_bottom + FLAGS.stepX
            count = count + 1
            
            if(x_bottom > pil_im.width):
                break

        x_top = 0
        x_bottom = FLAGS.block_width
        y_top = y_top + FLAGS.stepY
        y_bottom = y_bottom + FLAGS.stepY
        
        if(y_bottom > pil_im.height):
            break
    
    #write block filename & IOU type to txt files;
    file_object = open(FLAGS.save_path + 'filelist.txt', 'w+')
    file_object.write(all_the_text)
    file_object.close( )

    if(FLAGS.debug_mode):
        print(count)
   
def IsSubString(SubStrList,Str):
    
    flag=True
    for substr in SubStrList:
        if not(substr in Str):
            flag=False
    return flag

def GetFileList(FindPath,FlagStr=[]):
    
    FileList=[]
    FileNames=os.listdir(FindPath)
    if (len(FileNames)>0):
        for fn in FileNames:
            if (len(FlagStr)>0):
                if (IsSubString(FlagStr,fn)):
                    Split2Blocks(fn)
                    FileList.append(fn)
                else:
                    Split2Blocks(fn)
                    FileList.append(fn)
    if (len(FileList)>0):
        FileList.sort()
    
    return FileList

if (FLAGS.debug_mode):
    Split2Blocks('2017-06-09_18.08.16.ndpi.16.17875_16008.2048x2048.tiff')    
else:
    FileList = GetFileList(FLAGS.load_path,'TIFF')    
