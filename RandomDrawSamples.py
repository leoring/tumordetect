# -*- coding: utf-8 -*-
			
""" 
    Randomly Draw Training & Testing Samples
    2017.8.30
"""

from __future__ import division, print_function, absolute_import

import os
import tensorflow as tf
import random
import sys, shutil

#define paramaters
tf.app.flags.DEFINE_string('pre_postive','../data/presamples/postive/',"directory which save original postive samples")
tf.app.flags.DEFINE_string('pre_negtive','../data/presamples/negtive/',"directory which save original negtive samples")
tf.app.flags.DEFINE_string('post_postive','../data/postsamples/postive/',"directory which save selected postive samples")
tf.app.flags.DEFINE_string('post_negtive','../data/postsamples/negtive/',"directory which save selected negtive samples")
tf.app.flags.DEFINE_string('save_path','../data/postsamples/',"directory which save results")
tf.app.flags.DEFINE_boolean('debug_mode', True, "Debug mode")
FLAGS = tf.app.flags.FLAGS

#get file number in selected directory
def filenumber(FilePath):
    count = 0
    for root, dirs, files in os.walk(FilePath):
        fileLength = len(files)
        if fileLength != 0:
            count = count + fileLength
    return count

def IsSubString(SubStrList,Str):
    
    flag=True
    for substr in SubStrList:
        if not(substr in Str):
            flag=False
    return flag

#get the file name in selected directory
def CopyFileList(SrcPath, TarPath, FlagStr=[], Ratio = 1.0):
    
    FileList=[]
    
    r
    FileNames=os.listdir(SrcPath)
    if (len(FileNames)>0):
        for fn in FileNames:
            if (len(FlagStr)>0):
                if (IsSubString(FlagStr,fn)):
                    if(Ratio > 0.99 and Ratio < 1.5):
                        shutil.copy(ScrPath + FileName,TarPath + FileName)
                    else if():
                        shutil.copy(ScrPath + FileName,TarPath + FileName)
                    FileList.append(fn)
        
                else:
                    if(Ratio > 0.99 and Ratio < 1.5):
                        shutil.copy(ScrPath + FileName,TarPath + FileName) 
                    else if():
                        shutil.copy(ScrPath + FileName,TarPath + FileName)

                    FileList.append(fn)
    if (len(FileList)>0):
        FileList.sort()
    
    return FileList

PostiveNum = filenumber(FLAGS.pre_postive)
NegtiveNum = filenumber(FLAGS.pre_negtive)

#copy all postive samples to post sample sets
CopyFileList(FLAGS.pre_postive, FLAGS.post_postive, 'jpg')

#draw negtive samples according to ratio between negtive & postive samples
if (PostiveNum > 0):
    ratio = 1.0 * NegtiveNum / PostiveNum
    CopyFileList(FLAGS.pre_negtive, FLAGS.post_negtive, 'jpg', ratio)
    
