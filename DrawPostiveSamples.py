# -*- coding: utf-8 -*-
			
""" 
    Randomly Draw Training & Testing Samples
    2017.8.30
"""

from __future__ import division, print_function, absolute_import

import os
import tensorflow as tf
import random
import sys, shutil

#define paramaters
tf.app.flags.DEFINE_string('pre_postive','../data/presamples/postive/',"directory which save original postive samples")
tf.app.flags.DEFINE_string('post_postive','../data/postsamples/postive/',"directory which save selected postive samples")
tf.app.flags.DEFINE_string('save_path','../data/postsamples/',"directory which save results")
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

def CopyPostiveSamples(FlagStr=[]):
    
    FileList=[]
    
    FileNames=os.listdir(FLAGS.pre_postive)

    all_the_text = ''
    if (len(FileNames)>0):
        for fn in FileNames:
            if (len(FlagStr)>0):
                if (IsSubString(FlagStr,fn)):
                    
                    shutil.copy(FLAGS.pre_postive + fn, FLAGS.post_postive + fn)
                    all_the_text = all_the_text + FLAGS.post_postive + fn + ' 1' + '\n'

                    FileList.append(fn)
        
                else:
                    shutil.copy(FLAGS.pre_postive + fn, FLAGS.post_postive + fn)
                    all_the_text = all_the_text + FLAGS.post_postive + fn + ' 1' + '\n'

                    FileList.append(fn)
     
    if (len(FileList)>0):
        FileList.sort()
    
    #write filename & IOU type;
    file_object = open(FLAGS.save_path + 'filelist_Pos.txt', 'w+')
    file_object.write(all_the_text)
    file_object.close( )

    return FileList

#copy all postive samples to post sample sets
CopyPostiveSamples('jpg')    
