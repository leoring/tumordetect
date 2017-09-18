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
def CopyNegtiveSamples(FlagStr=[], Ratio = 1.0):
    
    FileList=[]
    
    RND = random.randint(0, int(Ratio))

    count = 0     
    FileNames=os.listdir(FLAGS.pre_negtive)

    all_the_text = ''
    if (len(FileNames)>0):
        for fn in FileNames:
            if (len(FlagStr)>0):
                if (IsSubString(FlagStr,fn)):
                    if(Ratio > 0.99 and Ratio < 1.5):
                        shutil.copy(FLAGS.pre_negtive + fn, FLAGS.post_negtive + fn)
                        all_the_text = all_the_text + FLAGS.post_negtive + fn + ' 0' + '\n'

                    elif(count % int(Ratio) == RND):
                        shutil.copy(FLAGS.pre_negtive + fn, FLAGS.post_negtive + fn)
                        all_the_text = all_the_text + FLAGS.post_negtive + fn + ' 0' + '\n'

                    FileList.append(fn)
        
                else:
                    if(Ratio > 0.99 and Ratio < 1.5):
                        shutil.copy(FLAGS.pre_negtive + fn, FLAGS.post_negtive + fn)
                        all_the_text = all_the_text + FLAGS.post_negtive + fn + ' 0' + '\n'

                    elif(count % int(Ratio) == RND):
                        shutil.copy(FLAGS.pre_negtive + fn,FLAGS.post_negtive + fn)
                        all_the_text = all_the_text + FLAGS.post_negtive + fn + ' 0' + '\n'

                    FileList.append(fn)
                    
                count = count + 1
    
    if (len(FileList)>0):
        FileList.sort()
    
    return all_the_text

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
    
    return all_the_text

PostiveNum = filenumber(FLAGS.pre_postive)
NegtiveNum = filenumber(FLAGS.pre_negtive)
print('NegtiveNum:',NegtiveNum)
print('PostiveNum:',PostiveNum)

#copy all postive samples to post sample sets
all_the_text = CopyPostiveSamples('jpg')

#draw negtive samples according to ratio between negtive & postive samples
ratio = 1.0
#if (PostiveNum > 0):
#    ratio = 1.0 * NegtiveNum / PostiveNum

all_the_text = all_the_text + CopyNegtiveSamples('jpg',ratio)

#write filename & IOU type;
file_object = open(FLAGS.save_path + 'filelist.txt', 'w+')
file_object.write(all_the_text)
file_object.close( )
    
