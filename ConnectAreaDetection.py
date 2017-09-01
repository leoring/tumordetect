# -*- coding: utf-8 -*-

""" 
    Connection Area Detection
    2017.8.28
"""

from __future__ import division, print_function, absolute_import

import os
import tensorflow as tf
import matplotlib.pyplot as plt
from skimage import io
from skimage import data,color,morphology,feature

# Added by Le Ning on August 28,2017
tf.app.flags.DEFINE_string('load_path','../data/labelpng/',"directory which include marked pictures")
tf.app.flags.DEFINE_string('save_path','../data/ConnectArea/',"directory which include connected pictures")
FLAGS = tf.app.flags.FLAGS

def ConnectAreaDetect(FileName,OrginalPath = FLAGS.load_path,TargetPath = FLAGS.save_path): 
    
    #load target image
    img = io.imread(OrginalPath + FileName)

    #Detect canny edge & generate binary image
    img = color.rgb2gray(img)
    edgs = feature.canny(img, sigma = 3) 
    chull = morphology.convex_hull_object(edgs)
    
    #save results;
    img = 255 * chull

    #Generate Connect Area file name;
    JPGFile = TargetPath + FileName[:-3] + 'jpg'
    io.imsave(JPGFile,img)

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
                    ConnectAreaDetect(fn)
                    FileList.append(fn)
                  
                else:
                    ConnectAreaDetect(fn)
                    FileList.append(fn)
    if (len(FileList)>0):
        FileList.sort()
    
    return FileList

FileList = GetFileList(FLAGS.load_path,'PNG')
print(FileList)

