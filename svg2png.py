# -*- coding: utf-8 -*-

""" 
    SVG convert to PNG
    This function is developed based on 'inkscape' & python-based 'subprocess' library.
    2017.8.28
"""

from __future__ import division, print_function, absolute_import

import os
import subprocess
import tensorflow as tf

# Added by Le Ning on August 28,2017
tf.app.flags.DEFINE_string('svg_path','../data/labelsvg/',"directory which include svg files")
tf.app.flags.DEFINE_string('png_path','../data/labelpng/',"directory which include png files")
FLAGS = tf.app.flags.FLAGS

def IsSubString(SubStrList,Str):
    
    flag=True
    for substr in SubStrList:
        if not(substr in Str):
            flag=False
    return flag

def ConvertSvg2Png(FileName,OrginalPath = FLAGS.svg_path,TargetPath = FLAGS.png_path):
    if (FileName.find('svg')):
        #Generate PNG file name;
        PNGFile = TargetPath + FileName[:-3] + 'png'
        SVGFile = OrginalPath + FileName
        subprocess.call(["inkscape",SVGFile,"--export-png",
                         PNGFile, "--export-background","#000"])

def GetFileList(FindPath,FlagStr=[]):
    
    FileList=[]
    FileNames=os.listdir(FindPath)
    if (len(FileNames)>0):
        for fn in FileNames:
            if (len(FlagStr)>0):
                if (IsSubString(FlagStr,fn)):
                    #fullfilename=os.path.join(FindPath,fn)
                    #FileList.append(fullfilename)
                    ConvertSvg2Png(fn)
                    FileList.append(fn)
                  
                else:
                    #fullfilename=os.path.join(FindPath,fn)
                    #FileList.append(fullfilename)
                    ConvertSvg2Png(fn)
                    FileList.append(fn)
    if (len(FileList)>0):
        FileList.sort()
    
    return FileList

FileList = GetFileList(FLAGS.svg_path,'SVG')
print(FileList)

