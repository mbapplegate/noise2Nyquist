# -*- coding: utf-8 -*-
"""
Created on Tue Aug  2 17:33:47 2022

@author: Matthew
"""

import bm4d
from matplotlib import pyplot as plt
import os
import numpy as np
from PIL import Image, ImageFilter

#I want to include a bm4d, Median, and OOF average processed frame for the OCT data
#This is a super-simple script to do that
#Give it the data directory the scan number and the frames to process
#and it will save a bm4d filtered image
if __name__ == '__main__':
    #Raw data location
    rawDataDir = 'D:\\datasets\\OCT Denoise\\Data\\fullFrames'
    #rawDataDir = '/home/matthew/Documents/datasets/Denoising_Planaria/fullFrames/condition2'
    #Patient number
    scanNum = 15
    #Adjacent frames (for bm4d)
    im1 = np.load(os.path.join(rawDataDir,'%02d'%scanNum,'frame394.npy'))
    im2 = np.load(os.path.join(rawDataDir,'%02d'%scanNum,'frame395.npy'))
    im3 = np.load(os.path.join(rawDataDir,'%02d'%scanNum,'frame396.npy'))
    #Stack
    imStack = np.stack((im1,im2,im3),axis=2)
    #Filter and convert
    filtStack = bm4d.bm4d(imStack,25)
    filtIm8bit= np.clip(np.round(filtStack[:,:,1]),0,255).astype('uint8') 
    filtIm = Image.fromarray(filtIm8bit)
    #Also median filter
    im2PIL = Image.fromarray(im2)
    thisFiltImage = im2PIL.filter(ImageFilter.MedianFilter(3))
    #Also average the frames
    imAvg = np.mean(imStack,axis=2)
    imAvg8bit = np.clip(np.round(imAvg),0,255).astype('uint8')
    imAvgPIL = Image.fromarray(imAvg8bit)
    #Save them all
    filtIm.save('../../communications/paper/figures/oct/bm4dFiltFrame.png')
    thisFiltImage.save('../../communications/paper/figures/oct/medianFiltFrame.png')
    imAvgPIL.save('../../communications/paper/figures/oct/oofAvgFiltFrame.png')