# -*- coding: utf-8 -*-
"""
Created on Tue Aug  2 17:33:47 2022

@author: Matthew
"""

import bm4d
from matplotlib import pyplot as plt
import os
import numpy as np
from PIL import Image

#I want to include a bm4d processed frame for the confocal data
#This is a super-simple script to do that
#Give it the data directory the scan number and the frames to process
#and it will save a bm4d filtered image
if __name__ == '__main__':
    
    #Raw data location
    #rawDataDir = 'D:\\datasets\\Denoising_Planaria\\fullFrames\\condition2'
    rawDataDir = '/home/matthew/Documents/datasets/Denoising_Planaria/fullFrames/condition2'
    #Scan number
    scanNum = 3 
    #Three adjacent frames
    im1 = np.load(os.path.join(rawDataDir,'%02d'%scanNum,'frame058.npy'))
    im2 = np.load(os.path.join(rawDataDir,'%02d'%scanNum,'frame059.npy'))
    im3 = np.load(os.path.join(rawDataDir,'%02d'%scanNum,'frame060.npy'))
    
    #Make into a stack (this is cropped for speed)
    imStack = np.stack((im1[0:512,512:1024],im2[0:512,512:1024],im3[0:512,512:1024]),axis=2)
    #Run the algorithm
    filtStack = bm4d.bm4d(imStack,0.2)
    #Convert to 8-bit and save
    filtIm8bit= np.clip(np.round(filtStack[:,:,1]*255),0,255).astype('uint8') 
    filtIm = Image.fromarray(filtIm8bit)
    filtIm.save('../../communications/paper/figures/confocal/confocalbm4d.png')