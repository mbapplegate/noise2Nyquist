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
#I want to include a bm4d processed frame for the CT data
#This is a super-simple script to do that
#Give it the data directory the patient and the frames to process
#and it will save a bm4d filtered image
if __name__ == '__main__':
    #Location of raw data
    rawDataDir = '/home/matthew/Documents/datasets/lowDoseCT/fullFrames/noisy'
    #Patient
    patient = 'L192'
    #Three adjacent frames
    im1 = np.load(os.path.join(rawDataDir,'%s'%patient,'frame188.npy'))
    im2 = np.load(os.path.join(rawDataDir,'%s'%patient,'frame189.npy'))
    im3 = np.load(os.path.join(rawDataDir,'%s'%patient,'frame190.npy'))
    #In a stack
    imStack = np.stack((im1,im2,im3),axis=2)
    #Normalize the stack
    p = np.percentile(imStack,[0.1,99.9])
    imNorm = (imStack-p[0])/(p[1]-p[0])
    #Filter the stack
    filtStack = bm4d.bm4d(imNorm,0.05)
    #Convert to 8-bit and save
    filtIm8bit = np.clip(np.round(filtStack[:,:,1]*255),0,255).astype('uint8')
    filtIm = Image.fromarray(filtIm8bit)
    
    filtIm.save('../../communications/paper/figures/ct/ctImbm4d.png')
    
    
    
    
    