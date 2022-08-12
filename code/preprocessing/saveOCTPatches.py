#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 14 13:37:36 2022

@author: matthew
"""

import sys
import os
if not '..' in sys.path:
    sys.path.append('..')
import cv2
import numpy as np
import glob
from skimage.measure import shannon_entropy
from matplotlib import pyplot as plt
from utils import utilsOOP as utils
from tqdm import trange

#Location of the raw data avi files
#root = '/home/matthew/Documents/datasets/OCT_Denoise/Data/'
root = 'D:\\datasets\\OCT Denoise\\Data\\'
frameSz = 64
entropyCutoff = 7
allEntropies = []
#Make directories for noisy images and full frames (no clean data)
os.makedirs(os.path.join(root,'patches%03d'%frameSz,'prev'),exist_ok=True)
os.makedirs(os.path.join(root,'patches%03d'%frameSz,'current'),exist_ok=True)
os.makedirs(os.path.join(root,'patches%03d'%frameSz,'next'),exist_ok=True)
os.makedirs(os.path.join(root,'fullFrames'),exist_ok=True)
#List all the AVI files (note patient 10.avi is corrupted and has no frames)
fList = sorted(glob.glob(os.path.join(root,'*.avi')))
print(len(fList))
#Iterate through each AVI file
for i in trange(len(fList)):
    fname = fList[i]
    #Make video reading object
    Vid = cv2.VideoCapture(fname)
    frameCount = int(Vid.get(cv2.CAP_PROP_FRAME_COUNT))
    frameWidth = int(Vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    frameHeight = int(Vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    #Make Directories for this patient
    prevStackSaveDir = os.path.join(root,'patches%03d'%frameSz,'prev','%02d'%i)
    currStackSaveDir = os.path.join(root,'patches%03d'%frameSz,'current','%02d'%i)
    nextStackSaveDir = os.path.join(root,'patches%03d'%frameSz,'next','%02d'%i)
    fullFrameSaveDir = os.path.join(root,'fullFrames','%02d'%i)
    os.makedirs(currStackSaveDir,exist_ok=True)
    os.makedirs(nextStackSaveDir,exist_ok=True)
    os.makedirs(prevStackSaveDir,exist_ok=True)
    os.makedirs(fullFrameSaveDir,exist_ok=True)
    
    
    fc = 0
    ret = True
    currImStack=0
    prevImStack=0
    #Read the video frame by frame (skip the first and last 10 because they are entirely background)
    while (fc < frameCount-10  and ret):
        #Read the frame
        ret, temp = Vid.read()
        #Turn into patches
        nextImStack,_,_ = utils.decimateImage(temp[:,:,0],frameSz)
        if fc < 10:
            fc+=1
            prevImStack = currImStack
            currImStack = nextImStack
            continue
        #Write the full frame
        with open(os.path.join(fullFrameSaveDir,'frame%03d.npy'%fc),'wb') as full:
            np.save(full,temp[:,:,0]) #It's a black and white image so only take 1 channel
        #Look at each patch
        for j in range(len(currImStack)):
            currPatch = currImStack[j,:,:]
            #Calculate entropy
            thisEnt = shannon_entropy(currPatch)
            allEntropies.append(thisEnt)
            #If the entropy is high enough save it and the prev. and the next frame version of the patch
            if thisEnt >= entropyCutoff:
                prevPatch = prevImStack[j,:,:]
                nextPatch = nextImStack[j,:,:]
                imName = 'patch%02d_%03d_%03d.npy'%(i,fc,j)
                currFrameName = os.path.join(currStackSaveDir,imName)
                nextFrameName = os.path.join(nextStackSaveDir,imName)
                prevFrameName = os.path.join(prevStackSaveDir,imName)
                with open(prevFrameName,'wb') as f:
                    np.save(f,prevPatch)
                with open(currFrameName,'wb') as f2:
                    np.save(f2,currPatch)
                with open(nextFrameName,'wb') as f2:
                    np.save(f2,nextPatch)
        fc += 1
        prevImStack = currImStack
        currImStack = nextImStack
    
    Vid.release()
#Look at and save the entropy histogram
plt.hist(allEntropies)
plt.savefig('entropiesOCT_co%.1f.png'%entropyCutoff)
