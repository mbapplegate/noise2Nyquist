#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 14 10:49:22 2022

@author: matthew
"""
import sys
import os
if not '..' in sys.path:
    sys.path.append('..')
import numpy as np
import glob
from matplotlib import pyplot as plt
from PIL import Image
from utils import utilsOOP as utils
from skimage.measure import shannon_entropy
import tqdm
#Access the Denoise Planaria dataset from the CARE Nature paper
#NOTE: Patches saved are pre-normalized between 0 and 1 so don't do it a second time
if __name__ == '__main__':
    #I have to use the test data because the training data doesn't have middle values of noise
    #folderName = '/home/matthew/Documents/datasets/Denoising_Planaria/test_data/'
    folderName = 'D:\\datasets\\Denoising_Planaria\\test_data'
    #Which noise condition to use. 1 is least noisy, 3 is most noisy
    conditionNum = 2
    #Define some directories to save the data in
    cleanDir = os.path.join(folderName,'GT')
    noisyDir = os.path.join(folderName,'condition_%d'%conditionNum)
    patchSz = 64 #Patch size of the saved data
    saveDirRoot = os.path.join(folderName,'..','patches%03d'%patchSz)
    saveDirFull = os.path.join(folderName,'..','fullFrames')
    os.makedirs(saveDirFull,exist_ok=True)
    os.makedirs(saveDirRoot,exist_ok=True)
    #Define some subdirectories to put various manifestations of the data
    GTDir = os.path.join(saveDirRoot,'clean')
    GTDirFull = os.path.join(saveDirFull,'clean')
    noisyDirFull = os.path.join(saveDirFull,'condition%d'%conditionNum)
    noisyDirPrev = os.path.join(saveDirRoot,'condition%d'%conditionNum,'prev')
    noisyDirCurr = os.path.join(saveDirRoot,'condition%d'%conditionNum,'current')
    noisyDirNext = os.path.join(saveDirRoot,'condition%d'%conditionNum,'next')
    os.makedirs(GTDir,exist_ok=True)
    os.makedirs(noisyDirCurr,exist_ok=True)
    os.makedirs(noisyDirNext,exist_ok=True)
    os.makedirs(noisyDirPrev,exist_ok=True)
    os.makedirs(GTDirFull,exist_ok=True)
    os.makedirs(noisyDirFull,exist_ok=True)
    
    allEnts = []    #Entropy values
    entCutoff=11  #Cutoff entropy value to be "interesting enough"
    totalSavedIms = 0
    #Read the raw clean images
    imStacks = sorted(glob.glob(os.path.join(cleanDir,'*.tif*')))
    for stackNum,im in enumerate(tqdm.tqdm(imStacks)):
        stackName = os.path.basename(im)
        #There are "fixed" and "live" samples. I don't want to deal with light-induced twitching, so only use fixed
        if 'fixed' in stackName:
            #Full stack of clean data
            cleanStack = Image.open(im)
            #Full stack of noisy data
            noisyStack = Image.open(os.path.join(noisyDir,stackName))
            #Put each stack into a folder to easily split into train/test
            thisGTDir = os.path.join(GTDir,'%02d'%stackNum)
            thisGTFullDir = os.path.join(GTDirFull,'%02d'%stackNum)
            thisNoisyFullDir = os.path.join(noisyDirFull,'%02d'%stackNum)
            thisNoisyDirPrev = os.path.join(noisyDirPrev,'%02d'%stackNum)
            thisNoisyDirCurr = os.path.join(noisyDirCurr,'%02d'%stackNum)
            thisNoisyDirNext = os.path.join(noisyDirNext,'%02d'%stackNum)
            os.makedirs(thisGTDir,exist_ok=True)
            os.makedirs(thisNoisyDirCurr,exist_ok=True)
            os.makedirs(thisNoisyDirNext,exist_ok=True)
            os.makedirs(thisNoisyDirPrev,exist_ok=True)
            os.makedirs(thisGTFullDir,exist_ok=True)
            os.makedirs(thisNoisyFullDir,exist_ok=True)
            #Go through each frame of the data
            for frameNum in range(cleanStack.n_frames):
                #Seek finds the particular image within the Dicom
                cleanStack.seek(frameNum)
                noisyStack.seek(frameNum)
                thisClean = np.array(cleanStack)
                thisNoisy = np.array(noisyStack)
                #Normalize the clean image
                cleanPercs = np.percentile(thisClean,[0.1,99.9])
                #Make sure the minimum value is high enough to avoid turning a totally black image into amplified noise
                if cleanPercs[1] < 25000:
                    cleanPercs[1] = 25000
                #Normalize the noisy image
                noisyPercs = np.percentile(thisNoisy,[2,99.7])
                cleanNorm = np.clip((thisClean-cleanPercs[0])/(cleanPercs[1]-cleanPercs[0]),0,1)
                noisyNorm = np.clip((thisNoisy-noisyPercs[0])/(noisyPercs[1]-noisyPercs[0]),0,1)
                #Skip the first and last 10 frames because they are out of focus
                if frameNum > 10 and frameNum <= cleanStack.n_frames-10:
                    with open(os.path.join(thisGTFullDir,'frame%03d.npy'%frameNum),'wb') as cleanFull:
                        np.save(cleanFull,cleanNorm)
                    with open(os.path.join(thisNoisyFullDir,'frame%03d.npy'%frameNum),'wb') as noisyFull:
                        np.save(noisyFull,noisyNorm)
                #Split the image into patches
                nextCleanPatches,_,_ = utils.decimateImage(cleanNorm,patchSz,overlap=0.5)
                nextNoisyPatches,_,_ = utils.decimateImage(noisyNorm,patchSz,overlap=0.5)
                if frameNum == 0:
                    currClean = nextCleanPatches
                    currNoisy = nextNoisyPatches
                    continue
                elif frameNum == 1:
                    prevClean = currClean
                    prevNoisy = currNoisy
                    currClean = nextCleanPatches
                    currNoisy = nextNoisyPatches
                    continue
                #Index of how many patches have been saved for this image
                tilesSaved=0
                for tileNum in range(currClean.shape[0]):
                    #"Next" tile
                    cleanTile = currClean[tileNum,:,:]
                    noisyTile = currNoisy[tileNum,:,:]
                    
                    #We've been through at least once, now calculate the entropy of the previous tile
                    cleanEntropy = shannon_entropy(cleanTile)
                    allEnts.append(cleanEntropy)
                    #If the tile is interesting enough then save it
                    if cleanEntropy >= entCutoff:
                        prevCleanTile = prevClean[tileNum,:,:]
                        prevNoisyTile = prevNoisy[tileNum,:,:]
                        nextCleanTile = nextCleanPatches[tileNum,:,:]
                        nextNoisyTile = nextNoisyPatches[tileNum,:,:]
                        #Save the patches
                        imName = 'patch%02d_%02d_%03d.npy'%(stackNum,frameNum-1,tilesSaved)
                        #Noise-free ground truth
                        with open(os.path.join(thisGTDir,imName),'wb') as f1:
                            np.save(f1,cleanTile)
                        #Current tile
                        with open(os.path.join(thisNoisyDirCurr,imName),'wb') as f2:
                            np.save(f2,noisyTile)
                        #Next tile
                        with open(os.path.join(thisNoisyDirNext,imName),'wb') as f3:
                            np.save(f3,nextNoisyTile)
                        with open(os.path.join(thisNoisyDirPrev,imName),'wb') as f4:
                            np.save(f4,prevNoisyTile)
                        tilesSaved += 1
                        totalSavedIms +=1
                prevClean = currClean
                prevNoisy = currNoisy
                currClean = nextCleanPatches
                currNoisy = nextNoisyPatches
    plt.hist(allEnts)
    print("Total Number of Images Saved: %d"%totalSavedIms)
    
