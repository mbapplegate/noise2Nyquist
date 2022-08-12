# -*- coding: utf-8 -*-
"""
Created on Mon Jun 20 15:33:27 2022

@author: Matthew
"""

#Read and patchify DICOM files from the Mayo Clinic 2016 Low Dose CT grand challenge
import sys
import os
if not '..' in sys.path:
    sys.path.append('..')
import pydicom
import numpy as np
from skimage.measure import shannon_entropy
import glob
from utils import utilsOOP as utils
from matplotlib import pyplot as plt
import tqdm

if __name__ == '__main__':
    #Location of the clean and noisy images
    #FDRoot = '/home/matthew/Documents/datasets/lowDoseCT/Training_Image_Data/1mm B30/FD_1mm/full_1mm'
    #QDRoot = '/home/matthew/Documents/datasets/lowDoseCT/Training_Image_Data/1mm B30/QD_1mm/quarter_1mm'
    FDRoot = 'D:\\datasets\lowDoseCT\\Training_Image_Data\\1mm B30\\FD_1mm\\full_1mm'
    QDRoot = 'D:\\datasets\lowDoseCT\\Training_Image_data\\1mm B30\\QD_1mm\\quarter_1mm'
    #Patch size and entropy cutoff
    patchSz = 64
    entropyCutoff = 9
    allEnts = []
    
    #Where to save the patichified data
    saveDir = 'D:\\datasets\\lowDoseCT\\patches%03d'%patchSz
    fullSaveDir = 'D:\\datasets\\lowDoseCT\\fullFrames'
    #saveDir = '/home/matthew/Documents/datasets/lowDoseCT/patches%03d'%patchSz
    #fullSaveDir = '/home/matthew/Documents/datasets/lowDoseCT/fullFrames'
    
    #Make all directories
    fullCleanDir = os.path.join(fullSaveDir,'clean')
    fullNoisyDir = os.path.join(fullSaveDir,'noisy')
    cleanDir = os.path.join(saveDir,'clean')
    currentDir = os.path.join(saveDir,'current')
    prevDir = os.path.join(saveDir,'prev')
    nextDir = os.path.join(saveDir,'next')
    os.makedirs(fullCleanDir,exist_ok=True)
    os.makedirs(fullNoisyDir,exist_ok=True)
    os.makedirs(saveDir,exist_ok=True)
    os.makedirs(prevDir,exist_ok=True)
    os.makedirs(currentDir,exist_ok=True)
    os.makedirs(nextDir,exist_ok=True)
    os.makedirs(cleanDir,exist_ok=True)
    
    #There is one directory per patient
    patientList = [f.name for f in os.scandir(FDRoot) if f.is_dir()]
    totalPatches = 0
    #Iterate through each patient
    for patient in tqdm.tqdm(patientList):
        #Make clean, noisy, next, and prev directories for each patient
        thisCleanDir = os.path.join(cleanDir,patient)
        thisFullCleanDir = os.path.join(fullCleanDir,patient)
        thisFullNoisyDir = os.path.join(fullNoisyDir,patient)
        thisPrevDir = os.path.join(prevDir,patient)
        thisCurrDir = os.path.join(currentDir,patient)
        thisNextDir = os.path.join(nextDir,patient)
        os.makedirs(thisFullCleanDir,exist_ok=True)
        os.makedirs(thisFullNoisyDir,exist_ok=True)
        os.makedirs(thisCleanDir,exist_ok=True)
        os.makedirs(thisPrevDir,exist_ok=True)
        os.makedirs(thisCurrDir,exist_ok=True)
        os.makedirs(thisNextDir,exist_ok=True)
        
        #The data have a few subdirectories
        thisFDDir = os.path.join(FDRoot,patient,'full_1mm')
        thisQDDir = os.path.join(QDRoot,patient,'quarter_1mm')
        #Get all the raw DICOM images
        FDImList = sorted(glob.glob(os.path.join(thisFDDir,'*.IMA')))
        QDImList = sorted(glob.glob(os.path.join(thisQDDir,'*.IMA')))
        frameNum = 0
        assert len(FDImList)==len(QDImList)
        #Iterate through all files
        for fdFile,qdFile in zip(FDImList,QDImList):
            #Read teh DICOM files
            FD_ds= pydicom.dcmread(fdFile)
            FDIm = FD_ds.pixel_array
            QD_ds = pydicom.dcmread(qdFile)
            QDIm = QD_ds.pixel_array
            #Decimate the image into patches
            nextPatchesClean,_,_ = utils.decimateImage(FDIm,patchSz)
            nextPatchesNoisy,_,_ = utils.decimateImage(QDIm,patchSz)
            patchIdx = 0
            #Skip the first two frames because we need to have room for the previous and next versions of the images
            if frameNum == 0:
                frameNum += 1
                currPatchesClean = nextPatchesClean
                currPatchesNoisy = nextPatchesNoisy
                continue
            elif frameNum == 1:
                frameNum += 1
                prevPatchesClean = currPatchesClean
                prevPatchesNoisy = currPatchesNoisy
                currPatchesClean = nextPatchesClean
                currPatchesNoisy = nextPatchesNoisy
                continue
            #If we're on at least the third frame save the full clean and noisy versions as numpy arrays
            with open(os.path.join(thisFullCleanDir,'frame%03d.npy'%frameNum),'wb') as fullClean:
                np.save(fullClean,FDIm)
            with open(os.path.join(thisFullNoisyDir,'frame%03d.npy'%frameNum),'wb') as fullNoisy:
                np.save(fullNoisy,QDIm)
            #Now go through each patch    
            for p in range(currPatchesClean.shape[0]):
                thisCleanPatch = currPatchesClean[p,:,:]
                #Calculate entropy
                thisEnt= shannon_entropy(thisCleanPatch)
                allEnts.append(thisEnt)
               # If the patch is interesting enough save it and the previous, and the next patches
                if thisEnt > entropyCutoff:
                    prevPatch = prevPatchesNoisy[p,:,:]
                    currPatch = currPatchesNoisy[p,:,:]
                    nextPatch = nextPatchesNoisy[p,:,:]
                    imName = 'patch%03d_%04d.npy'%(frameNum,patchIdx)
                    with open(os.path.join(thisCleanDir,imName),'wb') as f:
                        np.save(f,thisCleanPatch)
                    with open(os.path.join(thisPrevDir,imName),'wb') as f2:
                        np.save(f2,prevPatch)
                    with open(os.path.join(thisCurrDir,imName),'wb') as f3:
                        np.save(f3,currPatch)
                    with open(os.path.join(thisNextDir,imName),'wb') as f4:
                        np.save(f4,nextPatch)
                    
                    patchIdx+=1
                    totalPatches += 1
            frameNum += 1
            prevPatchesClean = currPatchesClean
            prevPatchesNoisy = currPatchesNoisy
            currPatchesClean = nextPatchesClean
            currPatchesNoisy = nextPatchesNoisy
        #Endfor this patient
    #End for all patients
    #It's nice to look at the histogram of all the entropies
    print('Total Number of patches saved: %d'%totalPatches)
    plt.hist(allEnts)
                    
                
                
            
                
            
