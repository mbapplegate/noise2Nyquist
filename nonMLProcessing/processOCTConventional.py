
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  7 16:38:59 2022

@author: matthew
"""
import sys
import os
if not '..' in sys.path:
    sys.path.append('..')
import numpy as np
from PIL import Image, ImageFilter
from scipy.ndimage import gaussian_filter
import glob
import bm3d
import bm4d
from utils import utilsOOP as utils

import tqdm

#Gets the center N files from each patient
#Inputs:
    #fList:  List of files to process
    #nFiles: Number of files to return per patient
#Output:
    #outFrame: A list of lists, one for each patient containing nFiles filenames
###############################################################################3
def getCenterNFiles(fList,nFiles):
    frames = []
    scans = [] 
    outFrame = []
    #Look through each file in the list
    for i in range(len(fList)):
        thisPath = fList[i]
        thisFName = os.path.splitext(os.path.basename(thisPath))[0]
        #Get the frame number based on the file name
        thisFNum = int(thisFName[-3:])
        #Get the patient number based on the filename
        thisScanNum = os.path.dirname(thisPath)[-2::]
        frames.append(thisFNum) #Make a big list of all the frames
        scans.append(thisScanNum) #Make a big list of all the scans
    
    scans = np.array(scans)
    frames = np.array(frames)
    #The number of unique patients in this list of files
    numScans = np.unique(scans)
    imsPerScan = []
    firstFrame = []
    
    #For each patient
    for j in range(len(numScans)):
        #Get a list of how many images are in this patient's scan
        imsPerScan.append(np.sum(scans == numScans[j]))
        #Frame numbers for this patient
        f = frames[scans==numScans[j]]
        #First frame from this patient
        firstFrame.append(np.min(f))
    #For each patient      
    for k in range(len(imsPerScan)):
        scanFrames = []
        #Calculate the starting frame
        startFrame = int(firstFrame[k]+(imsPerScan[k]/2 - nFiles/2))
        #Iterate through all of the frames you want to save
        for im in range(startFrame,startFrame+nFiles):
            #Find the particular patient and frame in the dataset
            datOneHot = np.array([x&y for (x,y) in zip(scans==numScans[k],frames==im)])
            assert np.sum(datOneHot) == 1 #There needs to only be one
            #Get the index
            idx = np.where(datOneHot == True)[0][0]
            #Return the file name
            scanFrames.append(fList[idx])
        outFrame.append(scanFrames)
    return outFrame

#Loads data as well as previous and next frame (for oof Averaging and bm4d)
#Inputs:
    #imPath:      Path to the image to load
    #dataDir:     Directory where the data live
    #imsToReturn: How many images to return (either 1 or 3)
#Outputs:
    #noisyIm: The original frame
    #prevIm:  The previous frame
    #nextIm:  The next frame
#####################################################
def loadData(imPath,dataDir,imsToReturn=4):
    fname = os.path.basename(imPath)
    fnameNoExt = os.path.splitext(fname)[0]
    fNum = int(fnameNoExt[-3::])
    prevFName = 'frame%03d.npy'%(fNum-1)
    nextFName = 'frame%03d.npy'%(fNum+1)
    
    imPathParts = imPath.split(os.path.sep)
    noisyIm = np.load(os.path.join(dataDir,imPathParts[-2],fname))
    if imsToReturn == 1:
        return noisyIm
    else:
        prevIm = np.load(os.path.join(dataDir,imPathParts[-2],prevFName))
        nextIm = np.load(os.path.join(dataDir,imPathParts[-2],nextFName))
    
        return noisyIm,prevIm,nextIm

#OCT non-learning methods
if __name__ == '__main__':
    #This is a list of the patients in each fold
    testScans=[[8,11,28,31 ],
              [2, 6,15,25 ],
              [5,14,24,27 ],
              [7,9,16,26  ],
              [19,23,29,34],
              [13,18,30,35],
              [17,22],
              [1,3,12],
              [20,32,33],
              [0,4,21]]
   
    #Raw data directory
    rawDataDir = '/scratch/m.applegate/OCTData/fullFrames'
    #Methods to test
    methods = ['none','median','gaussian','oofAvg','bm3d','bm4d']
    methodArgs = [0,3,1,3,25,25]
    #methods = ['bm4d']
    #methodArgs = [25]
    numIms=96
    #iterate through each method
    for method,methodArg in zip(methods,methodArgs):
        saveName = 'testResults_%s%s.csv'%(method,str(methodArg))
        #Iterate through each split
        for i,split in enumerate(tqdm.tqdm(testScans,desc='%8s'%method,leave=True)):
            fListofLists = []
            for j in range(len(split)):
                fList = sorted(glob.glob(os.path.join(rawDataDir,'%02d'%split[j],'*.npy')))
                fListofLists.append(fList)
            allFiles = [item for sublist in fListofLists for item in sublist]
           
            #Just the center numIms files (plus possibly some extra for stacking)
            imFiles = getCenterNFiles(allFiles,numIms)
            #List of all the patients processed in this split
            
            allData = [] #Data to write to file
            saveDir = os.path.join('..','results','oct','conventional')
            os.makedirs(saveDir,exist_ok=True)
            saveImageDir = os.path.join(saveDir,'Images_%s%s'%(method,methodArg))
            os.makedirs(saveImageDir,exist_ok=True)
            #Iterate through each volume in the split    
            for vol in range(len(split)):
                #ScanIms has all the patches in the scan that need to be processed
                scanIms = imFiles[vol]
                scanDir = os.path.join(saveImageDir,'%02d'%split[vol])
                os.makedirs(scanDir,exist_ok=True)
                #No processing -- get raw noisy Image data
                if method=='none':
                    for i in range(numIms):
                        noisyIm = loadData(scanIms[i],rawDataDir,1)
                        noisyPatches = utils.decimateMosaic(noisyIm,64,overlap=0.5)
                        noisyImage = utils.stitchMosaic(noisyPatches,32)
                        #Save the "Processed Image"
                        noisyIm = Image.fromarray(np.clip(np.round(noisyImage),0,255).astype('uint8'))
                        noisyIm.save(os.path.join(scanDir,'frame_%03d.png'%i))
                        #Calculate statist
                       
                elif method ==  'median':
                    for i in range(numIms):
                        noisyIm = loadData(scanIms[i],rawDataDir,1)
                        noisyPatches = utils.decimateMosaic(noisyIm,64,overlap=0.5)
                        filteredPatches = np.zeros(noisyPatches.shape)
                        for j in range(noisyPatches.shape[2]):
                            for k in range(noisyPatches.shape[3]):
                                thisPatch = noisyPatches[:,:,j,k]
                                thisImage = Image.fromarray(thisPatch)
                                thisFiltImage = np.array(thisImage.filter(ImageFilter.MedianFilter(methodArg)))
                                filteredPatches[:,:,j,k] = thisFiltImage
                        
                        stitchImage = utils.stitchMosaic(filteredPatches,32)
                        procIm = Image.fromarray(np.clip(np.round(stitchImage),0,255).astype('uint8'))
                        procIm.save(os.path.join(scanDir,'frame_%03d.png'%i))
                      
                elif method =='gaussian':
                    for i in range(numIms):
                        noisyIm = loadData(scanIms[i],rawDataDir,1)
                        noisyPatches = utils.decimateMosaic(noisyIm,64,overlap=0.5)
                        filteredPatches = np.zeros(noisyPatches.shape)
                        for j in range(noisyPatches.shape[2]):
                            for k in range(noisyPatches.shape[3]):
                                thisPatch = noisyPatches[:,:,j,k]
                                thisFiltImage = gaussian_filter(thisPatch,methodArg)
                                filteredPatches[:,:,j,k] = thisFiltImage
                        
                        stitchImage = utils.stitchMosaic(filteredPatches,32)
                        #Save the Processed Image
                        procIm = Image.fromarray(np.clip(np.round(stitchImage),0,255).astype('uint8'))
                        procIm.save(os.path.join(scanDir,'frame_%03d.png'%i))
                elif method == 'oofAvg':
                    bracket = int((methodArg-1)/2)
                    for i in range(numIms):
                        noisyIm,prevIm,nextIm = loadData(scanIms[i],rawDataDir,3)
                        noisyStack = np.stack((prevIm,noisyIm,nextIm),axis=2)
                        noisyPatches = utils.decimateMosaic(noisyStack,64,overlap=0.5)
                        filteredPatches = np.zeros((noisyPatches.shape[0],noisyPatches.shape[1],noisyPatches.shape[3],noisyPatches.shape[4]))
                        for j in range(noisyPatches.shape[3]):
                            for k in range(noisyPatches.shape[4]):
                                filteredPatches[:,:,j,k] = np.mean(noisyPatches[:,:,:,j,k],axis=2)
                        avgFrame = utils.stitchMosaic(filteredPatches,32)
                        procIm = Image.fromarray(np.clip(np.round(avgFrame),0,255).astype('uint8'))
                        procIm.save(os.path.join(scanDir,'frame_%03d.png'%i))
                elif method == 'bm3d':
                    for i in range(numIms):
                        noisyIm = loadData(scanIms[i],rawDataDir,1)
                        noisyPatches = utils.decimateMosaic(noisyIm,64,overlap=0.5)
                        filteredPatches = np.zeros(noisyPatches.shape)
                        for j in range(noisyPatches.shape[2]):
                            for k in range(noisyPatches.shape[3]):
                                thisPatch = noisyPatches[:,:,j,k]
                                thisFiltImage = bm3d.bm3d(thisPatch,methodArg)
                                filteredPatches[:,:,j,k] = thisFiltImage
                        stitchImage = utils.stitchMosaic(filteredPatches,32)
                      
                        #Save the Processed Image
                        procIm = Image.fromarray(np.clip(np.round(stitchImage),0,255).astype('uint8'))
                        procIm.save(os.path.join(scanDir,'frame_%03d.png'%i))
                elif method == 'bm4d':
                    #I face kind of a conundrum here. In all the ML methods I only take
                    #into account 3 frames. Should I restrict bm4d to 3 frames at a time?
                    #I know I want to split it into patches because bm4d'ing large images 
                    #will take literally forever. Also, when I patchify everything I only
                    #Save corresponding patches from 3 consecutive frames.
                    for i in range(numIms):
                        noisyIm,prevIm,nextIm = loadData(scanIms[i],rawDataDir,3)
                        noisyStack = np.stack((prevIm,noisyIm,nextIm),axis=2)
                        noisyPatches = utils.decimateMosaic(noisyStack,64,overlap=0.5)
                        filteredPatches = np.zeros((noisyPatches.shape[0],noisyPatches.shape[1],noisyPatches.shape[3],noisyPatches.shape[4]))
                        for j in range(noisyPatches.shape[3]):
                            for k in range(noisyPatches.shape[4]):
                                thisPatch = noisyPatches[:,:,:,j,k]
                                thisFiltImage = bm4d.bm4d(thisPatch,methodArg)
                                filteredPatches[:,:,j,k] = thisFiltImage[:,:,1]
                        stitchImage = utils.stitchMosaic(filteredPatches,32)
                        #Save the Processed Image
                        procIm = Image.fromarray(np.clip(np.round(stitchImage),0,255).astype('uint8'))
                        procIm.save(os.path.join(scanDir,'frame_%03d.png'%i))
                else:
                    raise ValueError('Method not recognized')
                                
            
