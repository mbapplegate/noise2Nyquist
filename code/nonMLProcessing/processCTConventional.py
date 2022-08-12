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
import pandas as pd
import bm3d
import bm4d
from utils import utilsOOP as utils
from skimage.metrics import peak_signal_noise_ratio as PSNR
from skimage.metrics import structural_similarity as SSIM
import tqdm
import argparse

#Calculate MSE between target and test
def getMSE(target, tester):
    return np.sum((target-tester)**2)/np.size(target)

#Get random image files from a dataframe
#Inputs:
#   dataframe: dataframe with all of the frames from that fold
#   nFiles:    Number of files to return
#Outputs:
#   outFrame: dataframe with the file to load and the patient number
#Note: This function will only work if each fold has only 1 patient
def getRandomNFiles(dataframe,nFiles):
    frames = []
    scans = []
    outFrame = pd.DataFrame({0:[],1:[],'patient':[]})
    rng = np.random.default_rng()
    for i in range(len(dataframe)):
        thisPath = dataframe[0][i]
        thisFName = os.path.splitext(os.path.basename(thisPath))[0]
        thisFNum = int(thisFName[-3:])
        thisScanNum = os.path.dirname(thisPath)[-4::]
        frames.append(thisFNum)
        scans.append(thisScanNum)
    scans = np.array(scans)
    frames = np.array(frames)
    numScans = np.unique(scans)
    
    #This is a kludge, but it's true for this dataset (10 people, 10 folds)
    #and it lets me pick indexes from 0 to the length of the dataframe
    assert len(numScans) == 1
    imsPerScan = len(dataframe)      
    #Choose random indexes
    idxLocs = rng.choice(range(1,imsPerScan-1),size=nFiles,replace=False)
    patientFrame = dataframe.iloc[idxLocs].copy()
    #Add patient to the new dataframe
    patientFrame['patient']=numScans[0]
    #Add output dataframe to list
    outFrame=pd.concat((outFrame,patientFrame))
    
    return outFrame

#Write processed data file to refer to in makeDataTable.py
#Inputs:
#   dat:      A matrix or potentially a dataframe that contains the PSNR, SSIM, and MSE of some images
#   saveName: Name (including path) of the file to save
#Outputs:
#   None
#It will save a file at saveName
############################################
def writeData(dat,saveName):
    with open(saveName,'w') as f:
        f.write('Scan,Image_name, PSNR, SSIM, MSE\n')
        for i in range(len(dat)):
            thisRow = dat[i]
            f.write('%s,%s,%.4f,%.4f,%.6f\n'%(thisRow[0],thisRow[1],thisRow[2],thisRow[3],thisRow[4]))
    return None

#loadData loads up to 4 images and returns them
#Inputs:
#   imPaths:     Filename of the image to load 
#   dataDir:     Location where the images are stored 
#   imsToReturn: Number of images to return [1,2, or 4] always clean, sometimes noisy, sometimes prev. and next.
#Outputs:
#   normClean: clean frame
#   normNoisy: noisy frame
#   normPrev:  co-located noisy frame from the previous Z frame
#   normNext:  co-located noisy frame from the following Z frame
########################################################
def loadData(imPaths,dataDir,imsToReturn=4):
    cleanPath = imPaths[0]
    noisyPath = imPaths[1]
    fname = os.path.basename(cleanPath)
    fnameNoExt = os.path.splitext(fname)[0]
    fNum = int(fnameNoExt[-3::])
    #Previous and next frames
    prevFName = 'frame%03d.npy'%(fNum-1)
    nextFName = 'frame%03d.npy'%(fNum+1)
    
    cleanPathParts = cleanPath.split('/')
    noisyPathParts = noisyPath.split('/')
    #Build clean path
    cleanIm = np.load(os.path.join(dataDir,'clean',cleanPathParts[-2],fname))
    #Normalize clean images
    pClean = np.percentile(cleanIm,[.1,99.9])
    normClean = np.clip((cleanIm-pClean[0])/(pClean[1]-pClean[0]),0,1)
    if imsToReturn == 1:
        return normClean
    #Build noisy paths and normalize
    noisyIm = np.load(os.path.join(dataDir,'noisy',noisyPathParts[-2],fname))
    pNoisy = np.percentile(noisyIm,[.1,99.9])
    normNoisy = np.clip((noisyIm - pNoisy[0])/(pNoisy[1]-pNoisy[0]),0,1)
    if imsToReturn == 2:
        return normClean, normNoisy
    #Build previous and next paths and normalize
    prevIm = np.load(os.path.join(dataDir,'noisy',noisyPathParts[-2],prevFName))
    pPrev = np.percentile(prevIm,[0.1,99.9])
    normPrev = np.clip((prevIm-pPrev[0])/(pPrev[1]-pPrev[0]),0,1)
    
    nextIm = np.load(os.path.join(dataDir,'noisy' ,noisyPathParts[-2],nextFName))
    pNext = np.percentile(nextIm,[0.1,99.9])
    normNext = np.clip((nextIm-pNext[0])/(pNext[1]-pNext[0]),0,1)
    return normClean,normNoisy,normPrev,normNext

#CT non-learning methods
if __name__ == '__main__':
    ###############
    #
    #This is a version that was run on a computational cluster
    #The command line arguments make it easy to split into different jobs
    #You'll have to uncomment the "methods" and "methodArgs" lines below
    #if you want to run the non BM#D methods which are fast
    ####################################################
    parser = argparse.ArgumentParser()
    parser.add_argument("--method",help="Which method to process",default='bm3d')
    parser.add_argument("--splitNum",help="Which split to process",type=int,default=0)
    args = parser.parse_args()
    #rawDataDir = 'D:\\datasets\\lowDoseCT\\fullFrames'
    rawDataDir = '/scratch/m.applegate/CTData/fullFrames'
    csvDir = os.path.join('./outOfPlane','dataSplits','ct')
    #csvDir = os.path.join('..','results','ct','dataSplits')
    testCsvs = sorted(glob.glob(os.path.join(csvDir,'*test*.csv')))
    splitToTrain = args.splitNum
    testCsvs = [testCsvs[splitToTrain]] #Two splits
    run = splitToTrain+1
    #methods = ['none','clean','median','gaussian','oofAvg']
    #methodArgs = [0,0,3,1,3]
    methods=[args.method]
    methodArgs = [0.05]
    #iterate through each method
    for method,methodArg in zip(methods,methodArgs):
        if method == 'bm4d':
            numIms=64
        else:
            numIms=256
        #Iterate through each split
        for i,split in enumerate(tqdm.tqdm(testCsvs,desc='%8s'%method,leave=True)):
           
            #All the files in the split
            allFiles=pd.read_csv(split,header=None)
            #Just the center numIms files (plus possibly some extra for stacking)
            imFiles = getRandomNFiles(allFiles,numIms)
            #List of all the patients processed in this split
            scansInSplit = np.unique(imFiles['patient'])
            allData = [] #Data to write to file
            saveDir = os.path.join('..','results','ct','conventional')
            os.makedirs(saveDir,exist_ok=True)
            saveImageDir = os.path.join(saveDir,'Images_%s%s'%(method,methodArg))
            os.makedirs(saveImageDir,exist_ok=True)
            #Iterate through each volume in the split    
            for vol in range(len(scansInSplit)):
                #ScanIms has all the patches in the scan that need to be processed
                scanIms = imFiles[imFiles['patient'] == scansInSplit[vol]]
                scanIms=scanIms.reset_index(drop=True)
                scanDir = os.path.join(saveImageDir,'%s'%scansInSplit[vol])
                saveName = 'testResults_%s%s_run%d_%s.csv'%(method,str(methodArg),run,scansInSplit[vol])
                os.makedirs(scanDir,exist_ok=True)
                #No processing -- get raw noisy Image data
                if method=='none':
                    for i in range(numIms):
                        cleanIm, noisyIm = loadData([scanIms[0][i],scanIms[1][i]],rawDataDir,2)
                        noisyPatches = utils.decimateMosaic(noisyIm,64,overlap=0.5)
                        noisyImage = utils.stitchMosaic(noisyPatches,32)
                        noisyImage8bit = np.clip(np.round(noisyImage*255),0,255).astype('uint8')
                        cleanImage8bit = np.clip(np.round(cleanIm*255),0,255).astype('uint8')
                        #Save the "Processed Image"
                        noisyIm = Image.fromarray(noisyImage8bit)
                        noisyIm.save(os.path.join(scanDir,'frame_%03d.png'%i))
                        #Calculate statistics
                        thisPSNR = PSNR(cleanImage8bit,noisyImage8bit)
                        thisSSIM = SSIM(cleanImage8bit,noisyImage8bit)
                        thisMSE = getMSE(cleanImage8bit,noisyImage8bit)
                        #Store stats
                        allData.append([scansInSplit[vol],i,thisPSNR,thisSSIM,thisMSE])
                elif method ==  'median':
                    for i in range(numIms):
                        cleanIm, noisyIm = loadData([scanIms[0][i],scanIms[1][i]],rawDataDir,2)
                        noisyPatches = utils.decimateMosaic(noisyIm,64,overlap=0.5)
                        filteredPatches = np.zeros(noisyPatches.shape)
                        for j in range(noisyPatches.shape[2]):
                            for k in range(noisyPatches.shape[3]):
                                thisPatch = noisyPatches[:,:,j,k]
                                thisImage = Image.fromarray(thisPatch)
                                thisFiltImage = np.array(thisImage.filter(ImageFilter.MedianFilter(methodArg)))
                                filteredPatches[:,:,j,k] = thisFiltImage
                        
                        stitchImage = utils.stitchMosaic(filteredPatches,32)
                        a,b = utils.getLinearScale(stitchImage,cleanIm)
                        scaledIm = np.clip(np.round((a*stitchImage+b)*255),0,255).astype('uint8')
                        cleanImage8bit = np.clip(np.round(cleanIm*255),0,255).astype('uint8')
                        #Save the Processed Image
                        procIm = Image.fromarray(scaledIm)
                        procIm.save(os.path.join(scanDir,'frame_%03d.png'%i))
                        #Calculate statistics
                        thisPSNR = PSNR(cleanImage8bit,scaledIm)
                        thisSSIM = SSIM(cleanImage8bit,scaledIm)
                        thisMSE = getMSE(cleanImage8bit,scaledIm)
                        allData.append([scansInSplit[vol],i,thisPSNR,thisSSIM,thisMSE])
                elif method =='gaussian':
                    for i in range(numIms):
                        cleanIm, noisyIm = loadData([scanIms[0][i],scanIms[1][i]],rawDataDir,2)
                        noisyPatches = utils.decimateMosaic(noisyIm,64,overlap=0.5)
                        filteredPatches = np.zeros(noisyPatches.shape)
                        for j in range(noisyPatches.shape[2]):
                            for k in range(noisyPatches.shape[3]):
                                thisPatch = noisyPatches[:,:,j,k]
                                thisFiltImage = gaussian_filter(thisPatch,methodArg)
                                filteredPatches[:,:,j,k] = thisFiltImage
                        
                        stitchImage = utils.stitchMosaic(filteredPatches,32)
                        a,b = utils.getLinearScale(stitchImage,cleanIm)
                        scaledIm = np.clip(np.round((a*stitchImage+b)*255),0,255).astype('uint8')
                        cleanImage8bit = np.clip(np.round(cleanIm*255),0,255).astype('uint8')
                        #Save the Processed Image
                        procIm = Image.fromarray(scaledIm)
                        procIm.save(os.path.join(scanDir,'frame_%03d.png'%i))
                        #Calculate statistics
                        thisPSNR = PSNR(cleanImage8bit,scaledIm)
                        thisSSIM = SSIM(cleanImage8bit,scaledIm)
                        thisMSE = getMSE(cleanImage8bit,scaledIm)
                        allData.append([scansInSplit[vol],i,thisPSNR,thisSSIM,thisMSE])
                elif method == 'oofAvg':
                    bracket = int((methodArg-1)/2)
                    for i in range(numIms):
                        cleanIm, noisyIm,prevIm,nextIm = loadData([scanIms[0][i],scanIms[1][i]],rawDataDir,4)
                        noisyStack = np.stack((prevIm,noisyIm,nextIm),axis=2)
                        noisyPatches = utils.decimateMosaic(noisyStack,64,overlap=0.5)
                        filteredPatches = np.zeros((noisyPatches.shape[0],noisyPatches.shape[1],noisyPatches.shape[3],noisyPatches.shape[4]))
                        for j in range(noisyPatches.shape[3]):
                            for k in range(noisyPatches.shape[4]):
                                filteredPatches[:,:,j,k] = np.mean(noisyPatches[:,:,:,j,k],axis=2)
                        avgFrame = utils.stitchMosaic(filteredPatches,32)
                        a,b = utils.getLinearScale(avgFrame,cleanIm)
                        scaledIm = np.clip(np.round((a*avgFrame+b)*255),0,255).astype('uint8')
                        cleanImage8bit = np.clip(np.round(cleanIm*255),0,255).astype('uint8')
                        #Save the Processed Image
                        procIm = Image.fromarray(scaledIm)
                        procIm.save(os.path.join(scanDir,'frame_%03d.png'%i))
                        #Calculate statistics
                        thisPSNR = PSNR(cleanImage8bit,scaledIm)
                        thisSSIM = SSIM(cleanImage8bit,scaledIm)
                        thisMSE = getMSE(cleanImage8bit,scaledIm)
                        allData.append([scansInSplit[vol],i,thisPSNR,thisSSIM,thisMSE])
                elif method == 'bm3d':
                    for i in range(numIms):
                        cleanIm, noisyIm = loadData([scanIms[0][i],scanIms[1][i]],rawDataDir,2)
                        noisyPatches = utils.decimateMosaic(noisyIm,64,overlap=0.5)
                        filteredPatches = np.zeros(noisyPatches.shape)
                        for j in range(noisyPatches.shape[2]):
                            for k in range(noisyPatches.shape[3]):
                                thisPatch = noisyPatches[:,:,j,k]
                                thisFiltImage = bm3d.bm3d(thisPatch,methodArg)
                                filteredPatches[:,:,j,k] = thisFiltImage
                        stitchImage = utils.stitchMosaic(filteredPatches,32)
                        a,b = utils.getLinearScale(stitchImage,cleanIm)
                        scaledIm = np.clip(np.round((a*stitchImage+b)*255),0,255).astype('uint8')
                        cleanImage8bit = np.clip(np.round(cleanIm*255),0,255).astype('uint8')
                        #Save the Processed Image
                        procIm = Image.fromarray(scaledIm)
                        procIm.save(os.path.join(scanDir,'frame_%03d.png'%i))
                        #Calculate statistics
                        thisPSNR = PSNR(cleanImage8bit,scaledIm)
                        thisSSIM = SSIM(cleanImage8bit,scaledIm)
                        thisMSE = getMSE(cleanImage8bit,scaledIm)
                        allData.append([scansInSplit[vol],i,thisPSNR,thisSSIM,thisMSE])
                elif method == 'bm4d':
                    #Only use 3 frames
                    for i in range(numIms):
                        cleanIm, noisyIm,prevIm,nextIm = loadData([scanIms[0][i],scanIms[1][i]],rawDataDir,4)
                        noisyStack = np.stack((prevIm,noisyIm,nextIm),axis=2)
                        noisyPatches = utils.decimateMosaic(noisyStack,64,overlap=0.5)
                        filteredPatches = np.zeros((noisyPatches.shape[0],noisyPatches.shape[1],noisyPatches.shape[3],noisyPatches.shape[4]))
                        for j in range(noisyPatches.shape[3]):
                            for k in range(noisyPatches.shape[4]):
                                thisPatch = noisyPatches[:,:,:,j,k]
                                thisFiltImage = bm4d.bm4d(thisPatch,methodArg)
                                filteredPatches[:,:,j,k] = thisFiltImage[:,:,1]
                        stitchImage = utils.stitchMosaic(filteredPatches,32)
                        a,b = utils.getLinearScale(stitchImage,cleanIm)
                        
                        scaledIm = np.clip(np.round((a*stitchImage+b)*255),0,255).astype('uint8')
                        cleanImage8bit = np.clip(np.round(cleanIm*255),0,255).astype('uint8')
                        #Save the Processed Image
                        procIm = Image.fromarray(scaledIm)
                        procIm.save(os.path.join(scanDir,'frame_%03d.png'%i))
                        #Calculate statistics
                        thisPSNR = PSNR(cleanImage8bit,scaledIm)
                        thisSSIM = SSIM(cleanImage8bit,scaledIm)
                        thisMSE = getMSE(cleanImage8bit,scaledIm)
                        allData.append([scansInSplit[vol],i,thisPSNR,thisSSIM,thisMSE])
                elif method=='clean':
                    for i in range(numIms):
                        #Save the "Processed Image"
                        cleanIm = loadData([scanIms[0][i],scanIms[1][i]],rawDataDir,1)
                        cleanImage8bit = np.clip(np.round(cleanIm*255),0,255).astype('uint8')
                        cleanImage = Image.fromarray(cleanImage8bit)
                        cleanImage.save(os.path.join(scanDir,'frame_%03d.png'%i))
                else:
                    raise ValueError('Method not recognized')
                    
                if method != 'clean':
                    writeData(allData,os.path.join(saveDir,saveName))          
