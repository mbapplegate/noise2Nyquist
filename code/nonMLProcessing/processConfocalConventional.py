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
from skimage.metrics import peak_signal_noise_ratio as PSNR
from skimage.metrics import structural_similarity as SSIM
import tqdm
from utils.utilsOOP import getLinearScale

#Calculate the mean squared error between two images
def getMSE(target, tester):
    return np.sum((target-tester)**2)/np.size(target)

#Function to return random patches from different confocal volumes
#Inputs:
#   dataframe: Pandas dataframe with filepaths of all the files
#   nFiles:    Number of files to return
#Outputs:
    #outFrame: A list of dataframes with one dataframe per volume
#################################################
def getRandomNFiles(dataframe,nFiles):
    scans = []
    outFrame = pd.DataFrame({0:[],1:[]})
    #Look at the whole dataframe and assign a scan to each entry
    for i in range(len(dataframe)):
        thisPath = dataframe[0][i]
        #The path contains the volume number 
        thisScanNum = int(os.path.dirname(thisPath)[-2:])
        scans.append(thisScanNum)
    scans = np.array(scans)
    dataframe['scan'] = scans

    numScans = np.unique(scans)
    #For each scan append a dataframe with randomly selected entries
    for j in range(len(numScans)):
        #Filter dataframe to this scan only
        filtDF = dataframe[dataframe['scan']==numScans[j]]
        getRandomIdxs = np.random.randint(0,len(filtDF),nFiles)
        randDF = filtDF.iloc[getRandomIdxs]   
        outFrame=pd.concat((outFrame,randDF),ignore_index=True)
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
            f.write('%d,%s,%.4f,%.4f,%.6f\n'%(thisRow[0],thisRow[1],thisRow[2],thisRow[3],thisRow[4]))
    return None


#loadData loads up to 4 images and returns them
#Inputs:
#   imPaths:     Filename of the image to load 
#   dataDir:     Location where the images are stored 
#   imsToReturn: Number of images to return [1,2, or 4] always clean, sometimes noisy, sometimes prev. and next.
#Outputs:
#   cleanIm: cleanPatch
#   noisyIm: noisyPatch
#   prevIm:  co-located noisy patch from the previous Z frame
#   nextIm:  co-located noisy patch from the following Z frame
###############################
def loadData(imPaths,dataDir,imsToReturn = 4):
    cleanPath = imPaths[0]
    noisyPath = imPaths[1]
    #Get path parts
    cleanPathParts = cleanPath.split('/')
    noisyPathParts = noisyPath.split('/')
    #Build a path to the clean image
    cleanIm = np.load(os.path.join(dataDir,'clean',cleanPathParts[-2],cleanPathParts[-1]))
    if imsToReturn == 1:
        return cleanIm
    #Build a path to the noisy image
    noisyIm = np.load(os.path.join(dataDir,noisyPathParts[-4],'current',noisyPathParts[-2],noisyPathParts[-1]))
    if imsToReturn == 2:
        return cleanIm, noisyIm
    #Build paths for the rest
    prevIm = np.load(os.path.join(dataDir,noisyPathParts[-4],'prev',noisyPathParts[-2],noisyPathParts[-1]))
    nextIm = np.load(os.path.join(dataDir,noisyPathParts[-4],'next',noisyPathParts[-2],noisyPathParts[-1]))
    return cleanIm,noisyIm,prevIm,nextIm

#Confocal non-learning methods
if __name__ == '__main__':
    #Data were processed across all different devices
    #RawDataDirectory is where the data is located on this device
    rawDataDirectory = '/scratch/m.applegate/Denoising_Planaria/patches064'
    #Location of CSVs used for processing the data
    csvDir = os.path.join('..','results','confocal','dataSplits')
    #Test splits of the CSVs so it's easy to compare splits with ML methods
    testCsvs = sorted(glob.glob(os.path.join(csvDir,'*test*.csv')))
    #Methods to process and their arguments
    methods = ['none','median','gaussian','oofAvg','bm3d','bm4d','clean']
    #Arguments for the methods
    methodArgs = [0,3,1,3,0.2,0.2,0];
     
    #How many 64x64 pixel patches to process
    numIms=1536
    #Iterate through each method
    for method,methodArg in zip(methods,methodArgs):
        #Iterate through each split
        for i,split in enumerate(tqdm.tqdm(testCsvs,desc='%8s'%method,leave=True)):
            allFiles=pd.read_csv(split,header=None) #All of the patches in this split
           
            #Get N patches per scan 
            imFiles = getRandomNFiles(allFiles,numIms) 
            scansInSplit = np.unique(imFiles['scan'][:]) 
            
            allDat = [] #Place to store all the data from this split
            saveDir = os.path.join('..','results','confocal','conventional','%02d'%i)
            os.makedirs(saveDir,exist_ok=True)
            saveImageDir = os.path.join(saveDir,'Images_%s%s'%(method,methodArg))
            os.makedirs(saveImageDir,exist_ok=True)
            
            #Iterate through each volume in the split    
            for vol in range(len(scansInSplit)):
                #ScanIms has all the patches in the scan that need to be processed
                scanIms = imFiles[imFiles['scan'] == scansInSplit[vol]]
                scanIms=scanIms.reset_index(drop=True)
                #Place to store the processed images
                scanDir = os.path.join(saveImageDir,'Scan%02d'%scansInSplit[vol])
                os.makedirs(scanDir,exist_ok=True)
                #Each block is a different method
                if method=='none':
                    for i in range(len(scanIms)):
                        cleanIm, noisyIm = loadData([scanIms[0][i],scanIms[1][i]],rawDataDirectory,2)
                        noisyImage8bit = np.clip(np.round(noisyIm*255),0,255).astype('uint8')
                        cleanIm8bit = np.clip(np.round(cleanIm*255),0,255).astype('uint8')
                        #Save the "Processed Image"
                        thisIm = Image.fromarray(noisyImage8bit)
                        thisIm.save(os.path.join(scanDir,'frame_%03d.png'%i))
                        #Calculate statistics
                        thisPSNR = PSNR(cleanIm8bit,noisyImage8bit)
                        thisSSIM = SSIM(cleanIm8bit,noisyImage8bit)
                        thisMSE = getMSE(cleanIm8bit,noisyImage8bit)
                        #Store stats
                        allDat.append([scansInSplit[vol],i,thisPSNR,thisSSIM,thisMSE])
                elif method ==  'median':
                    for i in range(len(scanIms)):
                        cleanIm, noisyIm = loadData([scanIms[0][i],scanIms[1][i]],rawDataDirectory,2)
                        noisyPIL = Image.fromarray(noisyIm)
                        filtImage = np.array(noisyPIL.filter(ImageFilter.MedianFilter(methodArg)))
                        #affinity scaling
                        a,b = getLinearScale(filtImage,cleanIm)
                        scaledIm = np.clip(np.round((a*filtImage+b)*255),0,255).astype('uint8')
                        cleanIm8bit = np.clip(np.round(cleanIm*255),0,255).astype('uint8')
                        #Save the Processed Image
                        thisIm = Image.fromarray(scaledIm)
                        thisIm.save(os.path.join(scanDir,'frame_%03d.png'%i))
                        #Calculate statistics
                        thisPSNR = PSNR(cleanIm8bit,scaledIm)
                        thisSSIM = SSIM(cleanIm8bit,scaledIm)
                        thisMSE = getMSE(cleanIm8bit,scaledIm)
                        allDat.append([scansInSplit[vol],i,thisPSNR,thisSSIM,thisMSE])
                elif method =='gaussian':
                    for i in range(len(scanIms)):
                        cleanIm, noisyIm = loadData([scanIms[0][i],scanIms[1][i]],rawDataDirectory,2)
                        filtImage = gaussian_filter(noisyIm,methodArg)
                        #affinity scaling
                        a,b = getLinearScale(filtImage,cleanIm)
                        scaledIm = np.clip(np.round((a*filtImage+b)*255),0,255).astype('uint8')
                        cleanIm8bit = np.clip(np.round(cleanIm*255),0,255).astype('uint8')
                        #Save the Processed Image
                        thisIm = Image.fromarray(scaledIm)
                        thisIm.save(os.path.join(scanDir,'frame_%03d.png'%i))
                        #Calculate statistics
                        thisPSNR = PSNR(cleanIm8bit,scaledIm)
                        thisSSIM = SSIM(cleanIm8bit,scaledIm)
                        thisMSE = getMSE(cleanIm8bit,scaledIm)
                        allDat.append([scansInSplit[vol],i,thisPSNR,thisSSIM,thisMSE])
                elif method == 'oofAvg':
                    assert methodArg == 3 #Don't have an easy way to do more (or less) averaging
                    for i in range(numIms):
                        cleanIm, noisyIm,prevIm,nextIm = loadData([scanIms[0][i],scanIms[1][i]],rawDataDirectory,4)
                        noisyStack= np.stack((prevIm,noisyIm,nextIm),axis=2)
                        avgFrame = np.mean(noisyStack,axis=2)
                        #affinity scaling
                        a,b = getLinearScale(avgFrame,cleanIm)
                        scaledIm = np.clip(np.round((a*avgFrame+b)*255),0,255).astype('uint8')
                        cleanIm8bit = np.clip(np.round(cleanIm*255),0,255).astype('uint8')
                        #Save the Processed Image
                        thisIm = Image.fromarray(scaledIm)
                        thisIm.save(os.path.join(scanDir,'frame_%03d.png'%i))
                        #Calculate statistics
                        thisPSNR = PSNR(cleanIm8bit,scaledIm)
                        thisSSIM = SSIM(cleanIm8bit,scaledIm)
                        thisMSE = getMSE(cleanIm8bit,scaledIm)
                        allDat.append([scansInSplit[vol],i,thisPSNR,thisSSIM,thisMSE])
                elif method == 'bm3d':
                    for i in range(numIms):
                        cleanIm, noisyIm = loadData([scanIms[0][i],scanIms[1][i]],rawDataDirectory,2)
                        filtImage = bm3d.bm3d(noisyIm,methodArg)
                        #affinity scaling
                        a,b = getLinearScale(filtImage,noisyIm)
                        scaledIm = np.clip(np.round((a*filtImage+b)*255),0,255).astype('uint8')
                        cleanIm8bit = np.clip(np.round(cleanIm*255),0,255).astype('uint8')
                        #Save the Processed Image
                        thisIm = Image.fromarray(scaledIm)
                        thisIm.save(os.path.join(scanDir,'frame_%03d.png'%i))
                        #Calculate statistics
                        thisPSNR = PSNR(cleanIm8bit,scaledIm)
                        thisSSIM = SSIM(cleanIm8bit,scaledIm)
                        thisMSE = getMSE(cleanIm8bit,scaledIm)
                        allDat.append([scansInSplit[vol],i,thisPSNR,thisSSIM,thisMSE])
                           
                elif method == 'bm4d':
                    for i in range(numIms):
                        cleanIm, noisyIm,prevIm,nextIm = loadData([scanIms[0][i],scanIms[1][i]],rawDataDirectory,4)
                        noisyStack= np.stack((prevIm,noisyIm,nextIm),axis=2)
                        #Process with BM4D
                        filtFrame = bm4d.bm4d(noisyStack,methodArg)
                        #affinity scaling
                        a,b = getLinearScale(filtFrame[:,:,1],cleanIm)
                        scaledIm = np.clip(np.round((a*filtFrame[:,:,1]+b)*255),0,255).astype('uint8')
                        cleanIm8bit = np.clip(np.round(cleanIm*255),0,255).astype('uint8')
                        #Save the Processed Image
                        thisIm = Image.fromarray(scaledIm)
                        thisIm.save(os.path.join(scanDir,'frame_%03d.png'%i))
                        #Calculate statistics
                        thisPSNR = PSNR(cleanIm8bit,scaledIm)
                        thisSSIM = SSIM(cleanIm8bit,scaledIm)
                        thisMSE = getMSE(cleanIm8bit,scaledIm)
                        allDat.append([scansInSplit[vol],i,thisPSNR,thisSSIM,thisMSE])
                elif method == 'clean':
                    for i in range(numIms):
                        #Don't bother with the decimation and rebuilding
                        cleanIm = loadData([scanIms[0][i],scanIms[1][i]],rawDataDirectory,1)
                        cleanIm8bit = np.clip(np.round(cleanIm*255),0,255).astype('uint8')
                        #Save the Processed Image
                        thisIm = Image.fromarray(cleanIm8bit)
                        thisIm.save(os.path.join(scanDir,'frame_%03d.png'%i))
            if method != 'clean':           
                saveName = 'testResults_%s%s.csv'%(method,methodArg)
                writeData(allDat,os.path.join(saveDir,saveName))
            
        
        
