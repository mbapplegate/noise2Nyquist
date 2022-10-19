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
from utils import utilsOOP as utils


import bm3d
import bm4d
from skimage.metrics import peak_signal_noise_ratio as PSNR
from skimage.metrics import structural_similarity as SSIM
from tqdm import trange

#Calculate the mean squared error between images
def getMSE(target, tester):
    return np.sum((target-tester)**2)/np.size(target)

#writeData
#Inputs:
    #dat:      matrix of data [Scan, Name, PSNR, SSIM, MSE]
    #saveName: File name (including path) to save the data
#Outputs:
    #None
###########################################################
def writeData(dat,saveName):
    with open(saveName,'w') as f:
        f.write('Scan,Image_name, PSNR, SSIM, MSE\n')
        for i in range(len(dat)):
            thisRow = dat[i]
            f.write('%d,%s,%.4f,%.4f,%.6f\n'%(thisRow[0],thisRow[1],thisRow[2],thisRow[3],thisRow[4]))
    return None

#Confocal non-learning methods
if __name__ == '__main__':
    
    simulationSz = 512 #The raw high-res data is 512x512x512
    nyquistSamplingRate = 4 #The sampling rate to use to simulate imaging
    sampMult = 1 #Simulate imaging at the Nyquist rate. 2 would be sub-nyquist, 0.5 would be 2x Nyquist
    simImSz = int(simulationSz/(nyquistSamplingRate*sampMult))
    numReps =10
    #Define the methods to use
    #methods = ['oofAvg','gaussian','none','median','bm3d']
    #Arguments (meaning varies between methods)
    #methodArgs = [3,1,0,3,45]
    methods = ['bm4d']
    methodArgs=[[3,45]]
    for rep in range(numReps):
        print('\nWorking on Rep %d of %d...'%(rep+1,numReps))
        #Where to save the processed data/images
        saveDir= os.path.join('..','..','results','phantom','conventional')
        os.makedirs(saveDir,exist_ok=True)
        #Get the simulated noisy and clean images (8-bit)
        dat = utils.getDataLoader(None,'phantom',numImgs=0,batch_size=1,setMean=0,setStd=1,singleImageTrain=False,shuffleFlag=False,
                          phanFile='../HRPhantomData/YuYeWangPhan.mat',nyquistSampling=4,sampMult=1,nextImFlag=True,noiseStd=45,workers=4,testFlag = True,
                          noiseType='additive')
        
        #Collect the raw and noisy images into a stack to ease processing
        rawNoisyData = np.zeros((simImSz,simImSz,len(dat)),dtype='uint8')
        rawCleanData = np.zeros((simImSz,simImSz,len(dat)),dtype='uint8')
        #Data come out of the dataloader as a Tensor scaled between 0 and 1
        #Need to rescale and convert to 8-bit simulated image
        for i,batch in enumerate(dat):
            rawNoisyData[:,:,i] = (batch[1][0,0,:,:].cpu().numpy()*255).astype('uint8')
            rawCleanData[:,:,i] = (batch[0][0,0,:,:].cpu().numpy()*255).astype('uint8')
        #Run the processing
        for method, methodArg in zip(methods,methodArgs):
            methodDir = os.path.join(saveDir,method,'%02d'%rep)
            os.makedirs(methodDir,exist_ok=True)
            #Name of the file that reflects the method and argument(s)
            saveName = 'testResults_%s%s.csv'%(method,str(methodArg))
            #Space for the results
            allData = []
            #Location where the images will be saved
            saveImageDir = os.path.join(methodDir,'Images_%s%s'%(method,methodArg))
            os.makedirs(saveImageDir,exist_ok=True)
            
            #This big if/else block has one section for each method. It's not elegant
            #but it's clear and it works
            #No processing -- get raw noisy Image data
            if method=='none':
                os.makedirs(saveImageDir,exist_ok=True)
                for i in range(len(dat)):
                    #For fairness, I want each method to decimate the image and then reconstruct it
                    #Even if no processing is done
                    #Split into 64x64 blocks
                    noisyPatches = utils.decimateMosaic(rawNoisyData[:,:,i],64,overlap=0.5)
                    #Stitch back together
                    noisyImage = utils.stitchMosaic(noisyPatches,32)
                    #Convert to 8 bit
                    noisyImage8bit = np.clip(np.round(noisyImage),0,255).astype('uint8')
                    #Save the "Processed Image"
                    noisyIm = Image.fromarray(noisyImage8bit)
                    noisyIm.save(os.path.join(saveImageDir,'frame_%03d.png'%i))
                    #Calculate statistics
                    thisPSNR = PSNR(rawCleanData[:,:,i],noisyImage8bit)
                    thisSSIM = SSIM(rawCleanData[:,:,i],noisyImage8bit)
                    thisMSE = getMSE(rawCleanData[:,:,i],noisyImage8bit)
                    #Store stats
                    allData.append([0,'phantom',thisPSNR,thisSSIM,thisMSE])
            elif method ==  'median':
                for i in range(len(dat)):
                    #Split into patches
                    noisyPatches = utils.decimateMosaic(rawNoisyData[:,:,i],64,overlap=0.5)
                    #Pre-allocate for filtered patches
                    filteredPatches = np.zeros(noisyPatches.shape)
                    #Iterate through each patch
                    for j in range(noisyPatches.shape[2]):
                        for k in range(noisyPatches.shape[3]):
                            thisPatch = noisyPatches[:,:,j,k] #Current patch
                            thisImage = Image.fromarray(thisPatch) #Convert to image
                            #Filter the patch
                            thisFiltImage = np.array(thisImage.filter(ImageFilter.MedianFilter(methodArg)))
                            #Store filtered patch
                            filteredPatches[:,:,j,k] = thisFiltImage
                    #Stitch image together from patches
                    stitchImage = utils.stitchMosaic(filteredPatches,32)
                    #Affinity scaling
                    a,b = utils.getLinearScale(stitchImage,rawCleanData[:,:,i])
                    #Affinity scale and convert to 8 bit
                    scaledIm = np.clip(np.round(a*stitchImage+b),0,255).astype('uint8')
                    #Save the Processed Image
                    procIm = Image.fromarray(scaledIm)
                    procIm.save(os.path.join(saveImageDir,'frame_%03d.png'%i))
                    #Calculate statistics
                    thisPSNR = PSNR(rawCleanData[:,:,i],scaledIm)
                    thisSSIM = SSIM(rawCleanData[:,:,i],scaledIm)
                    thisMSE = getMSE(rawCleanData[:,:,i],scaledIm)
                    allData.append([0,'phantom',thisPSNR,thisSSIM,thisMSE])
            elif method =='gaussian':
                for i in range(len(dat)):
                    #Split into 64x64 patches
                    noisyPatches = utils.decimateMosaic(rawNoisyData[:,:,i],64,overlap=0.5)
                    #Pre-allocate for filtered patches
                    filteredPatches = np.zeros(noisyPatches.shape)
                    for j in range(noisyPatches.shape[2]):
                        for k in range(noisyPatches.shape[3]):
                            #Filter the patches
                            thisPatch = noisyPatches[:,:,j,k]
                            thisFiltImage = gaussian_filter(thisPatch,methodArg)
                            filteredPatches[:,:,j,k] = thisFiltImage
                    #Stitch into a single image
                    stitchImage = utils.stitchMosaic(filteredPatches,32)
                    #Affinity scale
                    a,b = utils.getLinearScale(stitchImage,rawCleanData[:,:,i])
                    #Convert to 8 bit
                    scaledIm = np.clip(np.round(a*stitchImage+b),0,255).astype('uint8')
                    #Save the Processed Image
                    procIm = Image.fromarray(scaledIm)
                    procIm.save(os.path.join(saveImageDir,'frame_%03d.png'%i))
                    #Calculate statistics
                    thisPSNR = PSNR(rawCleanData[:,:,i],scaledIm)
                    thisSSIM = SSIM(rawCleanData[:,:,i],scaledIm)
                    thisMSE = getMSE(rawCleanData[:,:,i],scaledIm)
                    allData.append([0,'phantom',thisPSNR,thisSSIM,thisMSE])
            elif method == 'oofAvg':
                #Bracket is how far to go in each direction (ie for a 3 frame average
                #bracket will be +/- 1, for 5 frame +/- 2, etc.)
                bracket = int((methodArg-1)/2)
                #Get the final size of the patch matrix
                getSz= utils.decimateMosaic(rawNoisyData[:,:,0],64,overlap=0.5)
                #Iterate through all the frames, except the ones on the very end
                for i in range(bracket,len(dat)-bracket):
                    #Set noisyPatches to 0
                    noisyPatches = np.zeros(getSz.shape)
                    #Iterate through the frames to average
                    for z in range(-bracket,bracket+1):
                        #Decimate the appropriate frame and add it to running sum
                        #Not efficient, but this not too slow
                        noisyPatches = noisyPatches+utils.decimateMosaic(rawNoisyData[:,:,i+z],64,overlap=0.5)
                    #Average the noisy frames
                    filteredPatches = noisyPatches/methodArg
                    #Stitch into one big image
                    avgFrame = utils.stitchMosaic(filteredPatches,32)
                    #Affinity scale and convert to 8 bit
                    a,b = utils.getLinearScale(avgFrame,rawCleanData[:,:,i])
                    scaledIm = np.clip(np.round(a*avgFrame+b),0,255).astype('uint8')
                    #Save the Processed Image
                    procIm = Image.fromarray(scaledIm)
                    procIm.save(os.path.join(saveImageDir,'frame_%03d.png'%i))
                    #Calculate statistics
                    thisPSNR = PSNR(rawCleanData[:,:,i],scaledIm)
                    thisSSIM = SSIM(rawCleanData[:,:,i],scaledIm)
                    thisMSE = getMSE(rawCleanData[:,:,i],scaledIm)
                    allData.append([0,'phantom',thisPSNR,thisSSIM,thisMSE])
            elif method == 'bm3d':
                #Iterate through each image
                for i in trange(len(dat)):
                    #Split into 64x64 pixel patches
                    noisyPatches = utils.decimateMosaic(rawNoisyData[:,:,i],64,overlap=0.5)
                    #Pre-allocate for filtered patches
                    filteredPatches = np.zeros(noisyPatches.shape)
                    #Iterate through each patch
                    for j in range(noisyPatches.shape[2]):
                        for k in range(noisyPatches.shape[3]):
                            #Filter each patch
                            thisPatch = noisyPatches[:,:,j,k]
                            thisFiltImage = bm3d.bm3d(thisPatch,methodArg)
                            filteredPatches[:,:,j,k] = thisFiltImage
                    #Stitch the filtered patch
                    stitchImage = utils.stitchMosaic(filteredPatches,32)
                    #Affinity scale and convert to 8-bit
                    a,b = utils.getLinearScale(stitchImage,rawCleanData[:,:,i])
                    scaledIm = np.clip(np.round(a*stitchImage+b),0,255).astype('uint8')
                    #Save the Processed Image
                    procIm = Image.fromarray(scaledIm)
                    procIm.save(os.path.join(saveImageDir,'frame_%03d.png'%i))
                    #Calculate statistics
                    thisPSNR = PSNR(rawCleanData[:,:,i],scaledIm)
                    thisSSIM = SSIM(rawCleanData[:,:,i],scaledIm)
                    thisMSE = getMSE(rawCleanData[:,:,i],scaledIm)
                    allData.append([0,'phantom',thisPSNR,thisSSIM,thisMSE])
            elif method == 'bm4d':
                #Only BM4D on 3 frames
                bracket = int((methodArg[0]-1)/2)
                #Iterate through each frame (except the ends)
                for i in trange(bracket,len(dat)-bracket):
                    #Make a stack of the noisy data (full frames)
                    thisSubStack = rawNoisyData[:,:,i-bracket:i+bracket+1]
                    #Split the stack into 64x64x3 x Nx x Ny patches
                    noisyPatches = utils.decimateMosaic(thisSubStack,64,overlap=0.5)
                    #Pre-allocate space for filtered patches
                    filteredPatches = np.zeros((noisyPatches.shape[0],noisyPatches.shape[1],noisyPatches.shape[3],noisyPatches.shape[4]))
                    for j in range(noisyPatches.shape[3]):
                        for k in range(noisyPatches.shape[4]):
                            #noisy stack
                            thisPatch = noisyPatches[:,:,:,j,k]
                            #Filter the stack
                            thisFiltImage = bm4d.bm4d(thisPatch,methodArg[1])
                            #Collect the filtered patch
                            filteredPatches[:,:,j,k] = thisFiltImage[:,:,1]
                    #Stitch patches into image
                    stitchImage = utils.stitchMosaic(filteredPatches,32)
                    #Affinity scale and convert to 8-bit
                    a,b = utils.getLinearScale(stitchImage,rawCleanData[:,:,i])
                    scaledIm = np.clip(np.round(a*stitchImage+b),0,255).astype('uint8')
                    #Save the Processed Image
                    procIm = Image.fromarray(scaledIm)
                    procIm.save(os.path.join(saveImageDir,'frame_%03d.png'%i))
                    #Calculate statistics
                    thisPSNR = PSNR(rawCleanData[:,:,i],scaledIm)
                    thisSSIM = SSIM(rawCleanData[:,:,i],scaledIm)
                    thisMSE = getMSE(rawCleanData[:,:,i],scaledIm)
                    allData.append([0,'phantom',thisPSNR,thisSSIM,thisMSE])
            elif method=='clean':
                #With clean, I didn't bother decimating and reassembling
                os.makedirs(saveImageDir,exist_ok=True)
                for i in range(len(dat)):
                    #Save the "Processed Image"
                    cleanIm = Image.fromarray(rawCleanData[:,:,i])
                    cleanIm.save(os.path.join(saveImageDir,'frame_%03d.png'%i))
            #Write the PSNR, SSIM, and MSE to a file
            if method != 'clean':
                writeData(allData,os.path.join(methodDir,saveName))          