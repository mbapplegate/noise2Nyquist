#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 14 09:32:35 2022

@author: matthew
"""
import os
import glob
import numpy as np
import scipy.io
import scipy.signal
import pandas as pd
import sklearn.model_selection

import torch
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
import torchvision.transforms as T
from matplotlib import pyplot as plt
from  utils.models import UNet,UNet1D
import utils.arch_unet
from torchvision import utils as U
from PIL import Image,ImageFilter

#Phantom dataset
class SheppLoganDataset(Dataset):
    #Inputs:
        #sheppMatFile:    Mat file to load containing High Res. phantom data
        #nyquistSampling: Nyquist sampling rate in pixels
        #sampMult:        x * nyquistSampling is new rate. So 2 is half the nyqusit rate. 0.5 is twice the nyquist rate
        #noiseStd:        Standard deviation of noise to add
        #nextImage:       Should I group adjacent frames
        #singleImTrain:   Am I just using 1 image to train?
        #transform::      Augmentations to use
        #noiseType:       Additive or multiplicative noise
    ####################################################################
    def __init__(self,sheppMatFile,nyquistSampling,sampMult=1,noiseStd=40,nextImage=True,singleImTrain=False,transform=None,noiseType='additive'):
        super(SheppLoganDataset,self).__init__()
        self.transform = transform
        self.nextImage = nextImage
        self.noiseStd=noiseStd
        self.useSingleIm = singleImTrain
       
        #Get the Shepp Logan phantom
        print('Loading phantom...')
        phanDict = scipy.io.loadmat(sheppMatFile)
        self.truePhantom=np.array(phanDict[list(phanDict.keys())[-1]]) #head phantom 512x512x512 values:0-1
        self.truePhantom = self.truePhantom[:,:,50:-50]
        #Defines the nyquist sampling rate for this image
        self.nyquistSampling = nyquistSampling
        #Ratio of nyquist sampling to sample the images at
        self.sampMult = sampMult
        self.sampSpacingZ = int(np.round(nyquistSampling*sampMult))
        if self.useSingleIm:
            self.sampSpacingXY=int(np.round(nyquistSampling*sampMult))
        else:
            self.sampSpacingXY=int(np.round(nyquistSampling))
        #The FWHM of the point spread function at this sampling rate
        self.psfFWHM = (nyquistSampling * 2 * 0.51)/(0.61)
        #The standard deviation of the PSF
        self.psfStd = self.psfFWHM / (2*np.sqrt(2*np.log(2)))
        
        #Get the shape (not sure if necessary)
        self.x,self.y,self.z = np.shape(self.truePhantom)
        #The size of the PSF will be 8x the std (+/- 4)
        psfSz = int(np.round(4*self.psfStd))
        #Coordinates of PSF
        X,Y,Z = np.meshgrid(range(-psfSz,psfSz),range(-psfSz,psfSz),range(-psfSz,psfSz))
        #PSF itself
        self.PSF = gauss3d(X,Y,Z,0,0,0,self.psfStd,self.psfStd,self.psfStd)
        #True image of the phantom with given PSF
        print('Simulating Images...')
        self.trueImage = scipy.signal.convolve(self.truePhantom,self.PSF,mode='same')
        self.trueImage = self.trueImage/np.max(self.trueImage)
        #Starting location of the sampling grid
        startX,startY = np.random.randint(0,self.sampSpacingXY,2)
        startZ = np.random.randint(0,self.sampSpacingZ)
        self.sampledVolume = self.trueImage[startX::self.sampSpacingXY,startY::self.sampSpacingXY,startZ::self.sampSpacingZ]
        #Convert to 8 bit
        self.sampledVolume = (self.sampledVolume*255).astype('uint8')
        print('Sample generation finished!')
        #Define Noise
        awgn1 = np.random.normal(0,noiseStd,size=self.sampledVolume.shape)
        awgn2 = np.random.normal(0,noiseStd,size=self.sampledVolume.shape)
        if noiseType == 'additive':
            #Two different noise realizations for noise2noise algo
            self.noiseVolume1 = np.clip(np.round(self.sampledVolume+awgn1),0,255).astype('uint8')
            self.noiseVolume2 = np.clip(np.round(self.sampledVolume+awgn2),0,255).astype('uint8')
        #Correlated noise
        elif noiseType == 'corrMult':
            correlation_scale = 1.25
            x = np.arange(-correlation_scale, correlation_scale+1)
            y = np.arange(-correlation_scale, correlation_scale+1)
            z = np.arange(-correlation_scale, correlation_scale+1)
            X, Y, Z = np.meshgrid(x,y,z)
            dist = np.sqrt(X*X + Y*Y + Z*Z)
            filter_kernel = np.exp(-2*dist**2/(2*correlation_scale))
            noise1 = np.random.randn(self.sampledVolume.shape[0],self.sampledVolume.shape[1],self.sampledVolume.shape[2])*.2#*noiseStd/(1.5*correlation_scale)
            noise2 = np.random.randn(self.sampledVolume.shape[0],self.sampledVolume.shape[1],self.sampledVolume.shape[2])*.2#*noiseStd/(1.5*correlation_scale)
            noiseCorr1 = scipy.signal.fftconvolve(noise1, filter_kernel, mode='same')
            noiseCorr2 = scipy.signal.fftconvolve(noise2, filter_kernel, mode='same')
            
            self.noiseVolume1 = np.clip(awgn1+self.sampledVolume + self.sampledVolume*noiseCorr1,0,255).astype('uint8')
            self.noiseVolume2 = np.clip(awgn2+self.sampledVolume + self.sampledVolume*noiseCorr2,0,255).astype('uint8')
            #self.noiseVolume1 = np.clip(noiseCorr1+self.sampledVolume,0,255).astype('uint8')
            #self.noiseVolume2 = np.clip(noiseCorr2+self.sampledVolume,0,255).astype('uint8')
        else:
            raise ValueError('Unrecognized noise type')
        #self.noiseVolume1 = ((nv1 - np.min(nv1))/(np.max(nv1)-np.min(nv1))*255).astype('uint8')
        #self.noiseVolume2 = ((nv2-np.min(nv2))/(np.max(nv2)-np.min(nv2))*255).astype('uint8')
        #self.shiftAmt=1
    def __len__(self):
        if self.nextImage:
            return np.shape(self.sampledVolume)[2]-1
        else:
            return np.shape(self.sampledVolume)[2]
    #Returns [CleanIm, noisyIm, next(or prev or 2nd noisy)Im]
    def __getitem__(self,idx):
        img = []
        trueIm = self.sampledVolume[:,:,idx]
        noiseIm = self.noiseVolume1[:,:,idx]
        if self.nextImage:
            if idx == 0:
                nextIm = self.noiseVolume1[:,:,idx+1]
            else:
                if np.random.choice([True,False]):
                    nextIm = self.noiseVolume1[:,:,idx+1]
                else:
                    nextIm = self.noiseVolume1[:,:,idx-1]
        else:
            nextIm = self.noiseVolume2[:,:,idx]
        img.append(trueIm)
        img.append(noiseIm)
        img.append(nextIm)
        if self.transform:
            sample=self.transform(img)
        else:
            sample = img
       
        return sample
    
# define normalized 3D gaussian point spread function for phantom dataset
def gauss3d(x,y,z, mx=0, my=0,mz=0,psfStdx=1,psfStdy=1,psfStdz=1):
    return  np.exp(-((x - mx)**2. / (2. * psfStdx**2.) + (y - my)**2. / (2. * psfStdy**2.)+ (z - mz)**2. / (2. * psfStdz**2.)))

#Dataset with PRENORMALIZED confocal images in it
#All images have been normalized 0-1 by percentile
#See saveFluorescencePatches.py for details
class confocalDataset(Dataset):
    #Inputs:
        #csv:       Where is the CSV with the file list
        #numImgs:   How many images (0 is all)
        #transform: What transforms to use
        #testFlag:  Is this a test set?
    ##############################################
    def __init__(self,csv,numImgs=0,transform=None,testFlag=False):
        super(confocalDataset,self).__init__()
        self.files = pd.read_csv(csv)
        self.transform = transform
        self.imgLen = numImgs
        self.testFlag = testFlag
    #Length of the dataset
    def __len__(self):
        if self.imgLen==0 or self.imgLen > len(self.files):
            return len(self.files)
        else:
            return self.imgLen
    #If it's the test set
    #   returns: [cleanIm, noisyIm, imName]
    #If it's a training set
    #   returns: [cleanIm, noisyIm, next(or Prev)Im]
    def __getitem__(self,idx):
        img = []
        if self.testFlag:
            cleanPath = self.files.iloc[idx,0]
            currPath = self.files.iloc[idx,1]
           
            imDir,fName = os.path.split(cleanPath)
            scanName = imDir.split(os.path.sep)[-1]
            imName = scanName+'_'+fName
            thisClean = np.load(cleanPath)
            thisNoisy = np.load(currPath)
           
            img.append(thisClean.astype('float32'))
            img.append(thisNoisy.astype('float32'))
            sample=self.transform(img)
            sample.append(imName)
            return sample
        else:
            cleanPath = self.files.iloc[idx,0]
            prevPath = self.files.iloc[idx,1]
            currPath = self.files.iloc[idx,2]
            nextPath = self.files.iloc[idx,3]
           
            thisClean = np.load(cleanPath)
            thisNoisy = np.load(currPath)
            if np.random.choice([True,False]):
                thisNext = np.load(prevPath)
            else:
                thisNext = np.load(nextPath)
            
            img.append(thisClean.astype('float32'))
            img.append(thisNoisy.astype('float32'))
            img.append(thisNext.astype('float32'))
                
            if self.transform is not None:
                sample = self.transform(img)
                return sample
            else:
                return img
   
#Dataset with OCT images       
class octDataset(Dataset):
    #Inputs:
        #csv:       Where is the CSV with the file list
        #numImgs:   How many images (0 is all)
        #transform: What transforms to use
        #testFlag:  Is this a test set?
    ##############################################
    def __init__(self,csv,numImgs=0,transform=None,testFlag=False):
        super(octDataset,self).__init__()
        self.files = pd.read_csv(csv)
        self.transform = transform
        self.imgLen = numImgs
        self.testFlag = testFlag
    #Defines dataset length
    def __len__(self):
        if self.imgLen==0 or self.imgLen > len(self.files):
            return len(self.files)
        else:
            return self.imgLen
    #Get item from dataset
    #If it's from the test set
    #   returns: [noisyIm, imName]
    #If it's from the training set
    #   returns: [noisyIm, nextIm]
    def __getitem__(self,idx):
        img = []
        if self.testFlag:
            currPath = self.files.iloc[idx,0]
            imDir,fName = os.path.split(currPath)
            scanName = imDir.split(os.path.sep)[-1]
            imName = scanName+'_'+fName
            thisNoisy = np.load(currPath)
            img.append(thisNoisy)
            sample = self.transform(img)
            sample.append(imName)
            return sample
        else:
            prevPath = self.files.iloc[idx,0]
            currPath = self.files.iloc[idx,1]
            nextPath = self.files.iloc[idx,2]
            thisNoisy = np.load(currPath)
    
            if np.random.choice([True,False]):
                thisNext = np.load(prevPath)
            else:
                thisNext = np.load(nextPath)
              
            img.append(thisNoisy)
            img.append(thisNext)
                
            if self.transform is not None:
                sample = self.transform(img)
                return sample
            else:
                return img
            
#Dataset with CT Images
class CTDataset(Dataset):
    #Inputs:
        #csv:       Where is the CSV with the file list
        #numImgs:   How many images (0 is all)
        #transform: What transforms to use
        #testFlag:  Is this a test set?
    ##############################################
    def __init__(self,csv,numImgs=0,transform=None,testFlag=False):
        super(CTDataset,self).__init__()
        self.files = pd.read_csv(csv)
        self.transform = transform
        self.imgLen = numImgs
        self.testFlag = testFlag
    #Define dataset length
    def __len__(self):
        if self.imgLen==0 or self.imgLen > len(self.files):
            return len(self.files)
        else:
            return self.imgLen
    #Get a sample
    #If testSet:
        #Returns [cleanIm, noisyIm, imName]
    #If training set
        #Returns [cleanIm, noisyIm, next(or previous)Im]
    def __getitem__(self,idx):
        img = []
        #If it's the test set only load one clean and one noisy image
        if self.testFlag:
            cleanPath = self.files.iloc[idx,0]
            currPath = self.files.iloc[idx,1]
            
            imDir,fName = os.path.split(cleanPath)
            scanName = imDir.split(os.path.sep)[-1]
            imName = scanName+'_'+fName
            #Load the numpy files
            thisClean = np.load(cleanPath)
            thisNoisy = np.load(currPath)
            #Normalize clean and noisy data 0-1
            cleanPercs = np.percentile(thisClean,[0.1,99.9])
            noisyPercs = np.percentile(thisNoisy,[2,99.7])
            normClean = np.clip((thisClean-cleanPercs[0])/(cleanPercs[1]-cleanPercs[0]),0,1)
            normNoisy = np.clip((thisNoisy-noisyPercs[0])/(noisyPercs[1]-noisyPercs[0]),0,1)
            #Append images
            img.append(normClean.astype('float32'))
            img.append(normNoisy.astype('float32'))
            #Transform images
            sample = self.transform(img)
            sample.append(imName)
            return sample
        else:
            #Images paths from the csv
            cleanPath = self.files.iloc[idx,0]
            prevPath = self.files.iloc[idx,1]
            currPath = self.files.iloc[idx,2]
            nextPath = self.files.iloc[idx,3]
            #Load the clean and noisy
            thisClean = np.load(cleanPath)
            thisNoisy = np.load(currPath)
            #Choose whether next or previous is selected
            if np.random.choice([True,False]):
                thisNext = np.load(prevPath)
            else:
                thisNext = np.load(nextPath)
            #Normalize images
            cleanPercs = np.percentile(thisClean,[0.1,99.9])
            noisyPercs = np.percentile(thisNoisy,[2,99.7])
            nextPercs = np.percentile(thisNext,[2,99.7])
            
            normClean = np.clip((thisClean-cleanPercs[0])/(cleanPercs[1]-cleanPercs[0]),0,1)
            normNoisy = np.clip((thisNoisy-noisyPercs[0])/(noisyPercs[1]-noisyPercs[0]),0,1)
            normNext = np.clip((thisNext-nextPercs[0])/(nextPercs[1]-nextPercs[0]),0,1)
            #Append images
            img.append(normClean.astype('float32'))
            img.append(normNoisy.astype('float32'))
            img.append(normNext.astype('float32'))
            #Transform images
            if self.transform is not None:
                sample = self.transform(img)
                return sample
            else:
                return img
            
#Dataset with RCM Data in it       
class RCMDataset(Dataset):
    def __init__(self,csv,numImgs=0,transform=None,testFlag=False):
        super(RCMDataset,self).__init__()
        self.files = pd.read_csv(csv)
        self.transform = transform
        self.imgLen = numImgs
        self.testFlag = testFlag
    def __len__(self):
        if self.imgLen==0 or self.imgLen > len(self.files):
            return len(self.files)
        else:
            return self.imgLen
    def __getitem__(self,idx):
        img = []
        if self.testFlag:
            currPath = self.files.iloc[idx,0]
            imDir,fName = os.path.split(currPath)
            scanName = imDir.split(os.path.sep)[-1]
            imName = scanName+'_'+fName
            thisNoisy = np.load(currPath)
            img.append(thisNoisy)
            sample = self.transform(img)
            sample.append(imName)
            return sample
        else:
            imPath = self.files.iloc[idx,0]
            thisIm = np.load(imPath)
            img.append(thisIm) 
            img.append(thisIm.copy())
            if self.transform is not None:
                sample = self.transform(img)
                return sample
            else:
                return img
    
#Regularization term using Total Variation
class lossWithTV(torch.nn.Module):
    def __init__(self,TVwt,lossType):
        super(lossWithTV,self).__init__()
        if lossType=='l1':
            self.criterion = torch.nn.L1Loss()
        elif lossType == 'l2':
            self.criterion = torch.nn.MSELoss()
        else:
            raise ValueError('Loss type not recognized')
        self.TVwt = TVwt
    #Calculate total variation    
    def forward(self,inputs,targets):
        cLoss=self.criterion(inputs,targets)
        loss=0
        if len(inputs.shape) == 4:
            loss += torch.sum((inputs[:,:,1:,:]-inputs[:,:,:-1,:])**2)
            loss += torch.sum((inputs[:,:,:,1:]-inputs[:,:,:,:-1])**2)
        elif len(inputs.shape) == 3:
            loss += torch.sum((inputs[:,1:,:]-inputs[:,:-1,:])**2)
            loss += torch.sum((inputs[:,:,1:]-inputs[:,:,:-1])**2)
        return cLoss + self.TVwt*loss

#Function to get a dataloader from a csv file
#Inputs:
    #csv:              Name of the csv that contains image file locations
    #dataType:         What type of data is this
    #numImgs:          No. images to load
    #batch_size:       No. images in minibatch
    #setMean:          Avg of the training set for normalization
    #setStd:           Standard deviation of the dataset for normalization
    #SingleImageTrain: Flag for whether you want to train with a single image (noise2void or line2line)
    #ShuffleFlag:      Should the data be shuffled?
    #phanFile:         (phantom data specific) Location of phantom file
    #nyquistSampling:  (phantom data specific) Nyquist rate in pixels [must be > 1]
    #sampMult:         (phantom data specific) Sampling rate relative to Nyquist rate
    #nextImFlag:       Train with adjacent frames (noise2Nyquist)
    #noiseStd:         (phantom data specific) Standard deviation of noise to add
    #workers:          Number of workers to assign
    #testFlag:         Is this a test set (controls data augmentation)
    #noiseType:        (phantom data specific) Corrupt with 'additive' or 'multiplicative' noise
#Outputs:
    #dataloader: A pytorch dataloader object with all the junk needed to work it
###########################################################################################
def getDataLoader(csv,dataType,numImgs=0,batch_size=16,setMean=0,setStd=1,singleImageTrain=False,shuffleFlag=True,
                  phanFile=None,nyquistSampling=4,sampMult=1,nextImFlag=True,noiseStd=40,workers=4,testFlag = False,
                  noiseType='additive'): 
    #Only augment data during training
    if testFlag:
        transforms = T.Compose([ToTensor(),NormalizeTensors(setMean,setStd)])
    else:
        transforms = T.Compose([flipAugment(),ToTensor(),NormalizeTensors(setMean,setStd)])
    
    #Get the appropriate dataset for the datatype
    if dataType.lower() == 'phantom':
        dataset = SheppLoganDataset(phanFile,nyquistSampling,sampMult,noiseStd=noiseStd,
                                    transform=transforms,nextImage=nextImFlag,singleImTrain=singleImageTrain,
                                    noiseType=noiseType)
    elif dataType.lower() == 'confocal':
        dataset = confocalDataset(csv,numImgs=numImgs,transform=transforms,testFlag=testFlag)
    elif dataType.lower() == 'oct':
        dataset = octDataset(csv,numImgs=numImgs,transform=transforms,testFlag=testFlag)
    # elif dataType.lower() == 'ultrasound':
    #     dataset = ultrasoundDataset(csv,numImgs=numImgs,transform=transforms,testFlag=testFlag) 
    elif dataType.lower() == 'ct':
        dataset = CTDataset(csv,numImgs=numImgs,transform=transforms,testFlag=testFlag)
    elif dataType.lower() == 'rcm':
        dataset = RCMDataset(csv,numImgs=numImgs,transform=transforms,testFlag=testFlag)
    else:
        raise ValueError('Only "phantom", "confocal", and "oct" data implemented so far. You used: %s'%dataType)
    #Make a dataloader
    dataloader = DataLoader(dataset,batch_size,shuffle=shuffleFlag,num_workers=workers,pin_memory=True)
    
    return dataloader

#Function to make CSV files for the Confocal Dataset
#Inputs:
    #dataPath: Location of the data patches
    #outPath:  Location to save the csv files
    #nfolds:   Number of folds to split the data into
#Outputs:
    #None
#############################################################
def makeConfocalCsv(dataPath,outPath,nfolds=10,noiseCode='condition1'):
    subdirs = [f.name for f in os.scandir(os.path.join(dataPath,'clean')) if f.is_dir()]
    #First split the patients into training and test
        #Want to do kfold cross validation only on the training patients
    kf=sklearn.model_selection.KFold(n_splits=nfolds)
    splitIdx=0
    os.makedirs(outPath,exist_ok=True)
    if isinstance(noiseCode,int):
        noiseCode = 'condition%d'%noiseCode
    #Get the patient indexes for each of the folds
    for train, test in kf.split(subdirs):
        #Write the training part of this fold
        with open(os.path.join(outPath,'trainSet_split%d.csv'%splitIdx),'w') as f1:
            #Iterate through each patient
            for scanIdx in train:
                #Get a list of all the images for that patient
                cleanPaths = sorted(glob.glob(os.path.join(dataPath,'clean',subdirs[scanIdx],'*.npy')))
                #Iterate through each image
                for cp in cleanPaths:
                    patchName = os.path.basename(cp)
                    noisyPath = os.path.join(dataPath,noiseCode,'current',subdirs[scanIdx],patchName)
                    nextPath = os.path.join(dataPath,noiseCode,'next',subdirs[scanIdx],patchName)
                    prevPath = os.path.join(dataPath,noiseCode,'prev',subdirs[scanIdx],patchName)
                    #Write the image path to the csv file
                    f1.write("%s,%s,%s,%s\n"%(cp,prevPath,noisyPath,nextPath))
        #Repeat the above for this test split
        with open(os.path.join(outPath,'validSet_split%d.csv'%splitIdx),'w') as f2:
            for scanIdx in test:
                #Get a list of all the images for that patient
                cleanPaths = sorted(glob.glob(os.path.join(dataPath,'clean',subdirs[scanIdx],'*.npy')))
                #Iterate through each image
                for cp in cleanPaths:
                    patchName = os.path.basename(cp)
                    noisyPath = os.path.join(dataPath,noiseCode,'current',subdirs[scanIdx],patchName)
                    nextPath = os.path.join(dataPath,noiseCode,'next',subdirs[scanIdx],patchName)
                    prevPath = os.path.join(dataPath,noiseCode,'prev',subdirs[scanIdx],patchName)
                    #Write the image path to the csv file
                    f2.write("%s,%s,%s,%s\n"%(cp,prevPath,noisyPath,nextPath))
        with open(os.path.join(outPath,'testSet_split%d.csv'%splitIdx),'w') as f3:
            for scanIdx in test:
                #cleanFrames = sorted(glob.glob(os.path.join(dataPath,'..','fullFrames','clean',subdirs[scanIdx],'*.npy')))
                cleanFrames = sorted(glob.glob(os.path.join(dataPath,'clean',subdirs[scanIdx],'*.npy')))
                for cf in cleanFrames:
                    imName = os.path.basename(cf)
                    #noisyFrame = os.path.join(dataPath,'..','fullFrames',noiseCode,subdirs[scanIdx],imName)
                    noisyFrame = os.path.join(dataPath,noiseCode,'current',subdirs[scanIdx],imName)
                    f3.write("%s,%s\n"%(cf,noisyFrame))          
        splitIdx += 1
        #Print out how many patients are in each set
        print(len(train),len(test))
    return None

#Function to make CSV files for the OCT Dataset
#Inputs:
    #dataPath: Location of the data patches
    #outPath:  Location to save the csv files
    #nfolds:   Number of folds to split the data into
#Outputs:
    #None
#############################################################
def makeOCTCsv(dataPath,outPath,nfolds=10):
    subdirs = [f.name for f in os.scandir(os.path.join(dataPath,'current')) if f.is_dir()]
    #First split the patients into training and test
        #Want to do kfold cross validation only on the training patients
    kf=sklearn.model_selection.KFold(n_splits=nfolds)
    splitIdx=0
    os.makedirs(outPath,exist_ok=True)
    #Get the patient indexes for each of the folds
    for train, test in kf.split(subdirs):
        #Write the training part of this fold
        with open(os.path.join(outPath,'trainSet_split%d.csv'%splitIdx),'w') as f1:
            #Iterate through each patient
            for scanIdx in train:
                #Get a list of all the images for that patient
                currPaths = sorted(glob.glob(os.path.join(dataPath,'current',subdirs[scanIdx],'*.npy')))
                #Iterate through each image
                for cp in currPaths:
                    patchName = os.path.basename(cp)
                    nextPath = os.path.join(dataPath,'next',subdirs[scanIdx],patchName)
                    prevPath = os.path.join(dataPath,'prev',subdirs[scanIdx],patchName)
                    #Write the image path to the csv file
                    f1.write("%s,%s,%s\n"%(prevPath,cp,nextPath))
        #Repeat the above for this test split
        with open(os.path.join(outPath,'validSet_split%d.csv'%splitIdx),'w') as f2:
            for scanIdx in test:
                #Get a list of all the images for that patient
                currPaths = sorted(glob.glob(os.path.join(dataPath,'current',subdirs[scanIdx],'*.npy')))
                #Iterate through each image
                for cp in currPaths:
                    patchName = os.path.basename(cp)
                    nextPath = os.path.join(dataPath,'next',subdirs[scanIdx],patchName)
                    prevPath = os.path.join(dataPath,'prev',subdirs[scanIdx],patchName)
                    #Write the image path to the csv file
                    f2.write("%s,%s,%s\n"%(prevPath,cp,nextPath))
        with open(os.path.join(outPath,'testSet_split%d.csv'%splitIdx),'w') as f3:
            for scanIdx in test:
                #fullFrames = sorted(glob.glob(os.path.join(dataPath,'..','fullFrames',subdirs[scanIdx],'*.npy')))
                fullFrames = sorted(glob.glob(os.path.join(dataPath,subdirs[scanIdx],'*.npy')))
                for cf in fullFrames:
                    #imName = os.path.basename(cf)
                    f3.write("%s\n"%(cf))          
        splitIdx += 1
        #Print out how many patients are in each set
        print(len(train),len(test))
    return None

#Function to make CSV files for the CT Dataset
#Inputs:
    #dataPath: Location of the data patches
    #outPath:  Location to save the csv files
    #nfolds:   Number of folds to split the data into
#Outputs:
    #None
#############################################################
def makeCTCsv(dataPath,outPath,nfolds=10):
    subdirs = [f.name for f in os.scandir(os.path.join(dataPath,'clean')) if f.is_dir()]
    #First split the patients into training and test
        #Want to do kfold cross validation only on the training patients
    kf=sklearn.model_selection.KFold(n_splits=nfolds)
    splitIdx=0
    os.makedirs(outPath,exist_ok=True)
    #Get the patient indexes for each of the folds
    for train, test in kf.split(subdirs):
        #Write the training part of this fold
        with open(os.path.join(outPath,'trainSet_split%d.csv'%splitIdx),'w') as f1:
            #Iterate through each patient
            for scanIdx in train:
                #Get a list of all the images for that patient
                cleanPaths = sorted(glob.glob(os.path.join(dataPath,'clean',subdirs[scanIdx],'*.npy')))
                #Iterate through each image
                for cp in cleanPaths:
                    patchName = os.path.basename(cp)
                    noisyPath = os.path.join(dataPath,'current',subdirs[scanIdx],patchName)
                    nextPath = os.path.join(dataPath,'next',subdirs[scanIdx],patchName)
                    prevPath = os.path.join(dataPath,'prev',subdirs[scanIdx],patchName)
                    #Write the image paths to the csv file
                    f1.write("%s,%s,%s,%s\n"%(cp,prevPath,noisyPath,nextPath))
        #Repeat the above for this test split
        with open(os.path.join(outPath,'validSet_split%d.csv'%splitIdx),'w') as f2:
            for scanIdx in test:
                #Get a list of all the images for that patient
                cleanPaths = sorted(glob.glob(os.path.join(dataPath,'clean',subdirs[scanIdx],'*.npy')))
                #Iterate through each image
                for cp in cleanPaths:
                    patchName = os.path.basename(cp)
                    noisyPath = os.path.join(dataPath,'current',subdirs[scanIdx],patchName)
                    nextPath = os.path.join(dataPath,'next',subdirs[scanIdx],patchName)
                    prevPath = os.path.join(dataPath,'prev',subdirs[scanIdx],patchName)
                    #Write the image path to the csv file
                    f2.write("%s,%s,%s,%s\n"%(cp,prevPath,noisyPath,nextPath))
        with open(os.path.join(outPath,'testSet_split%d.csv'%splitIdx),'w') as f3:
            for scanIdx in test:
                #cleanFrames = sorted(glob.glob(os.path.join(dataPath,'..','fullFrames','clean',subdirs[scanIdx],'*.npy')))
                cleanFrames = sorted(glob.glob(os.path.join(dataPath,'clean',subdirs[scanIdx],'*.npy')))
                for cf in cleanFrames:
                    imName = os.path.basename(cf)
                    #noisyFrame = os.path.join(dataPath,'..','fullFrames','noisy',subdirs[scanIdx],imName)
                    noisyFrame = os.path.join(dataPath,'noisy',subdirs[scanIdx],imName)
                    f3.write("%s,%s\n"%(cf,noisyFrame))        
        splitIdx += 1
        #Print out how many patients are in each set
        print(len(train),len(test))
    return None

def makeRCMCsv(dataPath,outPath,nfolds=10):
    subdirs = [f.name for f in os.scandir(dataPath) if f.is_dir()]
    #First split the patients into training and test
        #Want to do kfold cross validation only on the training patients
    kf=sklearn.model_selection.KFold(n_splits=nfolds)
    splitIdx=0
    os.makedirs(outPath,exist_ok=True)
    #Get the patient indexes for each of the folds
    for train, test in kf.split(subdirs):
        #Write the training part of this fold
        with open(os.path.join(outPath,'trainSet_split%d.csv'%splitIdx),'w') as f1:
            #Iterate through each patient
            for scanIdx in train:
                #Get a list of all the images for that patient
                imPaths = sorted(glob.glob(os.path.join(dataPath,subdirs[scanIdx],'*.npy')))
                #Iterate through each image
                for cp in imPaths:
                    #Write the image path to the csv file
                    f1.write("%s\n"%(cp))
        #Repeat the above for this test split
        with open(os.path.join(outPath,'validSet_split%d.csv'%splitIdx),'w') as f2:
            for scanIdx in test:
                #Get a list of all the images for that patient
                imPaths = sorted(glob.glob(os.path.join(dataPath,subdirs[scanIdx],'*.npy')))
                #Iterate through each image
                for cp in imPaths:
                    #Write the image path to the csv file
                    f2.write("%s\n"%(cp))
        with open(os.path.join(outPath,'testSet_split%d.csv'%splitIdx),'w') as f3:
            for scanIdx in test:
                #fullFrames = sorted(glob.glob(os.path.join(dataPath,'..','fullNumpy',subdirs[scanIdx]+'.jpg')))
                fullFrames = sorted(glob.glob(os.path.join(dataPath,subdirs[scanIdx]+'.jpg')))
                for cf in fullFrames:
                    #imName = os.path.basename(cf)
                    f3.write("%s\n"%(cf))         
        splitIdx += 1
        #Print out how many patients are in each set
        print(len(train),len(test))
    return None

#Class to augment input data by flipping vertically or horizontally
#This is used as a "transform"
class flipAugment(object):
    def __call__(self,sample):       
        #Choose whether to flip an image or not
        flipHoriz = np.random.choice([True,False])
        flipVert = np.random.choice([True,False])
        if not flipHoriz and not flipVert:
            return sample
        #Flip horizontally
        flippedIms = []
        if flipHoriz:
            for z in range(len(sample)):
                flippedIms.append(np.fliplr(sample[z]).copy())
        #If there's no horizontal imaging you still need to make a copy
        else:
            for z in range(len(sample)):
                flippedIms.append(sample[z].copy())
        #Flip vertically
        if flipVert:
            for z in range(len(flippedIms)):
                flippedIms[z] = np.flipud(flippedIms[z]).copy()
            
        return flippedIms
    
#Transform to convert numpy objects into Tensors   
class ToTensor(object):
    def __call__(self, sample):
        tensorList = []
        for z in range(len(sample)):
            tensorList.append(T.functional.to_tensor(sample[z]))
        return tensorList

#Class to normalize tensors based on setMean and setStd
#Works on lists of tensors
class NormalizeTensors(object):
    def __init__(self, setMean, setStd):
        self.setMean= setMean
        self.setStd = setStd
        self.norm = T.Normalize(self.setMean,self.setStd)
    def __call__(self, sample):
        if self.setStd == 1 and self.setMean == 0:
            return sample
        else:
            normTensorList = []
            for z in range(len(sample)):
                normTensorList.append(self.norm(sample[z]))
            return normTensorList

#Reverses the normalization process
#Inputs:
#   dat: The data to unnormalize
#   mean: average value it was normalized with
#   std: standard deviation it was normalized with
#Outputs:
#   The un-normalized data
########################################   
def unNormalizeData(dat, mean, std):
    if mean == 0 and std == 1:
        return dat
    else:
        return dat*std + mean

#Normalizes the input to a mean of 0 and a std of 1
#Inputs:
#   dat: The data to normalize
#   mean: Average value of the training set
#   std: Standard deviation of the training set
#Outputs:
#   The normalized data
########################################    
def normalizeData(dat,mean,std):
    if mean==0 and std==1:
        return dat
    else:
        return (dat-mean) / std
    
#Function to get the mean and standard deviation of the training dataset
#Inputs:
#   dataloader: A dataloader object with the training set
#Outputs:
#   avg: The average of the training set
#   std: The average of each minibatch's standard deviation
##############################################################    
def getDataStats(dataloader):
    avg=0
    std=0
    totalIms = 0
    #Iterate through all data
    for batch in dataloader:
        image = batch[1] #Batch 1 always has a noisy image in it
        numIms = image.shape[0] #Number of images in the batch
        image = image.view(numIms,image.size(1),-1) #Compress the dimensions
        #Calculate stats
        avg += image.mean(2).sum(0)
        std += image.std(2).sum(0)
        totalIms += numIms
    #Average stats
    avg /= totalIms
    std /= totalIms
    return avg, std

#Helper function to plot and save the training progress
#Inputs:
    #logfile: full file path to the logfile
    #logdir:  Directory where plots will be saved
#Outputs:
    #None
#Makes two graphs one for train/validation loss and one for metrics (PSNR and SSIM)
#The latter won't be helpful if there is no ground truth image
################################################################
def plotLog(logfile,logdir):
    df = pd.read_csv(logfile,sep='\t')
    #vals = df.values
    
    fig,ax = plt.subplots(2,1,figsize=(9,9))
    ax[0].plot(df['Epoch'],df['TrainLoss'],label='Training')
    ax[1].plot(df['Epoch'],df['ValidLoss'],label='Validation')
    ax[0].set_ylabel('Loss/Image')
    ax[1].set_ylabel('Loss/Image')
    ax[1].set_xlabel('Epoch')
    ax[0].set_title('Training Loss')
    ax[1].set_title('Validation Loss')
    plt.tight_layout()
    plt.savefig(os.path.join(logdir,'lossOverTraining.png'))
    plt.close()
    
    fig,ax=plt.subplots(2,1,figsize=(9,9))
    ax[0].plot(df['Epoch'],df['PSNR'])
    ax[0].set_title('Peak SNR')
    ax[0].set_xlabel('Epoch')
    ax[0].set_ylabel('PSNR (dB)')
    ax[1].plot(df['Epoch'],df['SSIM'])
    ax[1].set_title('SSIM')
    ax[1].set_xlabel('Epoch')
    ax[1].set_ylabel('SSIM')
    plt.tight_layout()
    plt.savefig(os.path.join(logdir,'metricsOverTraining.png'))
    plt.close()

#Loads either Unet or Unet1D from models.py
#Inputs:
    #modelWtsPath: The pth file containing the model weights created by saving the state dict
    #modelType:    Either UNet or UNet1D
#Outputs:
    #model: The pre-trained pytorch model
#####################################################    
def loadModel(modelWtsPath,modelType='unet'):
    if modelType.lower() == 'unet':
        model = UNet(in_channels=1,out_channels=1)
    elif modelType.lower()=='unet1d':
        model = UNet1D(in_channels=1,out_channels=1)
    elif modelType.lower()=='neigh2neigh':
        model = utils.arch_unet.UNet(in_nc=1,out_nc=1,n_feature=48,)
    else:
        raise ValueError("Only 'UNet' and 'UNet1D' models coded")
    model.load_state_dict(torch.load(modelWtsPath))
    return model

#Code to save a grid of images to a file
#
#INPUTS:
#    batch:    Batch of images
#    saveDir:  Directory to save the file in
#    fileName: Name of the file to save including extension
#OUTPUTS:
#    imGrid: The grid image in case you want to use it for something else
###############################################################
def saveGrid(batch,saveDir,fileName):

    imGrid = U.make_grid(batch)
    #print('%s: dims: %d, size dim1: %d'%(fileName,batch.dim(),batch.shape[1]))
    if torch.max(batch) > 2:
        batch = batch / 255.0

    U.save_image(imGrid,os.path.join(saveDir,fileName))
    return imGrid

#A function for saving the hyperparameters during each run
#Inputs:
    #logdir: Path to the log directory
    #paramDict: Dictionary of hyperparameters to be written to a file
#Outputs:
    #None
#################################################################
def saveHyperParameters(logdir, paramDict):
    #These hyperparameters help determine which algorithm is being used
    dataType = paramDict['dataType']
    modelType = paramDict['modelType']
    singleIm = paramDict['singleImageTraining']
    nextIm = paramDict['nextImFlag']
    cleanTargets=paramDict['cleanTargets']
    noise2Void = paramDict['noise2VoidFlag']
    sampMult = paramDict['sampleRateMultiplier']
    #situation is the algorithm being used
    #It's appended to the file name and makes it 
    #easier to figure out what was being trained when you open a folder
    if modelType == 'unet1d':
        situation = 'noise2Nyquist_single'
    elif cleanTargets:
        situation = 'supervised'
    elif noise2Void:
        situation = 'noise2void'
    elif nextIm:
        if dataType == 'phantom':
            #Drill down to get the sampling rate for this run
            if sampMult == 2.:
                situation = 'noise2NyquistO2'
            elif sampMult == 3.:
                situation = 'noise2NyquistO3'
            elif sampMult == 4.:
                situation = 'noise2NyquistO4'
            elif sampMult == 0.5:
                situation = 'noise2Nyquistx2'
            else:
                situation = 'noise2Nyquist'
        else:
            situation = 'noise2Nyquist'
    else:
        situation = 'noise2noise'
    #Write all entries of paramDict to the file        
    with open(os.path.join(logdir,'Hyperparameters_%s.txt'%situation),'w') as f:
        for key in paramDict.keys():
            f.write("%s, %s\n"%(key,paramDict[key]))
    return None

#Function that updates the file 'trainingLog.txt' during training
#Inputs:
    #logdir: Directory where logs are stored
    #epoch:  The epoch we're working on
    #trainLoss: The value of the training loss at this epoch
    #valLoss:   The value of the validation loss at this epoch
    #psnr:      The validation PSNR
    #psnr_std:  The standard deviation of the PSNR
    #ssim:      The SSIM
    #ssim_std:  standard deviation of SSIM
    #nrmse:     The MSE of this epoch
    #nrmse_std  Standard deviation of MSE
#Outputs:
    #None
#It will write the file if it doesn't exist
#If the file does exist it appends to the end of it
######################################################
def updateLog(logdir,epoch,trainLoss,valLoss,psnr,psnr_std,ssim,ssim_std,nrmse,nrmse_std):
    #If the file doesn't exist create it and write the header
    if not os.path.exists(os.path.join(logdir,'trainingLog.txt')):
        with open(os.path.join(logdir,'trainingLog.txt'),'w') as f:
            f.write('Epoch\tTrainLoss\tValidLoss\tPSNR\tPSNR_std\tSSIM\tSSIM_std\tNRMSE\tNRMSE_std\n')
    #Otherwise append the data
    with open(os.path.join(logdir,'trainingLog.txt'),'a') as f:
        f.write('%d\t%.8f\t%.8f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\n'%(epoch,trainLoss,valLoss,psnr,psnr_std,ssim,ssim_std,nrmse,nrmse_std))
    return None

#Function that saves the results of testing the ML algorithm
#Inputs:
    #saveName: The full file path+name of the file you want to save
    #imNames:  A vector of all the image names
    #PSNRs:    A vector of all PSNR values
    #SSIMs:    A vector of all SSIM values
    #MSEs:     A vector of all MSE values
#Outputs:
    #None
#It does save a nice CSV file for you though
############################################################
def writeTestResults(saveName, imNames,PSNRs,SSIMs,MSEs):
    with open(saveName,'w') as f:
        f.write('ImageName, PSNR, SSIM, MSE\n')
        for i,p,s,m in zip(imNames,PSNRs,SSIMs,MSEs):
            f.write('%s,%.4f,%.4f,%.6f\n'%(i,p,s,m))
    return None

#Function that returns the slope and offset of a line that minimizes
#the mean squared error between the reconIm and the GTIm
#
#Inputs:
    #reconIm: reconstructed Image from some denoising algorithm
    #GTIm:    ground truth clean image
#Outputs:
    #alpha: The slope of the line a*x+b that minimizes MSE(reconIm, GTIm)
    #beta:  The y intercept of the line
##########################################################
def getLinearScale(reconIm,GTIm):
    m=np.size(reconIm)
    #Append a vector of 1s to the input
    X = np.append(reconIm.flatten()[:,None],np.ones((m,1)),axis=1)
    #Output
    y = GTIm.reshape(m,1)
    try:
        #Try to solve the linear system
        theta = np.dot(np.linalg.inv(np.dot(X.T, X)), np.dot(X.T, y))
    except:
        #If you can't then just don't scale the image
        theta = [[1],[0]]
    alpha = theta[0][0]
    beta = theta[1][0]
    return alpha,beta

#See  decimateMosaic() below
#Does essentially the same thing, but with 3 output args.
#Honestly, I'm not even sure I ever used this
def decimateImage(img, tileSz,overlap=0.5):
    stride = int(tileSz*(1-overlap))
    #assert(img.shape[0] % stride == 0 and img.shape[1] % stride == 0)
    numTilesY = int(img.shape[0]/stride)-1
    numTilesX = int(img.shape[1]/stride)-1
    subImg = np.zeros((numTilesY*numTilesX,tileSz,tileSz),dtype=img.dtype)
    k = 0
    for i in range(numTilesY):
        for j in range(numTilesX):
            subImg[k,:,:] = img[(i*stride):(i*stride)+tileSz,(j*stride):(j*stride)+tileSz]
            k+=1           
    return subImg, numTilesX, numTilesY

#Splits an image into patches of tileSz with an optional overlap
#Inputs:
    #img: 2D image or 3D image stack
    #tileSz: A single number that's the size of the tile in X,Y (i.e. 64 splits it into 64x64 pixel tiles)
    #overlap: Percent overlap between tiles
#Outputs:
    #subImg: A matrix of [H x W x (D) x Nx x Ny]. D is optional Depth, Nx is number of Rows of patches, Ny is number of columns of patches
    #
##Notes: This function isn't thoroughly tested for all values of tileSz and overlap
#I know it works for 64x64 tiles with an overlap of 0 or 0.5, but you should check it
#for other values. Also, the image may be truncated if it's not evenly divisible by tileSz*overlap
#See the OCT data for examples
######################################################33
def decimateMosaic(img, tileSz,overlap=0.5):
    assert overlap < 1 #overlap can't be greater than 100%
    #How far to shift the window for each patch
    stride = int(tileSz*(1-overlap))
    #Uncomment to make sure there's no cropping of image
    #assert(img.shape[0] % stride == 0 and img.shape[1] % stride == 0)
    #Number of tiles in each direction (Y is row, X is column)
    numTilesY = int(img.shape[0]/stride)-1
    numTilesX = int(img.shape[1]/stride)-1
    #Pre-allocate for the output
    #If the image is a stack the final size will have to be bigger
    if len(img.shape)==2:
        subImg = np.zeros((tileSz,tileSz,numTilesX,numTilesY),dtype=img.dtype)
    elif len(img.shape)==3:
        subImg = np.zeros((tileSz,tileSz,img.shape[2],numTilesX,numTilesY),dtype=img.dtype)
    else:
        raise ValueError("Only 2-D and 3-D images supported")
    #Iterate through each tile
    for i in range(numTilesY):
        for j in range(numTilesX):
            #Slice the image at the correct pixels to fill out sub-image
            if len(img.shape)==2:
                subImg[:,:,j,i] = img[(i*stride):(i*stride)+tileSz,(j*stride):(j*stride)+tileSz]
            elif len(img.shape)==3:
                subImg[:,:,:,j,i]=img[(i*stride):(i*stride)+tileSz,(j*stride):(j*stride)+tileSz,:]
    return subImg

#Function to take a large image, split it into patches and then process it
#
#Inputs:
    #fullImPath: Either the location of the image on the filesystem, or a matrix of the image
    #patchSize:  Size of the patch in pixels
    #overlapPx:  Size of the overlap between patches in pixels
    #model:      pytorch model to do the denoising
    #device:     Where does the pytorch model live (e.g. 'cpu', or 'cuda')
    #cropSz:     You can use this to crop the final image (not well tested)
#outputs:
    #imPatches: An Y x X x Nx x Ny matrix containing the processed image patches
#Note: Unlike decimateMosaic above, this only works with 2D images
###################################################################################
def despeckleMosaic(fullImPath,patchSize, overlapPx, model,device,cropSz=-1):
    #Load the image if necessary
    if isinstance(fullImPath,str):
        fullImg = np.array(Image.open(fullImPath))
    else:
        fullImg = fullImPath
    #Calculate the number of steps that need to be taken
    if overlapPx <= 0:
      numStepsX = int(np.floor(fullImg.shape[1]/patchSize))
      numStepsY = int(np.floor(fullImg.shape[0]/patchSize))
    else:
      numStepsX = int(np.floor((fullImg.shape[1]-patchSize)/(patchSize-overlapPx))+1)
      numStepsY = int(np.floor((fullImg.shape[0]-patchSize)/(patchSize-overlapPx))+1)
    #Crop the number of steps if you want 
    if cropSz > 0:
        numStepsX = int(cropSz/(patchSize-overlapPx))
        numStepsY = numStepsX
    #print('Image: %dx%d, Num inferences: %dx%d'%(fullImg.shape[1],fullImg.shape[0],numStepsX,numStepsY))
    #Preallocate for image patches
    imPatches = np.zeros((patchSize,patchSize,numStepsX,numStepsY),dtype='float32')
    startX = 0 #Current X pixel location
    with torch.no_grad():
        model.eval()
        for xStep in range(numStepsX): #Iterate through all the x patches
            startY = 0 #Current Y pixel location
            for yStep in range(numStepsY): #Iterate through all y patches
                #print("Working on step %d,%d of %d,%d"%(xStep+1,yStep+1,numStepsX,numStepsY))
                #Get the patch
                cropImg = fullImg[startY:startY+patchSize,
                                  startX:startX+patchSize]
                #Turn it into a tensor
                rawDat = Variable(torch.from_numpy(cropImg).float().to(device), requires_grad=False)
                #Processing for line2line which was removed from final paper
                if isinstance(model,UNet1D):
                    rawDat = torch.unsqueeze(rawDat,0)
                    #Get input of all the rows, and all teh columns
                    modelIn1 = torch.reshape(rawDat,(rawDat.shape[1],1,rawDat.shape[2]))
                    modelIn2 = torch.reshape(torch.transpose(rawDat,1,2),(rawDat.shape[1],1,rawDat.shape[2]))
                    #Convert to 0-1 scale
                    if cropImg.dtype == 'uint8':
                        modelIn1 =modelIn1/255.
                        modelIn2 = modelIn2/255.
                    #Run through the model twice
                    modelOut1 = model(modelIn1)
                    modelOut2 = model(modelIn2)
                    #Reassemble into 2D images
                    mo1 = torch.reshape(modelOut1,(1,rawDat.shape[1],rawDat.shape[2]))
                    mo2 = torch.transpose(torch.reshape(modelOut2,(1,rawDat.shape[1],rawDat.shape[2])),1,2)
                    #Average them
                    modelDespeck = (mo1+mo2)/2
                    #Make it 4D
                    modelDespeck = torch.unsqueeze(modelDespeck,0)
                #All other methods besides line2line use this simpler formulation
                else:
                    #Make 2D patch 4D
                    rawDat = torch.unsqueeze(torch.unsqueeze(rawDat,0),0)
                    #Make sure it's on a 0-1 scale before processing
                    if cropImg.dtype == 'uint8':
                        modelDespeck = model(rawDat/255)
                    else:
                        modelDespeck = model(rawDat)
                #Organize the image patches
                imPatches[:,:,xStep,yStep] = np.squeeze(np.squeeze(modelDespeck.cpu().numpy()))#*255.0
                #update Y pixel start location
                startY = startY+patchSize - overlapPx
            #update X pixel start location
            startX = startX + patchSize - overlapPx          
    return imPatches

# define normalized 2D gaussian
def gauss2d(x=0, y=0, mx=0, my=0, sx=1, sy=1):
    return  np.exp(-((x - mx)**2. / (2. * sx**2.) + (y - my)**2. / (2. * sy**2.)))

#Function to stick together a series of image patches with some overlap
#Inputs:
    #imPatches: a 4D matrix Y x X x Nx x Ny like from decimateMosaic or despeckkleMosaic
    #overlapPx: Overlap between patches in pixels
#Outputs:
    #scaleMosaic: All the patches put together into one nice image
#Note: This algorithm assigns a confidence to each patch that is highest at the center
#lowest at the edges. The weights are used to perform a weighted average of pixels in
#the final image
##########################################################
def stitchMosaic(imPatches,overlapPx):
    #Only stitch 1 2D image at a time. If you want more than that, call it in a loop
    assert len(imPatches.shape) == 4
    #Calculate final size of mosaic and allocate memory for the sum
    numXIms = imPatches.shape[2]
    numYIms = imPatches.shape[3]
    patchX = imPatches.shape[1]
    patchY = imPatches.shape[0]
    patchSize = patchX
    #Make a Gaussian to define the weights for the weighted average
    xCoord = np.arange(-patchX/2,patchX/2)
    yCoord = np.arange(-patchY/2,patchY/2)
    X,Y = np.meshgrid(xCoord,yCoord)
    overlapWeightMap = gauss2d(X,Y,sx=patchX/4,sy=patchY/4)
    #Calculate the final size of the mosaic
    if overlapPx <= 0:
        mosaic = np.zeros((numYIms*imPatches.shape[0],numXIms*imPatches.shape[1]),dtype='float32')
    else:
        mosaic = np.zeros(((numYIms-1)*(patchSize-overlapPx) + patchSize,
                           (numXIms-1)*(patchSize-overlapPx) + patchSize),dtype='float32')
    #mosFilled is an array that indicates which pixels have been filled in
    mosFilled = np.zeros(mosaic.shape,dtype='bool')
    #mosOverlapWeights keeps track of the weight at each pixel
    mosOverlapWeights = np.zeros(mosaic.shape)
    #thisImFilled keeps track of where the current image is located in the mosaic
    thisImFilled = np.zeros(mosaic.shape,dtype='bool')
    
    #Start by putting the first image in the uppler left hand corner
    mosaic[0:patchY,0:patchX] = overlapWeightMap*imPatches[:,:,0,0]
    mosFilled[0:patchY,0:patchX] = True
    mosOverlapWeights[0:patchY,0:patchX] = overlapWeightMap
    idx = 0

    #Iterate through all the images
    for x in range(numXIms):
        for y in range(numYIms):
            #Skip the first image b/c we just placed it above
            if x==0 and y==0:
                idx += 1
                continue
            #Open the next image
            thisIm = imPatches[:,:,x,y]
            #Figure out the starting location for this image
            if overlapPx <= 0:
                startX = x*patchSize
                startY = y*patchSize
            else:
                startX = x*(patchSize-overlapPx)
                startY = y*(patchSize-overlapPx)
            #Indicate the location of the image in the mosaic
            thisImFilled[startY:startY+patchSize,startX:startX+patchSize] = True
            #Add the weights of the current image to the location of the current image
            mosOverlapWeights[thisImFilled] += overlapWeightMap.flatten()
            #Calculate the weighted sum
            mosaic[thisImFilled] += overlapWeightMap.flatten()*thisIm.flatten()
            #Update the filled flag
            mosFilled[startY:startY+patchSize,startX:startX+patchSize] = True
            #Return thisImFilled to all zeros
            thisImFilled[startY:startY+patchSize,startX:startX+patchSize] = False
          
            idx+=1
            
    #If any areas haven't been filled in make them 1 so we can divide
    #They'll appear as 0s in the final image which should be obvious if they're misplaced        
    mosOverlapWeights[mosOverlapWeights == 0] = 1
    #Then divide the mosaic by the weights
    scaleMosaic = (mosaic / mosOverlapWeights)
    
    return scaleMosaic

if __name__=='__main__':
    #phanFile = './SheppLoganPhan.mat'
    #dl = getDataLoader(None,'phantom',phanFile=phanFile,nyquistSampling=4,sampMult=2)
    #setAvg,setStd = getDataStats(dl)
    #confocalPath = 'D:\\datasets\\Denoising_Planaria\\patches064'
    # confocalPath = '/home/matthew/Documents/datasets/Denoising_Planaria/patches064'
    # outPath = './dataSplits/confocal'
    # os.makedirs(outPath,exist_ok=True)

    # np.random.seed(2145)
    # makeConfocalCsv(confocalPath,outPath,nfolds=10,noiseCode='condition2')
    model = UNet(1,1)
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Trainable Params: %d"%pytorch_total_params)
    # csv = os.path.join(outPath,'trainSet_split0.csv')
    # dl = getDataLoader(csv,dataType='confocal')
    # dat = iter(dl).next()
    
    # fig,ax = plt.subplots(1,3)
    # ax[0].imshow(dat[0][2,0,:,:])
    # ax[1].imshow(dat[1][2,0,:,:])
    # ax[2].imshow(dat[2][2,0,:,:])
