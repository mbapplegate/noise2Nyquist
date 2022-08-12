# -*- coding: utf-8 -*-
"""
Created on Fri Jul 15 13:18:55 2022

@author: Matthew
"""

import utilsOOP as utils
import os

import glob

def makeConfocalCsv(dataPath,outPath,testVols,trainVols,nfolds=10,noiseCode='condition1'):
    subdirs = [f.name for f in os.scandir(os.path.join(dataPath,'clean')) if f.is_dir()]
    #First split the patients into training and test
        #Want to do kfold cross validation only on the training patients
   
    splitIdx=0
    os.makedirs(outPath,exist_ok=True)
    if isinstance(noiseCode,int):
        noiseCode = 'condition%d'%noiseCode
    #Get the patient indexes for each of the folds
    for i in range(10):
        test=testVols[i]
        train=trainVols[i]
    
        assert len(test)+len(train) == 16
        
        
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
if __name__ == '__main__':  
    confocalDataPath = 'D:\datasets\Denoising_Planaria\patches064'
    octDataPath = 'D:\datasets\OCT Denoise\Data\patches064'
    ctDataPath = 'D:\datasets\lowDoseCT\patches064'
    rcmDataPath = 'D:\datasets\RCM_mosaics\patches064'
    
    confocalSavePath = './dataSplits/confocal'
    octSavePath = './dataSplits/oct'
    ctSavePath = './dataSplits/ct'
    rcmSavePath = './dataSplits/rcm'
    
    nfolds=10
    
    confocalTest = [[8,11],[6,15],[2,5],[9,14],[7,13],[1,10],[12],[3],[0],[4]]
    confocalTrain = [[0,1,2,3,4,5,6,7,9,10,12,13,14,15],
                     [0,1,2,3,4,5,7,8,9,10,11,12,13,14],
                     [0,1,3,4,6,7,8,9,10,11,12,13,14,15],
                     [0,1,2,3,4,5,6,7,8,10,11,12,13,15],
                     [0,1,2,3,4,5,6,8,9,10,11,12,14,15],
                     [0,2,3,4,5,6,7,8,9,11,12,13,14,15],
                     [0,1,2,3,4,5,6,7,8,9,10,11,13,14,15],
                     [0,1,2,4,5,6,7,8,9,10,11,12,13,14,15],
                     [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15],
                     [0,1,2,3,5,6,7,8,9,10,11,12,13,14,15]]
    
    makeConfocalCsv(confocalDataPath,confocalSavePath,confocalTest,confocalTrain,nfolds=nfolds,noiseCode='condition2')
    utils.makeOCTCsv(octDataPath,octSavePath,nfolds=nfolds)
    utils.makeCTCsv(ctDataPath,ctSavePath,nfolds=nfolds)
    utils.makeRCMCsv(rcmDataPath,rcmSavePath,nfolds=nfolds)