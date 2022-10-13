# -*- coding: utf-8 -*-
"""
Created on Tue Aug 10 11:38:05 2021

@author: Matthew
"""

import numpy as np

#from matplotlib import pyplot as plt
import random
import os
import datetime
import torch
import sys

from skimage.metrics import peak_signal_noise_ratio as PSNR
from skimage.metrics import structural_similarity as SSIM

from utils.models import UNet, UNet1D
from utils import utilsOOP as utils
from PIL import Image
import argparse

#Function to train the neural network
#INPUTS:
#    dataloader:     Dataloader object with images
#    model:          Neural network model
#    optimizer:      Optimizer for SGD
#    criterion:      Loss criterion
#    epoch:          Current epoch
#    numEpochs:      Total epochs
#    device:         Pytorch device where the computations take place (eg 'cpu' or 'cuda')
#    randomOrder:    Flag that randomizes which is the target and which is the input
#    cleanImExists:  If there is a clean image, affects which batch list idx to use
#    cleanTargets:   If true, this is supervised algorithm
#    useSingleIm:    If true, either line2line or noise2void
#    noise2VoidFlag: If true, process with noise2void
#
#OUTPUTS:
#    epochLoss: The total loss per minibatch
#Note: Also updates the model weights
##############################################################
def train(dataloader, model, optimizer,criterion,run,epoch,numEpochs,device,
          randomOrder=True,cleanImExists=False,cleanTargets=False,useSingleIm=False,noise2VoidFlag=False):
    epoch_loss = 0  #Initialize loss for this epoch
    #Set model to training mode
    model.train()
    #idxRange defines the indicies of the noisy images in the data (batch)
    if cleanImExists:
        #If there's a clean image then that will always be at index 0
        idxRange = np.arange(1,3)
    else:
        #If there's no clean image, then the batch should only be 2 images long
        idxRange = np.arange(0,2)
    #Go through all the training images
    for i, batch in enumerate(dataloader):
        #We have to go through different setup depending on the inputs
        
        if useSingleIm: #Line2line or noise2void
            if not noise2VoidFlag: #We are definitely using line2line
                inIm = batch[idxRange[0]]  #The first noisy image is the one we'll use
                if randomOrder: #If we're randomizing the order of targets
                    #We need to shift the image to leverage the variation, so pick one direction
                    shiftDir = np.random.choice(['up','down','left','right'])
                    if shiftDir == 'up':
                        target=torch.cat((inIm[:,:,1::,:],inIm[:,:,-2:-1,:]),dim=2) #Target is image shifted up
                    elif shiftDir == 'down':
                        target=torch.cat((inIm[:,:,1:2,:],inIm[:,:,0:-1,:]),dim=2) #Target is image shifted down
                    elif shiftDir == 'right':
                        target=torch.cat((inIm[:,:,:,1:2],inIm[:,:,:,0:-1]),dim=3) #Target is image shifted right
                    elif shiftDir == 'left':
                        target=torch.cat((inIm[:,:,:,1::],inIm[:,:,:,-2:-1]),dim=3) #Target is image shifted left
                else: #We're not randomizing
                    #Otherwise, just pick the shift to the right
                    shiftDir = 'right'
                    target=torch.cat((inIm[:,:,:,1:2],inIm[:,:,:,0:-1]),dim=3)
                #Reshape inputs for use in 1 channel UNet1D
                #Turns the input and target into a 3D tensor [numIms, 1, numRows]
                #The target is the shifted version, so I'm comparing rows to adjacent rows
                if shiftDir == 'up' or shiftDir == 'down':
                    inIm = torch.reshape(inIm,(inIm.shape[0]*inIm.shape[2],1,inIm.shape[3]))          #Puts rows in dim 0
                    target=torch.reshape(target,(target.shape[0]*target.shape[2],1,target.shape[3]))  #What I want for up/down shift
                else:
                    inIm = torch.reshape(torch.transpose(inIm,2,3),(inIm.shape[0]*inIm.shape[2],1,inIm.shape[3]))
                    target = torch.reshape(torch.transpose(target,2,3),(target.shape[0]*target.shape[2],1,target.shape[3]))
            else: #Here we're using noise2void
                ratio = 0.9 #Replace 10% of pixels
                sampsPerImage = int(batch[0].shape[2]*batch[0].shape[3]*(1-ratio))
                windowSz = 5 #Replace pixels with neighbors in a 5x5 region
                #Target and input are the same image (idxRange[0] is the first noisy image)
                target = batch[idxRange[0]]
                inIm = torch.clone(target.detach())
                batchShape = target.shape
                #Need a mask so the loss only pays attention to the replaced pixels
                mask=np.ones(batchShape)
                for q in range(batchShape[0]):
                    #Choose indicies to replace
                    idy_msk = np.random.randint(0,batchShape[2],sampsPerImage)
                    idx_msk = np.random.randint(0,batchShape[3],sampsPerImage)
                    #Choose which neighbor to replace them with
                    idy_neigh = np.random.randint(-windowSz//2 + windowSz%2,windowSz//2 + windowSz%2,sampsPerImage)
                    idx_neigh = np.random.randint(-windowSz//2+windowSz%2,windowSz//2+windowSz%2,sampsPerImage)
                    #Replaced indicies
                    idy_msk_neigh = idy_msk + idy_neigh
                    idx_msk_neigh = idx_msk + idx_neigh
                    #Do some checks to make sure the values are within the image. Looks like it wraps around, which I'm not sure is right
                    #I'm not too worried since I'm dealing with patches that overlap
                    idy_msk_neigh = idy_msk_neigh + (idy_msk_neigh < 0) * batchShape[2] - (idy_msk_neigh >= batchShape[2]) * batchShape[2]
                    idx_msk_neigh = idx_msk_neigh + (idx_msk_neigh < 0) * batchShape[3] - (idx_msk_neigh >= batchShape[3]) * batchShape[3]
                    #Add dimensions to make it compatible with batchShape
                    id_msk = (q,0,idy_msk,idx_msk)
                    id_msk_neigh = (q,0,idy_msk_neigh,idx_msk_neigh)
                    #Replace pixels
                    inIm[id_msk] = target[id_msk_neigh]
                    #Update mask
                    mask[id_msk] = 0.0
                #Make a tensor out of mask
                maskTensor = torch.from_numpy(mask).float()
        #We're using multiple images
        else:
            #Clean targets mean supervised mode
            if cleanTargets:
                #Set the target to index 0
                target = batch[0]
                #Are we randomizing the order of the input image?
                if randomOrder:
                    #Get a noisy random input image
                    idx = np.random.choice(idxRange)
                    inIm = batch[idx]
                else:
                    #Otherwise index 1 is always the input image
                    inIm = batch[1]
            #No we're using noisy images as the target (noise2noise or noise2Nyquist)
            else:
                #Are we randomizing the order
                if randomOrder:
                    #Pick two random noisy images for target and input
                    idxChoice = np.random.choice(idxRange,size=2,replace=False)
                    target = batch[idxChoice[0]]
                    inIm = batch[idxChoice[1]]
                #We're not randomizing the order, so just pick the first two noisy images
                else:
                    target=batch[idxRange[0]]
                    inIm = batch[idxRange[1]]
                 
        #More efficient than using optimizer.zero_grad()
        #pytorch.org/tutorials/recipes/recipes/tuning_guide.html
        for param in model.parameters():
            param.grad = None
        #Run the model
        modelOut = model(inIm.to(device))
        #Have to use the mask if doing noise2void
        if noise2VoidFlag:
            maskTensor=maskTensor.to(device)
            loss = criterion(modelOut*(1-maskTensor),target.to(device)*(1-maskTensor))
        else:
            loss = criterion(modelOut,target.to(device))
        epoch_loss += loss.item() #Track loss
        loss.backward() #Backprop
        optimizer.step() #SGD step
        #Monitor training
        sys.stdout.write('\r[%d/%d][%d/%d] Loss: %.4f' 
                             % (epoch, numEpochs, i, len(dataloader), loss.item()))
    #print("\n ===> Epoch {} Complete: Avg. Loss: {:.4f}".format(epoch, epoch_loss / (i+1)))    
  
    #Average loss per batch
    return epoch_loss / (i+1)

#Function to run model on validation set
#
#INPUTS:
#    dataloader:     Dataloader containing validation images
#    model:          Pytorch neural network
#    criterion:      Criterion for loss
#    epoch:          Current epoch
#    device:         Device where computation happens
#    sampMean:       Training set mean
#    sampStd:        Training set standard deviation
#    dataType:       What data are we training on ['oct','ct','phantom','rcm','confocal']
#    usingSingleIm:  Flag to indicate single image denoising (line2line or noise2void)
#    cleanImExists:  Does a clean image exist? Affects batch index
#    noise2VoidFlag: Are we using noise2void algorithm
#
#OUTPUTS:
#    valLoss: Validation loss per minibatch
#    psnr:     Average validation PSNR 
#    psnr_std: Standard deviation of validation PSNR
#    ssim:     Average validation SSIM 
#    ssim_std: Standard deviation of validation SSIM 
#    mse:      Average validation MSE
#    mse_std:  Standard deviation of validation MSE
########################################################################

def validate(dataloader,model,criterion,epoch,device,
             sampMean=0,sampStd=1,dataType='oct',useSingleIm=False,cleanImExists=False,noise2VoidFlag=False):
    val_loss=0 #keep track of validation loss
    totalIms = 0
    epochSSIM =[]
    epochPSNR=[]
    epochNRMSE = []
    #Don't keep track of gradients
    with torch.no_grad():
        #Model in evaluation mode
        model.eval()
        #Iterate through all data
        for i, batch in enumerate(dataloader):
            #For validation, the target will always be at index 0 and input at index 1
            target=batch[0].to(device)
            inIm= batch[1].to(device)
            #If we're using line2line method
            if useSingleIm and not noise2VoidFlag:
                #Reshape the 2D image into a bunch of batches of single lines
                inIm1 = torch.reshape(inIm,(inIm.shape[0]*inIm.shape[2],1,inIm.shape[3]))
                inIm2 = torch.reshape(torch.transpose(inIm,2,3),(inIm.shape[0]*inIm.shape[2],1,inIm.shape[3]))
                #Make Denoised image batch by running model on lines, again on columns and averaging them
                modelOut1 = model(inIm1)
                modelOut2 = model(inIm2)
                mo1 = torch.reshape(modelOut1,batch[0].shape)
                mo2 = torch.transpose(torch.reshape(modelOut2,batch[0].shape),2,3)
                modelOut = (mo1 + mo2) / 2
            else:
                modelOut = model(inIm)
        
            #Go through each image in the batch
            for j in range(len(modelOut)):
                #Get the image onto the cpu
                thisTarget = utils.unNormalizeData(target[j,0,:,:].cpu().numpy(),sampMean,sampStd)
                thisResult = utils.unNormalizeData(modelOut[j,0,:,:].cpu().numpy(),sampMean,sampStd)
                #Do affinity scaling if there's a clean image to compare with
                if cleanImExists:
                    alpha,beta = utils.getLinearScale(thisResult,thisTarget)
                    thisResult = thisResult*alpha + beta
                #If the data are floating point clip between 0 and 1 (prevents error in PSNR calculation)
                if isinstance(thisResult[0,0],np.floating):
                    thisResult = np.clip(thisResult,0,1)
                    thisTarget = np.clip(thisTarget,0,1)
                #Calculate stats
                thisPSNR = PSNR(thisTarget,thisResult)
                thisSSIM = SSIM(thisTarget,thisResult)
                thisNRMSE = np.sqrt(np.sum((thisTarget-thisResult)**2)/np.size(thisTarget))
                epochSSIM.append(thisSSIM)
                epochPSNR.append(thisPSNR)
                epochNRMSE.append(thisNRMSE)
            
                totalIms+=1
            #Calculate loss
            loss = criterion(modelOut,target)
            val_loss += loss.item()
        #Report loss    
        print(" ===> Epoch %d PSNR: %.2f SSIM: %.3f" %(epoch,np.mean(epochPSNR),np.mean(epochSSIM)),flush=True)
      
    return val_loss/(i+1), np.mean(epochPSNR),np.std(epochPSNR), np.mean(epochSSIM),np.std(epochSSIM),np.mean(epochNRMSE),np.std(epochNRMSE)

#Function to test the model after training is done
#Inputs:
#   dataloader:    Pytorch dataloader
#   model:         ML Model
#   device:        Device where computations are done
#   logdir:        Location of log directory
#   dataType:      What type of data is this ['confocal','ct','oct','rcm']
#   sampMean:      Mean of training set
#   sampStd:       Standard deviation of training set
#   cleanImExists: Is there a clean image to compare with?
#
#Outputs:
#   testPSNR: List of PSNR values one for each image
#   testSSIM: List of SSIM values one for each image
#   testMSE:  List of MSE values  one for each image
#   testIms:  Names of all the images
########################################
def test(dataloader,model,device,logdir,dataType='oct',sampMean=0,sampStd=1,cleanImExists=False):
    testSSIM = []
    testPSNR = []
    testNRMSE = []
    testNames = []
    print("Testing")
    #Index of the noisy image
    if cleanImExists:
        inIdx=1
    else:
        inIdx = 0
    with torch.no_grad():
        #Model in evaluation mode
        model.eval()
        #Iterate through all data
        for i, batch in enumerate(dataloader):
            print("Working on image %d of %d"%(i+1,len(dataloader)))
            #For validation, the target will always be at index 0 and input at index 1
            if dataType == 'phantom':
                imNameNpy = 'phantomFrame%03d.npy'%i
            else:
                imNameNpy = batch[-1][0] #Test datasets have names as the last index (see utilsOOP.py)
            #Do some manipulations of the image name to make it into a png file    
            imName = os.path.splitext(imNameNpy)[0]
            testNames.append(imName)
            saveName = os.path.join(logdir,imName+'_AI.png')
            #Input im (often a large frame)
            inIm= batch[inIdx]
            inImNP = inIm.cpu().numpy()
            #Split into patches and denoise (see utilsOOP.py for details)
            patches = utils.despeckleMosaic(inImNP[0,0,:,:],64,32,model,device)
            #Data coming from the dataloader are normalized
            #So need to unnormalize the data
            patchesUN =utils.unNormalizeData(patches,sampMean,sampStd)
            #Stitch back together. This is the denoised image
            mosaic=utils.stitchMosaic(patchesUN,32)
            
            if cleanImExists:
                #Target is the clean image (needs to be unnormalized)
                target = utils.unNormalizeData(batch[0][0,0,:,:].cpu().numpy(),sampMean,sampStd)
                #Do affinity scaling and convert to 8-bit
                alpha,beta = utils.getLinearScale(mosaic,target)
                mosaic = mosaic*alpha+beta
                target8bit = np.clip(np.round(target*255),0,255).astype('uint8')
                mosaic8bit = np.clip(np.round(mosaic*255),0,255).astype('uint8')
                #Do metric calcs
                testPSNR.append(PSNR(target8bit,mosaic8bit))
                testSSIM.append(SSIM(target8bit,mosaic8bit))
                testNRMSE.append(getMSE(target8bit,mosaic8bit))
            else: #if there isn't a clean image, just save the denoised image so NIQI can be calculated
                mosaic8bit = np.clip(np.round(mosaic*255),0,255).astype('uint8')
                testPSNR.append(0)
                testSSIM.append(0)
            #Save denoised images
            mosaicIm = Image.fromarray(mosaic8bit)
            mosaicIm.save(saveName)
    return testPSNR,testSSIM,testNRMSE,testNames
           
#Helper function to save a batch of images for use during training to monitor progress
#Inputs:
#   dataloader:       Pytorch dataloader object
#   dataType:         What type of data is this?
#   model:            Denoising model 
#   epoch:            Epoch number
#   batches_to_viz:   How many minibatches to visualize
#   logdir:           Log directory
#   device:           Where are computations taking place 'cpu' or 'cuda'
#   cleanImageExists: Is there a clean image?
#   sampMean:         Mean value of training set
#   sampStd:          Standard deviation of training set
#   useSingleIm:      Are we doing noise2void or line2line?
#   noise2voidFlag:   Are we doing noise2void?
#
#Outputs:
#   None
#Note: Saves grid of original and denoised images to logdir
###########################################################################
def saveImages(dataloader,dataType,model,epoch,batches_to_viz,logdir,device,
               cleanImageExists=False,sampMean=0,sampStd=1,useSingleIm=False,noise2VoidFlag=False):

    saveDir = os.path.join(logdir,'Epoch%03d'%epoch)
    os.makedirs(saveDir)
    #Make dataloader iterable
    dataIter = iter(dataloader)
    i=0
    with torch.no_grad():
        model.eval()
        #Get the number of batches for visualization
        while i < batches_to_viz:
            dat = dataIter.next() 
            #Iterate through each image in the batch
            for z in range(len(dat)):
                #Save the input
                #Save grid of input data
                _=utils.saveGrid(utils.unNormalizeData(dat[z],sampMean,sampStd),saveDir,'batch%d_image%d.png'%(i,z))
                #Only save the clean image once since it will always be the same (also don't process the clean image)
                if cleanImageExists and z==0:
                    continue
                #Get the noisy image and the next image
                thisDat = dat[z]
                #If we're doing line2line
                if useSingleIm and not noise2VoidFlag:
                   #Reshape into rows and columns
                   thisDat1 = torch.reshape(dat[z],(dat[z].shape[0]*dat[z].shape[2],1,dat[z].shape[3]))
                   thisDat2 = torch.reshape(torch.transpose(dat[z],2,3),(dat[z].shape[0]*dat[z].shape[2],1,dat[z].shape[3]))
                   #Run the model on rows and columns
                   modelOut1 = model(thisDat1.to(device))
                   modelOut2 = model(thisDat2.to(device))
                   mo1 = torch.reshape(modelOut1,dat[z].shape)
                   mo2 = torch.transpose(torch.reshape(modelOut2,dat[z].shape),2,3)
                   #Average two outputs
                   modelOut = (mo1 + mo2) / 2
                else:
                    modelOut = model(thisDat.to(device))
                #Have to unnormalize data or else shit gets weird
                thisResult = np.clip(utils.unNormalizeData(modelOut.cpu(),sampMean,sampStd),0,1)
                if cleanImageExists:
                    for im in range(thisResult.shape[0]):
                        unDat = utils.unNormalizeData(dat[0][im,0,:,:],sampMean,sampStd)
                        alpha,beta = utils.getLinearScale(np.array(thisResult[im,0,:,:]),np.array(unDat))
                        thisResult[im,0,:,:] = thisResult[im,0,:,:]*alpha+beta
                #Save the results
                _=utils.saveGrid(thisResult,saveDir,'batch%d_Processed%d.png'%(i,z))
     
            i+=1 #Update loop index variable
    return None

#Calculates MSE between inputs -- should be the same size
def getMSE(target, tester):
    return np.sum((target-tester)**2)/np.size(target)

#Main function that does all the stuff -- See if __name__=='__main__' block below   
def main(dataType,dataPath,csvDir,saveDir,remakeCSVs=False,cleanTargets=False,nextImFlag=False,randomOrder=True,
         singleImageTraining=False,noise2VoidFlag=False,loss_fn='l1',batch_size=4,initLR=0.001,numTrainingIms=1024,
         numValidIms=0,numTestIms=0,numEpochs=150,splitsToTrain=[0],nfolds=7,calcStats=True,phantomNoiseLevel=45,schedulerGamma=0.97,
         TVwt=0,noiseCondition=1,nyquistSampling=4,sampMult=1,trainPhantomFile='./SheppLoganPhan.mat',
         valPhantomFile='./YuYeWangPhan.mat',testOnly=False,noiseType='additive'):

    #Define device
    if torch.cuda.is_available():
        device='cuda:0'
        torch.backends.cudnn.benchmark=True
    else:
        device='cpu'
    
    #Only some data types have clean images
    if dataType == 'phantom' or dataType == 'confocal' or dataType=='ct':
        cleanImageExists = True
    else:
        cleanImageExists = False
   
    if singleImageTraining and not noise2VoidFlag:
        modelType = 'neigh2neigh'
    else:
        modelType='unet'   
   
    seed = 2145 #Random seed
  
    vizInterval = 50 #Visualize the results every X epochs
    benchmarkInterval = 100 #Save model weights every x epochs
    batchesToViz = 1 #Number of batches to visualize
    #Calculate final learning rate
    finalLR = initLR * schedulerGamma**(numEpochs)
    #Set up the hyperparameter dictionary so it can be written to a file
    paramDict = {'dataPath':dataPath,'csvLocation':csvDir,'saveLocation':saveDir,'remakeCSVFlag':remakeCSVs,'BatchSize':batch_size,
                 'initial_LR':initLR, 'schedularGamma':schedulerGamma,'finalLR':'%.5f'%finalLR, 'numEpochs':numEpochs,'numTrainingIms':numTrainingIms,
                 'numValidationIms':numValidIms,'numTestIms':numTestIms,'randomSeed':seed,
                 'splitsToTrain':splitsToTrain,'numFolds':nfolds,'vizInterval':vizInterval,'batchesToViz':batchesToViz,'cleanImageExists':cleanImageExists,
                 'BenchmarkInterval':benchmarkInterval,'lossFn':loss_fn,'nyquistSampling':nyquistSampling,'sampleRateMultiplier':sampMult,
                 'RandomTrainingOrder':randomOrder,'dataType':dataType,'cleanTargets':cleanTargets,'nextImFlag':nextImFlag,'phantomNoiseLevel':phantomNoiseLevel,
                 'modelType':modelType,'singleImageTraining':singleImageTraining,'TotalVariationWeight':TVwt,'noise2VoidFlag':noise2VoidFlag,'phantomNoiseType':noiseType}
    
    #Confocal data has 3 noise conditions
    if dataType.lower()=='confocal':
        paramDict['noiseCondition']=noiseCondition
    
    #Do some basic sanity checks. Prevents running long training cycle on garbage combo of inputs
    #Obviously not ideal, but this was built up over time so some of the options don't make the most sense
    if not singleImageTraining and modelType.lower() == 'unet1d':
        raise ValueError('UNet1d model without singleImageTraining won\'t work')
    #if singleImageTraining and modelType.lower() != 'unet1d' and not noise2VoidFlag:
    #     raise ValueError('UNet with single image training should use the noise2Void flag')
    if singleImageTraining and cleanTargets:
        raise ValueError('Single Image with Clean Targets Doesn\'t make sense')
    if not cleanImageExists and cleanTargets:
        raise ValueError('No clean images to use as targets')
    if dataType.lower() =='oct' and cleanTargets:
        raise ValueError('I don\'t have the clean images for the OCT dataset :-(')
    if dataType.lower() =='rcm' and not singleImageTraining:
        raise ValueError('I don\'t have image stacks for RCM :-(')
    if noise2VoidFlag and modelType.lower() == 'unet1d':
        raise ValueError('Noise 2 Void works on a 2D image, and unet1d is for 1d lines')
    # if testOnly and remakeCSVs:
    #     raise ValueError('It\'s not a good idea to remake the CSV if you\'re just testing a model')
    #######################################################333
    #Set random seeds for reproducibility
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    #Run defines the file where everything is saved
    run = datetime.datetime.now().strftime("%Y-%m-%d--%H-%M-%S")
    #Loop through for each datasplit to train
    for splitNum in splitsToTrain:
        print("Working on split number: %d"%splitNum,flush=True)
        #Set up logging
        if not testOnly:
            logdir = os.path.join(saveDir,'runs',run,'%02d'%splitNum)
            os.makedirs(logdir)
            weights_path = os.path.join(logdir,'modelWeights_best.pth')
       
        #You can remake the CSVs, but you probably shouldn't
        if dataType=='confocal':
            if remakeCSVs:
                utils.makeConfocalCsv(dataPath,csvDir,nfolds=nfolds,noiseCode=noiseCondition)
        elif dataType=='oct':
            if remakeCSVs:
                utils.makeOCTCsv(dataPath,csvDir,nfolds=nfolds)
        elif dataType=='ultrasound':
            if remakeCSVs:
                utils.makeUltrasoundCsv(dataPath,csvDir,nfolds=nfolds)
        elif dataType == 'ct':
            if remakeCSVs:
                utils.makeCTCsv(dataPath,csvDir,nfolds=nfolds)
        elif dataType=='rcm':
            if remakeCSVs:
                utils.makeRCMCsv(dataPath,csvDir,nfolds=nfolds)
        #Define the CSV files for this split
        if csvDir is not None:
            trainPathOrig = os.path.join(csvDir,'trainSet_split%d.csv'%splitNum)
            valPathOrig = os.path.join(csvDir,'validSet_split%d.csv'%splitNum)   
            testPathOrig = os.path.join(csvDir,'testSet_split%d.csv'%splitNum)
        #Datasets that don't have csv files don't need this
        else:
            trainPathOrig = None
            valPathOrig = None
            testPathOrig = None
                
          
        #Get the dataloaders
        if calcStats:
            print('Calculating dataset statistics...',flush=True)
            trainLoader=utils.getDataLoader(trainPathOrig,dataType,numImgs=numTrainingIms,batch_size=batch_size,setMean=0,
                                            setStd=1,singleImageTrain=singleImageTraining,shuffleFlag=True,phanFile=trainPhantomFile,
                                            nyquistSampling=nyquistSampling,sampMult=sampMult,nextImFlag=nextImFlag,
                                            noiseStd=phantomNoiseLevel,workers=8,testFlag=False,noiseType='additive')
            sampleMean,sampleStd = utils.getDataStats(trainLoader)
            sampleMean = sampleMean[0].numpy()
            sampleStd = sampleStd[0].numpy()
        else:
            sampleMean = 0
            sampleStd=1
        #Add calculated mean and std to hyperparams    
        paramDict['sampleMean']=sampleMean
        paramDict['sampleStd'] = sampleStd
        testLoader = utils.getDataLoader(testPathOrig,dataType,numImgs=numTestIms,batch_size=1,setMean=sampleMean,
                                        setStd=sampleStd,singleImageTrain=singleImageTraining,shuffleFlag=False,phanFile=valPhantomFile,
                                        nyquistSampling=nyquistSampling,sampMult=sampMult,nextImFlag=nextImFlag,
                                        noiseStd=phantomNoiseLevel,workers=8,testFlag=True,noiseType=noiseType)
        
        if not testOnly:
            #This is not ideal because it overwrites this every split
            #Not a problem except for sampleMean and sampleStd
            #!!!IF YOU WANT TO TEST ON EARLIER SPLITS MAKE SURE TO RECALCULATE SAMPLE STATS!!!!
            utils.saveHyperParameters(os.path.join(logdir,'..'),paramDict)    
            
            trainLoader=utils.getDataLoader(trainPathOrig,dataType,numImgs=numTrainingIms,batch_size=batch_size,setMean=sampleMean,
                                            setStd=sampleStd,singleImageTrain=singleImageTraining,shuffleFlag=True,phanFile=trainPhantomFile,
                                            nyquistSampling=nyquistSampling,sampMult=sampMult,nextImFlag=nextImFlag,
                                            noiseStd=phantomNoiseLevel,workers=8,testFlag=False,noiseType=noiseType)
            
            valLoader = utils.getDataLoader(valPathOrig,dataType,numImgs=numValidIms,batch_size=batch_size,setMean=sampleMean,
                                            setStd=sampleStd,singleImageTrain=singleImageTraining,shuffleFlag=True,phanFile=valPhantomFile,
                                            nyquistSampling=nyquistSampling,sampMult=sampMult,nextImFlag=nextImFlag,
                                            noiseStd=phantomNoiseLevel,workers=8,testFlag=False,noiseType=noiseType)
        
            #Start the model
            if modelType.lower()=='unet':
                model = UNet(in_channels=1,out_channels=1)
            elif modelType.lower()=='unet1d':
                model = UNet1D(in_channels=1,out_channels=1)
            else:
                raise ValueError('Only UNET and UNet1D implemented')
        
            model.to(device)
            
            #There is an option to add a total variation term to the loss if you want smoother output
            if TVwt == 0:
                if loss_fn=='l1':
                    criterion = torch.nn.L1Loss()
                elif loss_fn=='l2':
                    criterion = torch.nn.MSELoss()
                else:
                    raise ValueError('Loss type not supported')
            else:
                criterion = utils.lossWithTV(TVwt, loss_fn)
            #Set up optimizer and scheduler
            optimizer = torch.optim.Adam(model.parameters(), lr=initLR)
            scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=schedulerGamma)
            bestValLoss = 1e4
            
            ##################################################################
            ##################################################################
            #Training Loop
            ##################################################################
            ##################################################################
            for epoch in range(1,numEpochs+1):
                #Train
                thisTrainLoss=train(trainLoader,model,optimizer,criterion,run,epoch,numEpochs,device,
                                    randomOrder=randomOrder,cleanImExists=cleanImageExists,cleanTargets=cleanTargets,
                                    useSingleIm=singleImageTraining,noise2VoidFlag=noise2VoidFlag)
                #Validate
                thisValidLoss,valPSNR,valPSNR_std,valSSIM,valSSIM_std,valNRMSE,valNRMSE_std=validate(valLoader,model,criterion,epoch,device,
                                                                                                     sampMean=sampleMean,sampStd=sampleStd,
                                                                                                     dataType=dataType,useSingleIm=singleImageTraining,
                                                                                                     cleanImExists=cleanImageExists,noise2VoidFlag=noise2VoidFlag)
                #Update learning rate
                scheduler.step()
                #pbar.set_description('Loss: %.3f, PSNR: %.2f, SSIM: %.3f'%(thisTrainLoss, valPSNR, valSSIM))
                utils.updateLog(logdir,epoch,thisTrainLoss,thisValidLoss,valPSNR,valPSNR_std,valSSIM,valSSIM_std,valNRMSE,valNRMSE_std)
                #Update all the logging every 10 epochs
                if epoch % 10 == 0:
                    utils.plotLog(os.path.join(logdir,'trainingLog.txt'),logdir)
                #Save the model weights if this is the best model so far
                if thisValidLoss < bestValLoss:
                    bestValLoss = thisValidLoss
                    torch.save(model.state_dict(), weights_path)
                #Save the model weights at this epoch
                if epoch % benchmarkInterval == 0:
                    torch.save(model.state_dict(), os.path.join(logdir,'modelWeights_Epoch%03d.pth'%epoch))
                #Save images every vizInterval epochs
                if epoch % vizInterval == 0 or epoch == 1:
                    saveImages(valLoader,dataType,model,epoch,batchesToViz,logdir,device,
                               cleanImageExists=cleanImageExists,sampMean=sampleMean,sampStd=sampleStd,
                               useSingleIm=singleImageTraining,noise2VoidFlag=noise2VoidFlag)
                #Save the latest model
                torch.save(model.state_dict(), os.path.join(logdir,'modelWeights_last.pth'))
            print("Training Complete")
            #Define test directories
            testDirLast = os.path.join(logdir,'testImages','last')
            testDirBest = os.path.join(logdir,'testImages','best')
            os.makedirs(testDirLast)
            os.makedirs(testDirBest)
        #If we're only testing, we have to load the trained model
        if testOnly:
            testDirLast = os.path.join(saveDir,'%02d'%splitNum,'testImages','last')
            testDirBest = os.path.join(saveDir,'%02d'%splitNum,'testImages','best')
            os.makedirs(testDirLast,exist_ok=True)
            os.makedirs(testDirBest,exist_ok=True)
            model = utils.loadModel(os.path.join(saveDir,'%02d'%splitNum,'modelWeights_last.pth'),modelType=modelType)
            model.to(device)
        #Do the actual testing for the last model    
        testPSNR_last, testSSIM_last, testNRMSE_last,testNames_last = test(testLoader,model,device,testDirLast,dataType=dataType,sampMean=sampleMean,sampStd=sampleStd,cleanImExists=cleanImageExists)
        #if testOnly:
        #    bestModel = utils.loadModel(os.path.join(saveDir,'%02d'%splitNum,'modelWeights_best.pth'),modelType=modelType)
        #else:
        #    bestModel=utils.loadModel(os.path.join(logdir,'modelWeights_best.pth'),modelType=modelType)
        #bestModel.to(device)
        #Do the testing for the best model
        #testPSNR_best, testSSIM_best, testNRMSE_best,testNames_best = test(testLoader,bestModel,device,testDirBest,dataType=dataType,sampMean=sampleMean,sampStd=sampleStd,cleanImExists=cleanImageExists)
        if testOnly:
            utils.writeTestResults(os.path.join(saveDir,'%02d'%splitNum,'testResults_last.csv'),testNames_last,testPSNR_last,testSSIM_last,testNRMSE_last)
            #utils.writeTestResults(os.path.join(saveDir,'%02d'%splitNum,'testResults_best.csv'),testNames_best,testPSNR_best,testSSIM_best,testNRMSE_best)
        else:
            utils.writeTestResults(os.path.join(logdir,'testResults_last.csv'),testNames_last,testPSNR_last,testSSIM_last,testNRMSE_last)
          #  utils.writeTestResults(os.path.join(logdir,'testResults_best.csv'),testNames_best,testPSNR_best,testSSIM_best,testNRMSE_best)
if __name__ == '__main__':
    
    #Command line arguments that will let you flip all the different switches from the CL
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataType",help="Type of data to process: 'phantom','confocal','oct','ct','rcm'",default='oct')
    parser.add_argument("--dataPath",help="Path to the data set",default='./')
    parser.add_argument("--csvDir",help="Location where the CSV files defining the dataset are located",default='./dataSplits')
    parser.add_argument("--saveDir",help="Where do you want to save the output",default='./')
    parser.add_argument("--remakeCSV",help="Add this option to remake the CSV files",action="store_true")
    parser.add_argument("--cleanTargets",help="Add this option to use the clean images as targets",action="store_true")
    parser.add_argument("--nextImFlag",help="Add this option to use the next (or previous) image in the stack as the target",action="store_true")
    parser.add_argument("--randomOrder",help="Add this option to randomize whether the current frame can be used as either input or target",action="store_true")
    parser.add_argument("--singleImage",help="Add this option to train using only a single noisy image",action="store_true")
    parser.add_argument("--noise2VoidFlag",help="Add this option to train using the noise2void method",action="store_true")
    parser.add_argument("--loss_fn",help="Use this to choose 'l1' or 'l2' loss",choices=['l1','l2'],default='l1')
    parser.add_argument("--batch_size",help="Mini batch size",type=int,default=4)
    parser.add_argument("--init_lr",help="Initial learning rate for model",type=float,default=0.001)
    parser.add_argument("--numTrainingIms",help="Number of images to use in training (0 for all)",default=0,type=int)
    parser.add_argument("--numValidIms",help="Number of images to use for validation (0 for all)",default=0,type=int)
    parser.add_argument("--numTestIms",help="Number of images to use in testing (0 for all)",default=0,type=int)
    parser.add_argument("--numEpochs",help="Number of epochs to train for",default=100,type=int)
    parser.add_argument("--startSplit",help="Split to start training with",default=0,type=int)
    parser.add_argument("--splitsToTrain",help="Number of splits to train from 1-N (inclusive)",type=int,default=1)
    parser.add_argument("--nfolds",help="Number of folds to split the dataset into",type=int,default=7)
    parser.add_argument("--calcStats",help="Add this argument to calculate the mean and standard deviation of the dataset to use in normalization. Otherwise the mean will assumed to be 0 and the std will be 1",action="store_true")
    parser.add_argument("--phantomNoiseLevel",help="Standard deviation of the noise to add to the phantom dataset",type=int,default=45)
    parser.add_argument("--schedulerGamma",help="LR is scheduled with an exponential schedular, this controls the rate",type=float,default=0.97)
    parser.add_argument("--TVwt",help="Total Variation Penalty weight to use during training",type=float,default=0)
    parser.add_argument("--noiseCondition",help="Defines which noise level to use for the confocal measurement",choices=[1,2,3],type=int,default=1)
    parser.add_argument("--nyquistSampling",help="Defines the Nyquist rate for the phantom in pixels",type=int,default=4)
    parser.add_argument("--sampMult",help="Change the actual sampling rate to Nx the nyquist rate. Values greater than 1 mean a lower sampling rate",type=float,default=1.0)
    parser.add_argument("--trainPhantomFile",help="Location of the training phantom 'mat' file",default='./SheppLoganPhan.mat')
    parser.add_argument("--valPhantomFile",help="Location of the validation phantom 'mat' file",default='./YuYeWangPhan.mat')
    parser.add_argument("--dryRun", help="Use this flag to check the arguments without running training", action="store_true")    
    parser.add_argument("--testOnly", help="Use this flag to only run the test function. Make sure saveDir points to a run that contains a trained model",action="store_true")
    parser.add_argument("--noiseType", help="Select the type of noise to add to the phantom",choices=['additive','corrMult'],default='additive')
    args = parser.parse_args()
    
    #List of splits to train
    splitList = list(range(args.startSplit,args.startSplit+args.splitsToTrain))
    if args.dataType == 'phantom' or args.dataType == 'confocal' or args.dataType=='ct':
        cleanImageExists = True
    else:
        cleanImageExists = False
    #Sanity checks
    if args.splitsToTrain > args.nfolds:
        raise ValueError('Number of splits to train can\'t be larger than nfolds')
    if args.singleImage and args.cleanTargets:
        raise ValueError('Single Image with Clean Targets Doesn\'t make sense')
    if not cleanImageExists and args.cleanTargets:
        raise ValueError('No clean images to use as targets')
    if args.dataType.lower() =='oct' and args.cleanTargets:
        raise ValueError('I don\'t have the clean images for the OCT dataset :-(')
    if args.dataType.lower() =='rcm' and not args.singleImage:
        raise ValueError('I don\'t have image stacks for RCM :-(')
    # if args.testOnly and args.remakeCSV:
    #     raise ValueError('It\'s not a good idea to remake the CSV if you\'re just testing a model')
        
    if not args.dryRun:
        main(args.dataType,args.dataPath,args.csvDir,args.saveDir,remakeCSVs=args.remakeCSV,cleanTargets=args.cleanTargets,
             nextImFlag=args.nextImFlag,randomOrder=args.randomOrder,singleImageTraining=args.singleImage,
             noise2VoidFlag=args.noise2VoidFlag,loss_fn=args.loss_fn,batch_size=args.batch_size,initLR=args.init_lr,
             numTrainingIms=args.numTrainingIms,numValidIms=args.numValidIms,numTestIms=args.numTestIms,numEpochs=args.numEpochs,splitsToTrain=splitList,
             nfolds=args.nfolds,calcStats=args.calcStats,phantomNoiseLevel=args.phantomNoiseLevel,schedulerGamma=args.schedulerGamma,
             TVwt=args.TVwt,noiseCondition=args.noiseCondition,nyquistSampling=args.nyquistSampling,sampMult=args.sampMult,
             trainPhantomFile=args.trainPhantomFile,valPhantomFile=args.valPhantomFile,testOnly=args.testOnly,noiseType=args.noiseType)
    
