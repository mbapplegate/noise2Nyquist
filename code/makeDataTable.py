# -*- coding: utf-8 -*-
"""
Created on Thu Jul 14 11:17:33 2022

@author: Matthew
"""

import pandas as pd
import numpy as np
import os
import glob
from scipy.io import loadmat

#Helper function to calculate mean and std from the phantom conventional processing directory
#Inputs:
#   dataDir: Where are the csv files with tabulated data located
#   method:  What method are you interested in
#
#Outputs:
#   avgStats: Average PSNR, SSIM, and MSE
#   stdStats: Standard deviation of the 3 metrics
#   data:     The raw data from which these summary stats are calculated
##########################################################################
def getPhantomStats(dataDir,method):
    datFile = os.path.join(dataDir,'testResults_%s.csv'%method)
    data = pd.read_csv(datFile)
    avgStats = [np.mean(data[' PSNR']),np.mean(data[' SSIM']),np.mean(data[' MSE'])]
    stdStats = [np.std(data[' PSNR']),np.std(data[' SSIM']),np.std(data[' MSE'])]
    return avgStats, stdStats,data

#Helper function to calculate mean and std from the confocal conventional processing directory
#Inputs:
#   dataDir: Where are the csv files with tabulated data located
#   method:  What method are you interested in
#
#Outputs:
#   avgStats: Average PSNR, SSIM, and MSE
#   stdStats: Standard deviation of the 3 metrics
##########################################################################
def getConventionalStats(dataDir,method):
    folds = sorted(os.listdir(dataDir))
    volAvgData=[]
    for f in folds:
        foldDir = os.path.join(dataDir,f)
        datFile = os.path.join(foldDir,'testResults_%s.csv'%method)
        data = pd.read_csv(datFile)
        patientList = np.unique(data['Scan'])
        for pat in patientList:
            truncDat = data[data['Scan']==pat]
            volAvgData.append([np.mean(truncDat[' PSNR']),np.mean(truncDat[' SSIM']),np.mean(truncDat[' MSE'])])
            
    avgStats = np.mean(np.array(volAvgData),0)
    stdStats = np.std(np.array(volAvgData),0)
    return avgStats,stdStats

#Helper function to calculate mean and std from the conventional processing directory for CT data
#Inputs:
#   dataDir: Where are the csv files with tabulated data located
#   method:  What method are you interested in
#
#Outputs:
#   avgStats: Average PSNR, SSIM, and MSE
#   stdStats: Standard deviation of the 3 metrics
##########################################################################
def getConventionalStatsCT(dataDir,method):
    fList = sorted(glob.glob(os.path.join(dataDir,'testResults_%s*.csv'%method)))
    patientAvgData = []
    for f in fList:
        data = pd.read_csv(f)
        patientAvgData.append([np.mean(data[' PSNR']),np.mean(data[' SSIM']),np.mean(data[' MSE'])])
    avgStats = np.mean(np.array(patientAvgData),0)
    stdStats = np.std(np.array(patientAvgData),0)
    return avgStats, stdStats

#Get mean and std of data for confocal and CT ML methods
#Inputs:
#   listOfData: list of lists with results (PSNR, SSIM) for each fold. Folds with multiple patients have multiple items
#Outputs:
#   avgStats: Average stats for each patient/volume
#   stdStats: std stats for each patient/volume
#############################################################################
def getSummaryStats(listOfData):
    #Flatten list of lists to get number of patients total
    numPatients = len([item for sublist in listOfData for item in sublist])
    #Preallocate
    volAvgData = np.zeros((numPatients,3))
    scanNum = 0
    #Go through each fold
    for i in range(len(listOfData)):
        #Go through each patient in that fold
        for j in range(len(listOfData[i])):
            #Calc stats
            avgPSNR = np.mean(listOfData[i][j][' PSNR'])
            avgSSIM = np.mean(listOfData[i][j][' SSIM'])
            avgMSE = np.mean(listOfData[i][j][' MSE'])
            #Fill up dataset
            volAvgData[scanNum,:]=[avgPSNR,avgSSIM,avgMSE]
            scanNum+=1
    #Calc aggregate averages
    avgStats = np.mean(volAvgData,0)
    stdStats = np.std(volAvgData,0)
    return avgStats,stdStats

#Get results from Random frames of the CT scan
#Inputs:
#   dataframe: Pandas dataframe containing the results 
#   numFrames: Number of frames to return
#Output:
#   outFrameList: List of dataframes (one element per patient) with PSNR, SSIM, etc.
#########################################################
def getRandomNResultsCT(dataframe,numFrames): 
    patientScan = []
    frames = []
    outFrameList = []
    rng = np.random.default_rng() #Start random number generator
    #Assign names and patient to each element of the dataframe
    for i in range(len(dataframe)):
        thisName = dataframe['ImageName'][i]
        patientName = thisName[0:4]
        frameNum = int(thisName[-3::])
        patientScan.append(patientName)
        frames.append(frameNum)
        
    frames = np.array(frames)
    #list of unique patients
    patients =list(set(patientScan))
    #Add patient to the dataframe
    dataframe['Patient']=patientScan
    #Go through each patient
    for j in range(len(patients)):
        #Filter the dataframe to just this patient
        filtDF = dataframe[dataframe['Patient']==patients[j]]
        #Get random entries of the dataframe and make a new dataframe
        getRandomIdxs = rng.choice(len(filtDF),numFrames,replace=False)
        randDF = filtDF.iloc[getRandomIdxs]   
    #Append this dataframe to the list    
    outFrameList.append(randDF)
    return outFrameList

#Get results from each confocal fold
#Inputs:
#   dataframe: Pandas dataframe containing all of the test data
#Outputs:
#   outFrameList: A list of dataframes (one for each volume) containing results
########################################################
def getResultsFCM(dataframe):
    #Add Volume number to the dataframe
    dataframe['Scan'] = None
    outFrameList = []
    #Vol number is encoded in image name
    for i in range(len(dataframe)):
        thisName = dataframe['ImageName'][i]
        scanNum = int(thisName[0:2])
        dataframe.loc[i,'Scan'] = scanNum
    #Unique volumes        
    numScans = np.unique(dataframe['Scan'])
    #Filter and append the dataframes
    for k in range(len(numScans)):
        scanDF = dataframe[dataframe['Scan']==numScans[k]]
        outFrameList.append(scanDF)
    return outFrameList

if __name__ == '__main__':
    #Code to make an epic table summerizing all the results!
    
    folds = range(10) #Folds to process
    
    #Number of CT frames to consider
    numCTFrames = 256
    baseResultsDir = '../results'
    #####################
    #Phantom L1 Random order
    #######################
    supervisedRunPhan = '2022-10-18--14-34-55'
    noise2NyquistRunPhan = '2022-10-18--16-17-49'
    noise2noiseRunPhan =  '2022-10-18--15-26-26'
    noise2voidRunPhan = '2022-10-18--18-13-49'
    #line2lineRunPhan = '2022-06-29--12-32-07'
    neigh2neighRunPhan='2022-10-17-12-04'
    #supervisedPhan = pd.read_csv(os.path.join('..','results','phantom',supervisedRunPhan,'00','testResults_last.csv'))
    #noise2NyquistPhan = pd.read_csv(os.path.join('..','results','phantom',noise2NyquistRunPhan,'00','testResults_last.csv'))
    #noise2noisePhan = pd.read_csv(os.path.join('..','results','phantom',noise2noiseRunPhan,'00','testResults_last.csv'))
    #noise2voidPhan = pd.read_csv(os.path.join('..','results','phantom',noise2voidRunPhan,'00','testResults_last.csv'))
    neigh2neighPhan=pd.read_csv(os.path.join('..','results','phantom','neigh2neigh',neigh2neighRunPhan,'00','testResults_last.csv'))
    #line2linePhan = pd.read_csv(os.path.join('..','results','phantom',line2lineRunPhan,'00','testResults_last.csv'))
    convenDirPhan = os.path.join('..','results','phantom','conventional')
   
    
    ##Fluorescence Confocal Data Dirs
    #Note: FCM was tested on 1536 random patches patches, not the full images
    #Conventional methods were likewise done on patches
    #This is because there are large black areas in the full images. Conventional
    #processing with affinity scaling yielded very high metrics b/c the 0 level was matched
    #Even the noisy image had a PSNR of ~30
    supervisedRunFCM = '2022-07-18--11-21-32'
    noise2NyquistRunFCM = '2022-07-18--11-41-04'
    noise2voidRunFCM = '2022-07-18--11-48-01'
    line2lineRunFCM = '2022-07-18--12-04-06'
    neigh2neighRunFCM='2022-10-14-14-35'
    supervisedDirFCM = os.path.join('..','results','confocal',supervisedRunFCM)
    noise2NyquistDirFCM  =  os.path.join('..','results','confocal',noise2NyquistRunFCM)
    noise2VoidDirFCM   = os.path.join('..','results','confocal',noise2voidRunFCM)
    line2lineDirFCM = os.path.join('..','results','confocal',line2lineRunFCM)
    neigh2neighDirFCM = os.path.join('..','results','confocal','neigh2neigh',neigh2neighRunFCM)
    convenDirFCM =  os.path.join('..','results','confocal','conventional')
    
    ##Computed Tomography data dirs
    #Conventional computed tomography was done on 256 frames
    #This was the largest number that could be done in 24 hours of cluster time
    supervisedRunCT = '2022-07-04--03-38-00'
    noise2NyquistRunCT= '2022-07-06--07-54-21'
    noise2voidRunCT= '2022-07-08--13-02-37'
    line2lineRunCT= '2022-07-10--19-50-07'
    neigh2neighRunCT = '2022-10-12-18-47'
    
    supervisedDirCT = os.path.join('..','results','ct',supervisedRunCT)
    noise2NyquistDirCT  = os.path.join('..','results','ct',noise2NyquistRunCT)
    noise2VoidDirCT   = os.path.join('..','results','ct',noise2voidRunCT)
    line2lineDirCT = os.path.join('..','results','ct',line2lineRunCT)
    convenDirCT = os.path.join('..','results','ct','conventional')
    neigh2neighDirCT = os.path.join('..','results','ct','neigh2neigh',neigh2neighRunCT)
    
    ##OCT data dirs
    #Conventionally processed OCT data was done on 96 frames/volume
    #This was the most frames I could process in 24 hrs on the cluster
    #OCT was actually processed in Matlab by ./analyzeProcessedOCT.m
    noise2NyquistRunOCT='2022-07-15--13-45-14'
    noise2voidRunCT='2022-07-13--17-39-29'
    line2lineRunCT='2022-07-19--03-17-51'
    neigh2neighRunOCT='2022-10-13-15-33'
    
    #Get all phantom Data
    #Boil everything down to mean +/- standard deviation
    #But I want the standard deviation with n=number of patients
    print("Collecting Phantom data...")
    supervisedPhanResults = []
    noise2noisePhanResults=[]
    noise2nyquistPhanResults=[]
    noise2voidPhanResults=[]
    for f in folds:
        truncSup = pd.read_csv(os.path.join('..','results','phantom','runs',supervisedRunPhan,'%02d'%f,'testResults_last.csv'))
        supervisedPhanResults.append([np.mean(truncSup[' PSNR']),np.mean(truncSup[' SSIM']), np.mean(truncSup[' MSE'])])
        truncN2N = pd.read_csv(os.path.join('..','results','phantom','runs',noise2noiseRunPhan,'%02d'%f,'testResults_last.csv'))
        noise2noisePhanResults.append([np.mean(truncN2N[' PSNR']),np.mean(truncN2N[' SSIM']), np.mean(truncN2N[' MSE'])])
        truncN2Nyq=pd.read_csv(os.path.join('..','results','phantom','runs',noise2NyquistRunPhan,'%02d'%f,'testResults_last.csv'))
        noise2nyquistPhanResults.append([np.mean(truncN2Nyq[' PSNR']),np.mean(truncN2Nyq[' SSIM']), np.mean(truncN2Nyq[' MSE'])])
        truncN2V=pd.read_csv(os.path.join('..','results','phantom','runs',noise2voidRunPhan,'%02d'%f,'testResults_last.csv'))
        noise2voidPhanResults.append([np.mean(truncN2V[' PSNR']),np.mean(truncN2V[' SSIM']), np.mean(truncN2V[' MSE'])])
    supervisedPhanAvg = np.mean(np.array(supervisedPhanResults),0)
    supervisedPhanStd = np.std(np.array(supervisedPhanResults),0)
    noise2NoisePhanAvg = np.mean(np.array(noise2noisePhanResults),0)
    noise2NoisePhanStd = np.std(np.array(noise2noisePhanResults),0)
    noise2NyquistPhanAvg=np.mean(np.array(noise2nyquistPhanResults),0)
    noise2NyquistPhanStd=np.std(np.array(noise2nyquistPhanResults),0)
    noise2VoidPhanAvg=np.mean(np.array(noise2voidPhanResults),0)
    noise2VoidPhanStd=np.std(np.array(noise2voidPhanResults),0)
    
    #supervisedPhanAvg = [np.mean(supervisedPhan[' PSNR']), np.mean(supervisedPhan[' SSIM']), np.mean(supervisedPhan[' MSE'])]
    #supervisedPhanStd= [np.std(supervisedPhan[' PSNR']), np.std(supervisedPhan[' SSIM']), np.std(supervisedPhan[' MSE'])]
    #noise2NyquistPhanAvg =[np.mean(noise2NyquistPhan[' PSNR']), np.mean(noise2NyquistPhan[' SSIM']), np.mean(noise2NyquistPhan[' MSE'])]
    #noise2NyquistPhanStd =[np.std(noise2NyquistPhan[' PSNR']), np.std(noise2NyquistPhan[' SSIM']), np.std(noise2NyquistPhan[' MSE'])]
    #noise2VoidPhanAvg = [np.mean(noise2voidPhan[' PSNR']), np.mean(noise2voidPhan[' SSIM']), np.mean(noise2voidPhan[' MSE'])]
    #noise2VoidPhanStd = [np.std(noise2voidPhan[' PSNR']), np.std(noise2voidPhan[' SSIM']), np.std(noise2voidPhan[' MSE'])]
    #noise2NoisePhanAvg =  [np.mean(noise2noisePhan[' PSNR']), np.mean(noise2noisePhan[' SSIM']), np.mean(noise2noisePhan[' MSE'])]
    #noise2NoisePhanStd =  [np.std(noise2noisePhan[' PSNR']), np.std(noise2noisePhan[' SSIM']), np.std(noise2noisePhan[' MSE'])]
    neigh2neighPhanAvg=[np.mean(neigh2neighPhan[' PSNR']),np.mean(neigh2neighPhan[' SSIM']), np.mean(neigh2neighPhan[' MSE'])]
    neigh2neighPhanStd=[np.std(neigh2neighPhan[' PSNR']),np.std(neigh2neighPhan[' SSIM']), np.std(neigh2neighPhan[' MSE'])]
    #line2LinePhanAvg =  [np.mean(line2linePhan[' PSNR']), np.mean(line2linePhan[' SSIM']), np.mean(line2linePhan[' MSE'])]
    #line2LinePhanStd =  [np.std(line2linePhan[' PSNR']), np.std(line2linePhan[' SSIM']), np.std(line2linePhan[' MSE'])]
   
    noisyPhanAvg, noisyPhanStd,noisyData = getPhantomStats(convenDirPhan,'none0')
    medianPhanAvg, medianPhanStd,medianData = getPhantomStats(convenDirPhan,'median3')
    gaussianPhanAvg, gaussianPhanStd,gaussianData = getPhantomStats( convenDirPhan,'gaussian1')
    oofPhanAvg,oofPhanStd,oofData = getPhantomStats(convenDirPhan,'oofAvg3')
    bm3dPhanAvg,bm3dPhanStd,bm3dData = getPhantomStats(convenDirPhan,'bm3d45')
    bm4dPhanAvg,bm4dPhanStd,bm4dData = getPhantomStats(convenDirPhan,'bm4d[3, 45]')
    
    #Get all Confocal Data
    #Boil everything down to mean +/- standard deviation
    #But I want the standard deviation with n=number of volumes
    print("Collecting Confocal Data...")
    supervisedFCMResults = []
    noise2NyquistFCMResults = []
    noise2VoidFCMResults = []
    line2LineFCMResults = []
    neigh2neighFCMResults=[]
    for f in folds:
        truncClean=pd.read_csv(os.path.join(supervisedDirFCM,'%02d'%f,'testResults_last.csv'))
        supervisedFCMResults.append(getResultsFCM(truncClean))
    
        truncNext=pd.read_csv(os.path.join(noise2NyquistDirFCM,'%02d'%f,'testResults_last.csv'))
        noise2NyquistFCMResults.append(getResultsFCM(truncNext))
        truncN2V=pd.read_csv(os.path.join(noise2VoidDirFCM,'%02d'%f,'testResults_last.csv'))
        noise2VoidFCMResults.append(getResultsFCM(truncN2V))
        truncL2L=pd.read_csv(os.path.join(line2lineDirFCM,'%02d'%f,'testResults_last.csv'))
        line2LineFCMResults.append(getResultsFCM(truncL2L))
        truncNeigh2Neigh=pd.read_csv(os.path.join(neigh2neighDirFCM,'%02d'%f,'testResults_last.csv'))
        neigh2neighFCMResults.append(getResultsFCM(truncNeigh2Neigh))
        
    supervisedFCMAvg, supervisedFCMStd = getSummaryStats(supervisedFCMResults)
    noise2NyquistFCMAvg, noise2NyquistFCMStd = getSummaryStats(noise2NyquistFCMResults)
    noise2VoidFCMAvg, noise2VoidFCMStd = getSummaryStats(noise2VoidFCMResults)
    line2LineFCMAvg, line2LineFCMStd = getSummaryStats(line2LineFCMResults)
    neigh2neighFCMAvg,neigh2neighFCMStd = getSummaryStats(neigh2neighFCMResults)
    
    noisyFCMAvg, noisyFCMStd = getConventionalStats(convenDirFCM,'none0')
    medianFCMAvg, medianFCMStd = getConventionalStats(convenDirFCM,'median3')
    gaussianFCMAvg, gaussianFCMStd = getConventionalStats(convenDirFCM,'gaussian1')
    oof3FCMAvg, oof3FCMStd = getConventionalStats(convenDirFCM,'oofAvg3')
    bm3dFCMAvg, bm3dFCMStd = getConventionalStats(convenDirFCM,'bm3d0.2')
    bm4dFCMAvg, bm4dFCMStd = getConventionalStats(convenDirFCM,'bm4d0.2')
    
    #Get all CT Data
    #Boil everything down to mean +/- standard deviation
    #But I want the standard deviation with n=number of patients
    print("Collecting CT Data...")
    supervisedCTResults = []
    noise2NyquistCTResults = []
    noise2VoidCTResults = []
    line2LineCTResults = []
    neigh2neighCTResults = []
    for f in folds:
        truncClean=pd.read_csv(os.path.join(supervisedDirCT,'%02d'%f,'testResults_last.csv'))
        supervisedCTResults.append(getRandomNResultsCT(truncClean,numCTFrames))
        truncNext=pd.read_csv(os.path.join(noise2NyquistDirCT,'%02d'%f,'testResults_last.csv'))
        noise2NyquistCTResults.append(getRandomNResultsCT(truncNext,numCTFrames))
        truncN2V=pd.read_csv(os.path.join(noise2VoidDirCT,'%02d'%f,'testResults_last.csv'))
        noise2VoidCTResults.append(getRandomNResultsCT(truncN2V,numCTFrames))
        truncL2L=pd.read_csv(os.path.join(line2lineDirCT,'%02d'%f,'testResults_last.csv'))
        line2LineCTResults.append(getRandomNResultsCT(truncL2L,numCTFrames))
        truncNe2Ne=pd.read_csv(os.path.join(neigh2neighDirCT,'%02d'%f,'testResults_last.csv'))
        neigh2neighCTResults.append(getRandomNResultsCT(truncNe2Ne,numCTFrames))
        
    supervisedCTAvg, supervisedCTStd = getSummaryStats(supervisedCTResults)
    noise2NyquistCTAvg, noise2NyquistCTStd = getSummaryStats(noise2NyquistCTResults)
    noise2VoidCTAvg, noise2VoidCTStd = getSummaryStats(noise2VoidCTResults)
    line2LineCTAvg, line2LineCTStd = getSummaryStats(line2LineCTResults)
    neigh2neighCTAvg,neigh2neighCTStd=getSummaryStats(neigh2neighCTResults)
    
    noisyCTAvg, noisyCTStd = getConventionalStatsCT(convenDirCT,'none0')
    medianCTAvg, medianCTStd = getConventionalStatsCT(convenDirCT,'median3')
    gaussianCTAvg, gaussianCTStd = getConventionalStatsCT(convenDirCT,'gaussian1')
    oof3CTAvg, oof3CTStd = getConventionalStatsCT(convenDirCT,'oofAvg3')
    bm3dCTAvg, bm3dCTStd = getConventionalStatsCT(convenDirCT,'bm3d0.05')
    bm4dCTAvg, bm4dCTStd = getConventionalStatsCT(convenDirCT,'bm4d0.05')
    
    #Boil everything down to mean +/- standard deviation
    #But I want the standard deviation with n=number of patients
    print("Collecting OCT Data...")
    conventionalNIQI = '../results/oct/NIQIRatio_Conventional.mat'
    MLNIQI = '../results/oct/NIQIRatio_MLMethods.mat'
    #So this is 3 ML methods [n2nyq, n2void, l2l], 35 volumes, 96 central frames
    MLDat = loadmat(MLNIQI)
    MLDat = np.array(MLDat['MLNIQIRatio'])
    volumeListML = np.zeros((MLDat.shape[1],MLDat.shape[2]),dtype='int')
    frameListML = np.zeros((MLDat.shape[1],MLDat.shape[2]),dtype='int')
    for i in range(volumeListML.shape[0]):
        volumeListML[i,:] = i
        frameListML[i,:] = range(frameListML.shape[1])
    n2nyqDF = pd.DataFrame({'Method':[],'Volume':[],'Frame':[],'NIQI':[]})
    n2voidDF = pd.DataFrame({'Method':[],'Volume':[],'Frame':[],'NIQI':[]})
    neigh2neighDF=pd.DataFrame({'Method':[],'Volume':[],'Frame':[],'NIQI':[]})
    
    n2nyqDF['NIQI'] = MLDat[0,:,:].flatten()
    n2voidDF['NIQI'] = MLDat[1,:,:].flatten()
    neigh2neighDF['NIQI']=MLDat[3,:,:].flatten()
    n2nyqDF['Method']='noise2Nyquist'
    n2voidDF['Method'] = 'noise2void'
    neigh2neighDF['Method']='neighbor2neighbor'
    n2nyqDF['Volume'] = volumeListML.flatten()
    n2voidDF['Volume'] = volumeListML.flatten()
    neigh2neighDF['Volume']=volumeListML.flatten()
    n2nyqDF['Frame'] = frameListML.flatten()
    n2voidDF['Frame'] = frameListML.flatten()
    neigh2neighDF['Frame']=frameListML.flatten()
    MLdf = pd.concat((n2nyqDF,n2voidDF,neigh2neighDF))
    
    volDF=MLdf.groupby(['Volume','Method']).mean()
    volDF.reset_index(inplace=True)
    
    MLOCTAvg = volDF.groupby(['Method']).mean()
    MLOCTStd = volDF.groupby(['Method']).std()
    
    convDat = loadmat(conventionalNIQI)
    #This is 5 conventional methods [median,gaussian,oofAvg,bm3d,bm4d] x 35 vols x 96 center frames
    convDat = np.array(convDat['NIQIRatio'])
   
    methods = ['median','gaussian','oofAvg','bm3d','bm4d']
    volumeListConv = np.zeros(convDat.shape,dtype='int')
    frameListConv = np.zeros(convDat.shape,dtype='int')
    methodMatrix = np.empty(convDat.shape,dtype='object')
    for i in range(volumeListConv.shape[0]):
        methodMatrix[i,:,:] = methods[i]
        for j in range(volumeListConv.shape[1]):
            volumeListConv[i,j,:] = j
            frameListConv[i,j,:] = range(frameListConv.shape[2])
    
    convDF = pd.DataFrame({'Method':[],'Volume':[],'Frame':[],'NIQI':[]})
    convDF['NIQI'] = convDat.flatten()
    convDF['Volume']=volumeListConv.flatten()
    convDF['Method']=methodMatrix.flatten()
    convDF['Frame'] =frameListConv.flatten()
    
    #Here I group by volume and denoising method
    volDF=convDF.groupby(['Volume','Method']).mean()
    volDF.reset_index(inplace=True)
    #Then I calculate mean and std
    convOCTAvg = volDF.groupby(['Method']).mean()
    convOCTStd = volDF.groupby(['Method']).std()
    
    ############################################
    #Collect phantom data into nice dataframe to copy to table
    ################################################
    phantomStatsDF = pd.DataFrame({'Method':[],'avgPSNR':[],'stdPSNR':[],'avgSSIM':[],'stdSSIM':[],'avgMSE':[],'stdMSE':[]})
    phantomStatsDF['Method']=['Noisy','Supervised','noise2Nyquist','noise2noise','noise2void','neighbor2neighbor',
                              'Median','Gaussian','Stack Avg.','BM3D','BM4D']
    phantomStatsDF.iloc[0,1::] =  [val for pair in zip(noisyPhanAvg,noisyPhanStd) for val in pair]
    phantomStatsDF.iloc[1,1::] =  [val for pair in zip(supervisedPhanAvg,supervisedPhanStd) for val in pair]
    phantomStatsDF.iloc[2,1::] =  [val for pair in zip(noise2NyquistPhanAvg,noise2NyquistPhanStd) for val in pair]
    phantomStatsDF.iloc[3,1::] =  [val for pair in zip(noise2NoisePhanAvg,noise2NoisePhanStd) for val in pair]
    phantomStatsDF.iloc[4,1::] =  [val for pair in zip(noise2VoidPhanAvg,noise2VoidPhanStd) for val in pair]
    phantomStatsDF.iloc[5,1::] =  [val for pair in zip(neigh2neighPhanAvg,neigh2neighPhanStd) for val in pair]
    phantomStatsDF.iloc[6,1::] =  [val for pair in zip(medianPhanAvg,medianPhanStd) for val in pair]
    phantomStatsDF.iloc[7,1::] =  [val for pair in zip(gaussianPhanAvg,gaussianPhanStd) for val in pair]
    phantomStatsDF.iloc[8,1::] =  [val for pair in zip(oofPhanAvg,oofPhanStd) for val in pair]
    phantomStatsDF.iloc[9,1::] =  [val for pair in zip(bm3dPhanAvg,bm3dPhanStd) for val in pair]
    phantomStatsDF.iloc[10,1::] =  [val for pair in zip(bm4dPhanAvg,bm4dPhanStd) for val in pair]
    
    ########################################
    #Collect FCM data into nice dataframe to copy to table
    ########################################
    FCMStatsDF = pd.DataFrame({'Method':[],'avgPSNR':[],'stdPSNR':[],'avgSSIM':[],'stdSSIM':[],'avgMSE':[],'stdMSE':[]})
    FCMStatsDF['Method']=['Noisy','Supervised','noise2Nyquist','noise2void','neighbor2neighbor',
                          'Median','Gaussian','Stack Avg.','BM3D','BM4D']
    FCMStatsDF.iloc[0,1::] = [val for pair in zip(noisyFCMAvg, noisyFCMStd) for val in pair]
    FCMStatsDF.iloc[1,1::] = [val for pair in zip(supervisedFCMAvg, supervisedFCMStd) for val in pair]
    FCMStatsDF.iloc[2,1::] = [val for pair in zip(noise2NyquistFCMAvg, noise2NyquistFCMStd) for val in pair]
    FCMStatsDF.iloc[3,1::] = [val for pair in zip(noise2VoidFCMAvg, noise2VoidFCMStd) for val in pair]
    FCMStatsDF.iloc[4,1::] = [val for pair in zip(neigh2neighFCMAvg,neigh2neighFCMStd) for val in pair]
    FCMStatsDF.iloc[5,1::] = [val for pair in zip(medianFCMAvg, medianFCMStd) for val in pair]
    FCMStatsDF.iloc[6,1::] = [val for pair in zip(gaussianFCMAvg, gaussianFCMStd) for val in pair]
    FCMStatsDF.iloc[7,1::] = [val for pair in zip(oof3FCMAvg, oof3FCMStd) for val in pair]
    FCMStatsDF.iloc[8,1::] = [val for pair in zip(bm3dFCMAvg, bm3dFCMStd) for val in pair]
    FCMStatsDF.iloc[9,1::] = [val for pair in zip(bm4dFCMAvg, bm4dFCMStd) for val in pair]
             
    ########################################
    #Collect CT data into nice dataframe to copy to table
    ########################################
    CTStatsDF = pd.DataFrame({'Method':[],'avgPSNR':[],'stdPSNR':[],'avgSSIM':[],'stdSSIM':[],'avgMSE':[],'stdMSE':[]})
    CTStatsDF['Method']=['Noisy','Supervised','noise2Nyquist','noise2void','neighbor2neighbor',
                          'Median','Gaussian','Stack Avg.','BM3D','BM4D']
    CTStatsDF.iloc[0,1::] = [val for pair in zip(noisyCTAvg, noisyCTStd) for val in pair]
    CTStatsDF.iloc[1,1::] = [val for pair in zip(supervisedCTAvg, supervisedCTStd) for val in pair]
    CTStatsDF.iloc[2,1::] = [val for pair in zip(noise2NyquistCTAvg,noise2NyquistCTStd) for val in pair]
    CTStatsDF.iloc[3,1::] = [val for pair in zip(noise2VoidCTAvg,noise2VoidCTStd) for val in pair]
    CTStatsDF.iloc[4,1::] = [val for pair in zip(neigh2neighCTAvg,neigh2neighCTStd) for val in pair]
    CTStatsDF.iloc[5,1::] = [val for pair in zip(medianCTAvg,medianCTStd) for val in pair]
    CTStatsDF.iloc[6,1::] = [val for pair in zip(gaussianCTAvg,gaussianCTStd) for val in pair]
    CTStatsDF.iloc[7,1::] = [val for pair in zip(oof3CTAvg,oof3CTStd) for val in pair]
    CTStatsDF.iloc[8,1::] = [val for pair in zip(bm3dCTAvg,bm3dCTStd) for val in pair]
    CTStatsDF.iloc[9,1::] = [val for pair in zip(bm4dCTAvg,bm4dCTStd) for val in pair]
    
    #########################################
    #Collect OCT Data into nice dataframe
    ########################################
    OCTStatsDF=pd.DataFrame({'Method':[],'avgNIQI':[],'stdNIQI':[]})
    OCTStatsDF['Method'] = ['neighbor2neighbor','noise2Nyquist','noise2void','Median','Gaussian','Stack Avg.','BM3D','BM4D']
    OCTStatsDF['avgNIQI'] = [MLOCTAvg['NIQI'][0].item(),MLOCTAvg['NIQI'][1].item(),MLOCTAvg['NIQI'][2].item(),
                             convOCTAvg['NIQI']['median'].item(),convOCTAvg['NIQI']['gaussian'].item(),
                             convOCTAvg['NIQI']['oofAvg'].item(),convOCTAvg['NIQI']['bm3d'].item(),
                             convOCTAvg['NIQI']['bm4d'].item()]
    OCTStatsDF['stdNIQI'] = [MLOCTStd['NIQI'][0].item(),MLOCTStd['NIQI'][1].item(),MLOCTStd['NIQI'][2].item(),
                             convOCTStd['NIQI']['median'].item(),convOCTStd['NIQI']['gaussian'].item(),
                             convOCTStd['NIQI']['oofAvg'].item(),convOCTStd['NIQI']['bm3d'].item(),
                             convOCTStd['NIQI']['bm4d'].item()]

     