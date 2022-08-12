#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 23 17:05:56 2022

@author: matthew
"""
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import os
import seaborn as sns

#Setup matplotlib font sizes
SMALL_SIZE = 14
MEDIUM_SIZE = 20
BIGGER_SIZE = 22

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=18)    # fontsize of the tick labels
plt.rc('ytick', labelsize=18)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

#Helper function to get random idxs from a dataframe
#Inputs:
#   dataframe: Frame containing list of results from a single fold
#   numFrames: Number of frames to analyze
#Outputs:
#   outFrameList: A list of dataframes (one for each patient)
###################################################################
def getRandomNResults(dataframe,numFrames):
    patientScan = []
    frames = []
    outFrameList = []
    rng = np.random.default_rng()
    #Figure out which patients and frames belong to the input dataframe
    for i in range(len(dataframe)):
        thisName = dataframe['ImageName'][i]
        patientName = thisName[0:4]
        frameNum = int(thisName[-3::])
        patientScan.append(patientName)
        frames.append(frameNum)
        
    frames = np.array(frames)
    patients =list(set(patientScan))
    imsPerScan = []
    firstFrame = []
    #Calculate how many images are available for each patient and the lowest frame number
    for j in range(len(patients)):
        imsPerScan.append(patientScan.count(patients[j]))
        #f is the first index containing data from patient j
        f = [i for i,k in enumerate(patientScan) if k==patients[j]]
        firstFrame.append(np.min(f))
    #Go through each patient  
    for k in range(len(imsPerScan)):
        #Choose random indexes
        idxLocs = rng.choice(range(firstFrame[k],firstFrame[k]+imsPerScan[k]),size=numFrames,replace=False)
        outFrame = dataframe.iloc[idxLocs].copy()
        #Add patient to the new dataframe
        outFrame['Patient']=patients[k]
        #Add output dataframe to list
        outFrameList.append(outFrame)
    return outFrameList
        
    

if __name__ == '__main__':
    
    numFrames=256
    #####################
    #Confocal L1 Random order
    #######################
    cleanTargDir = '../../results/ct/2022-07-04--03-38-00'
    nextTargDir  = '../../results/ct/2022-07-06--07-54-21'
    n2vTargDir   = '../../results/ct/2022-07-08--13-02-37'
    line2lineDir = '../../results/ct/2022-07-10--19-50-07'
    folds = range(10)
    cleanDat = []
    nextDat = []
    n2vDat = []
    line2lineDat = []
    totalScans = 0
    #Give me a list of lists with each entry containing data from a patient (first idx fold, second idx patient)
    for f in folds:
        truncClean=pd.read_csv(os.path.join(cleanTargDir,'%02d'%f,'testResults_last.csv'))
        cleanResults=getRandomNResults(truncClean,numFrames)
        cleanDat.append(cleanResults)
        truncNext=pd.read_csv(os.path.join(nextTargDir,'%02d'%f,'testResults_last.csv'))
        nextDat.append(getRandomNResults(truncNext,numFrames))
        truncN2V=pd.read_csv(os.path.join(n2vTargDir,'%02d'%f,'testResults_last.csv'))
        n2vDat.append(getRandomNResults(truncN2V,numFrames))
        truncL2L=pd.read_csv(os.path.join(line2lineDir,'%02d'%f,'testResults_last.csv'))
        line2lineDat.append(getRandomNResults(truncL2L,numFrames))
        totalScans += len(cleanResults)
    #Plotting stuff    
    jitterAmt = 0.03
    spacing = [-0.3,-0.1,0.1,0.3]
    #Plot the first few patients as boxplots
    fig,ax = plt.subplots(1,1,figsize=(10,7))
    meanVal = 0
    trueMeanVal = 0
    patientCodes=[]
    for i in range(4):
        for j in range(len(cleanDat[i])):
            frameName = cleanDat[i][j]['ImageName'].iloc[0]
            patientCodes.append(cleanDat[i][j]['Patient'].iloc[0])
            cbox=ax.boxplot(cleanDat[i][j][' SSIM'],positions=[trueMeanVal+spacing[0]],labels=[''],patch_artist=True)
            nyqbox=ax.boxplot(nextDat[i][j][' SSIM'],positions=[trueMeanVal+spacing[1]],labels=[''],patch_artist=True)
            voidbox=ax.boxplot(n2vDat[i][j][' SSIM'],positions=[trueMeanVal+spacing[2]],labels=[''],patch_artist=True)
            linebox=ax.boxplot(line2lineDat[i][j][' SSIM'],positions=[trueMeanVal+spacing[3]],labels=[''],patch_artist=True)
            cbox['boxes'][0].set_facecolor('tab:blue')
            cbox['fliers'][0].set_markerfacecolor('tab:blue')
            [w.set_color('gray') for w in cbox['whiskers']]
            [c.set_color('gray') for c in cbox['caps']]
            cbox['medians'][0].set_color('black')
            cbox['medians'][0].set_linewidth(2)
            
            nyqbox['boxes'][0].set_facecolor('tab:orange')
            nyqbox['fliers'][0].set_markerfacecolor('tab:orange')
            [w.set_color('gray') for w in nyqbox['whiskers']]
            [c.set_color('gray') for c in nyqbox['caps']]
            nyqbox['medians'][0].set_color('black')
            nyqbox['medians'][0].set_linewidth(2)
            
            voidbox['boxes'][0].set_facecolor('tab:green')
            voidbox['fliers'][0].set_markerfacecolor('tab:green')
            [w.set_color('gray') for w in voidbox['whiskers']]
            [c.set_color('gray') for c in voidbox['caps']]
            voidbox['medians'][0].set_color('black')
            voidbox['medians'][0].set_linewidth(2)
            
            linebox['boxes'][0].set_facecolor('tab:red')
            linebox['fliers'][0].set_markerfacecolor('tab:red')
            [w.set_color('gray') for w in linebox['whiskers']]
            [c.set_color('gray') for c in linebox['caps']]
            linebox['medians'][0].set_color('black')
            linebox['medians'][0].set_linewidth(2)
            if i != 3 or j != len(cleanDat[i])-1:
                ax.plot([trueMeanVal+spacing[3]+0.2,trueMeanVal+spacing[3]+0.2],[0.6,1],'k',linewidth=2)
            trueMeanVal += 1
            meanVal +=1
    ax.plot([],[],'s',markersize=12,color='tab:blue',label='Supervised')
    ax.plot([],[],'s',markersize=12,color='tab:orange',label='Noise2Nyquist')
    ax.plot([],[],'s',markersize=12,color='tab:green',label='Noise2Void')
    ax.plot([],[],'s',markersize=12,color='tab:red',label='Line2Line')
    ax.set_xticks(range(len(patientCodes)))
    ax.set_xticklabels(patientCodes)
    ax.set_xlabel('Patient')
    #ax.xaxis.grid(True)
    ax.legend()
    ax.set_ylim([0.65,1])
    ax.yaxis.grid(True)
    ax.set_ylabel('SSIM')
    ax.set_title('CT Denoising Results')
  
    plt.tight_layout()
    plt.savefig('../../communications/paper/figures/ct/CTScanQuant.png')
    
    #Get average metrics for each patient
    avgClean = np.zeros((len(folds),3))
    avgNext = np.zeros((len(folds),3))
    avgVoid = np.zeros((len(folds),3))
    avgLine = np.zeros((len(folds),3))
    scanNum = 0
    for i in range(len(cleanDat)):
        for j in range(len(cleanDat[i])):
            avgCleanPSNR = np.mean(cleanDat[i][j][' PSNR'])
            avgCleanSSIM = np.mean(cleanDat[i][j][' SSIM'])
            avgCleanMSE = np.mean(cleanDat[i][j][' MSE'])
            avgClean[scanNum,:]=[avgCleanPSNR,avgCleanSSIM,avgCleanMSE]
            
            avgNextPSNR = np.mean(nextDat[i][j][' PSNR'])
            avgNextSSIM = np.mean(nextDat[i][j][' SSIM'])
            avgNextMSE = np.mean(nextDat[i][j][' MSE'])
            avgNext[scanNum,:]=[avgNextPSNR,avgNextSSIM,avgNextMSE]
            
            avgVoidPSNR = np.mean(n2vDat[i][j][' PSNR'])
            avgVoidSSIM = np.mean(n2vDat[i][j][' SSIM'])
            avgVoidMSE = np.mean(n2vDat[i][j][' MSE'])
            avgVoid[scanNum,:]=[avgVoidPSNR,avgVoidSSIM,avgVoidMSE]
            
            avgLinePSNR = np.mean(line2lineDat[i][j][' PSNR'])
            avgLineSSIM = np.mean(line2lineDat[i][j][' SSIM'])
            avgLineMSE = np.mean(line2lineDat[i][j][' MSE'])
            avgLine[scanNum,:]=[avgLinePSNR,avgLineSSIM,avgLineMSE]
            
            scanNum+=1
    #Then plot the average of each patient       
    fig,ax = plt.subplots(1,3,figsize=(10,7))
    ax[0].errorbar(spacing[0],np.mean(avgClean,0)[0],yerr=np.std(avgClean,0)[0],fmt='o',capsize=5,markersize=12,elinewidth=3,capthick=3)
    ax[0].errorbar(spacing[1],np.mean(avgNext,0)[0],yerr=np.std(avgNext,0)[0],fmt='o',capsize=5,markersize=12,elinewidth=3,capthick=3)
    ax[0].errorbar(spacing[2],np.mean(avgVoid,0)[0],yerr=np.std(avgVoid,0)[0],fmt='o',capsize=5,markersize=12,elinewidth=3,capthick=3)
    ax[0].errorbar(spacing[3],np.mean(avgLine,0)[0],yerr=np.std(avgLine,0)[0],fmt='o',capsize=5,markersize=12,elinewidth=3,capthick=3)
    ax[0].set_title(r'Peak SNR$\uparrow$')
    ax[0].set_ylabel('PSNR (dB)')
    ax[0].set_xticks(spacing)
    ax[0].set_xticklabels(['Sup.','N2Nyq','N2V','L2L'],rotation=35)
    
    ax[1].errorbar(spacing[0],np.mean(avgClean,0)[1],yerr=np.std(avgClean,0)[1],fmt='o',capsize=5,markersize=12,elinewidth=3,capthick=3)
    ax[1].errorbar(spacing[1],np.mean(avgNext,0)[1],yerr=np.std(avgNext,0)[1],fmt='o',capsize=5,markersize=12,elinewidth=3,capthick=3)
    ax[1].errorbar(spacing[2],np.mean(avgVoid,0)[1],yerr=np.std(avgVoid,0)[1],fmt='o',capsize=5,markersize=12,elinewidth=3,capthick=3)
    ax[1].errorbar(spacing[3],np.mean(avgLine,0)[1],yerr=np.std(avgLine,0)[1],fmt='o',capsize=5,markersize=12,elinewidth=3,capthick=3)
    ax[1].set_title(r'Structural Similarity$\uparrow$')
    ax[1].set_ylabel('SSIM')
    ax[1].set_xticks(spacing)
    ax[1].set_xticklabels(['Sup.','N2Nyq','N2V','L2L'],rotation=35)
     
    ax[2].errorbar(spacing[0],np.mean(avgClean,0)[2],yerr=np.std(avgClean,0)[2],fmt='o',capsize=5,markersize=12,elinewidth=3,capthick=3)
    ax[2].errorbar(spacing[1],np.mean(avgNext,0)[2],yerr=np.std(avgNext,0)[2],fmt='o',capsize=5,markersize=12,elinewidth=3,capthick=3)
    ax[2].errorbar(spacing[2],np.mean(avgVoid,0)[2],yerr=np.std(avgVoid,0)[2],fmt='o',capsize=5,markersize=12,elinewidth=3,capthick=3)
    ax[2].errorbar(spacing[3],np.mean(avgLine,0)[2],yerr=np.std(avgLine,0)[2],fmt='o',capsize=5,markersize=12,elinewidth=3,capthick=3)
    ax[2].set_title(r'Error$\downarrow$')
    ax[2].set_ylabel('MSE (AU)')
    ax[2].set_xticks(spacing)
    ax[2].set_xticklabels(['Sup.','N2Nyq','N2V','L2L'],rotation=35)
    plt.tight_layout()
    plt.savefig('../../communications/paper/figures/ct/CTMetrics.png')
    
    ######################
    #Version 2 with Seaborn
    #####################
    #Flatten each list of lists so there's one element for each patient
    supervisedFlat = [item for sublist in cleanDat for item in sublist]  
    n2nyqFlat = [item for sublist in nextDat for item in sublist]
    n2vFlat = [item for sublist in n2vDat for item in sublist]
    #Add method column
    for i in range(len(supervisedFlat)):
        supervisedFlat[i]['Method'] = 'Supervised'
        n2nyqFlat[i]['Method'] = 'noise2Nyquist'
        n2vFlat[i]['Method'] = 'noise2void'
    #Concatenate all patients    
    allSupDF = pd.concat(supervisedFlat)
    alln2nDF = pd.concat(n2nyqFlat)
    alln2vDF = pd.concat(n2vFlat)
    #Define patients to plot all data for
    patientsToPlot= ['L067','L096','L109','L143']
    #Make dataframe of all data
    allDatDF = pd.concat((allSupDF,alln2nDF,alln2vDF))
    #Subset dataframe to multiple patients
    boolList = allDatDF.Patient.isin(patientsToPlot)
    someScansDF = allDatDF[boolList]
    #Make violin plots
    fig,ax = plt.subplots(1,1,figsize=(10.5,6))
    ax.set_title('CT Denoising Results')
    p=sns.violinplot(x='Patient',y=' SSIM',hue='Method',data=someScansDF,ax=ax,inner='quartile',palette = 'Set2')
    plt.legend(loc='best')
    ax.set_xlabel('Patient Number')
    for l in p.lines:
     l.set_linestyle(':')
     l.set_linewidth(3)
     l.set_color('black')
     l.set_alpha(0.8)
    for l in p.lines[1::3]:
        l.set_linestyle('-')
    plt.savefig('../../communications/paper/figures/ct/patientResults.png')
    
    #Calculate mean values for each patient and method
    volDF=allDatDF.groupby(['Patient','Method']).mean()
    volDF.reset_index(inplace=True)
    
    #Make boxplots
    fig,ax=plt.subplots(1,3,figsize=(10.5,6))
    p1=sns.boxplot(x='Method',y=' PSNR',data=volDF,ax=ax[0],palette='Set2',width=.6,fliersize=0)
    sns.swarmplot(x='Method',y=' PSNR',data=volDF,ax=ax[0],color=".25",size=6)
    sns.boxplot(x='Method',y=' SSIM',data=volDF,ax=ax[1],palette='Set2',width=.6,fliersize=0)   
    sns.swarmplot(x='Method',y=' SSIM',data=volDF,ax=ax[1],color=".25",size=6)
    sns.boxplot(x='Method',y=' MSE',data=volDF,ax=ax[2],palette='Set2',width=.6,fliersize=0)
    sns.swarmplot(x='Method',y=' MSE',data=volDF,ax=ax[2],color=".25",size=6)
    ax[0].set_xticklabels(['Sup','n2Nyq.','n2v'])
    ax[1].set_xticklabels(['Sup.','n2Nyq.','n2v'])
    ax[2].set_xticklabels(['Sup.','n2Nyq.','n2v'])
    ax[0].set_title(r'Peak SNR$\uparrow$')
    ax[1].set_title(r'Structural Similarity$\uparrow$')
    ax[2].set_title(r'Error$\downarrow$')
    ax[0].set_ylabel('PSNR (dB)')
    plt.tight_layout()
    plt.savefig('../../communications/paper/figures/ct/aggregateResults.png')
    