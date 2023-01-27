# -*- coding: utf-8 -*-
"""
Created on Thu Aug  4 09:49:02 2022

@author: Matthew
"""
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import os
import seaborn as sns
from scipy.io import loadmat
import numpy as np

#Setup font sizes for matplotlib
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

if __name__ == '__main__':
    #NIQI values processed in Matlab
    conventionalNIQI = '../../results/oct/NIQIRatio_Conventional.mat'
    MLNIQI = '../../results/oct/NIQIRatio_MLMethods.mat'
    #So this is 3 methods [n2nyq, n2void, l2l], 35 volumes, 96 frames
    MLDat = loadmat(MLNIQI)
    MLDat = np.array(MLDat['MLNIQIRatio'])
    #Preallocate to add volumes/frames to dataframes
    volumeList = np.zeros((MLDat.shape[1],MLDat.shape[2]),dtype='int')
    frameList = np.zeros((MLDat.shape[1],MLDat.shape[2]),dtype='int')
    #This is basically just a meshgrid to get volume number and frame number for later processing
    for i in range(volumeList.shape[0]):
        volumeList[i,:] = i
        frameList[i,:] = range(frameList.shape[1])
    #Empty dataframes
    n2nyqDF = pd.DataFrame({'Method':[],'Volume':[],'Frame':[],'NIQI':[]})
    n2voidDF = pd.DataFrame({'Method':[],'Volume':[],'Frame':[],'NIQI':[]})
    ne2neDF = pd.DataFrame({'Method':[],'Volume':[],'Frame':[],'NIQI':[]})
    #Fill out the NIQI ratio, method, volume, and frame to the data
    n2nyqDF['NIQI'] = MLDat[0,:,:].flatten()
    n2voidDF['NIQI'] = MLDat[1,:,:].flatten()
    ne2neDF['NIQI'] = MLDat[2,:,:].flatten()
    n2nyqDF['Method']='noise2Nyquist'
    n2voidDF['Method'] = 'noise2void'
    ne2neDF['Method'] = 'neighbor2neighbor'
    n2nyqDF['Volume'] = volumeList.flatten()
    n2voidDF['Volume'] = volumeList.flatten()
    ne2neDF['Volume'] = volumeList.flatten()
    n2nyqDF['Frame'] = frameList.flatten()
    n2voidDF['Frame'] = frameList.flatten()
    ne2neDF['Frame'] = frameList.flatten()
    
    #Concatenate both ML methods
    MLdf = pd.concat((n2nyqDF,n2voidDF,ne2neDF))
    #filter dataframe by these volumes
    volsToPlot = [0,5,10,15]
    boolList = MLdf.Volume.isin(volsToPlot)
    someScansDF = MLdf[boolList]
    
    #Plot the violins
    pal = sns.color_palette('Set2')
    pal = pal[1::]
    fig,ax = plt.subplots(1,1,figsize=(10.5,6))
    ax.set_title('OCT Denoising Results')
    p=sns.violinplot(x='Volume',y='NIQI',hue='Method',data=someScansDF,ax=ax,inner='quartile',palette = pal)
    plt.legend(loc='upper left')
    ax.set_xlabel('Volume Number')
    ax.set_ylabel('NIQI Ratio')
    for l in p.lines:
     l.set_linestyle(':')
     l.set_linewidth(3)
     l.set_color('black')
     l.set_alpha(0.8)
    for l in p.lines[1::3]:
        l.set_linestyle('-')
    plt.savefig('../../results/oct/figures/scanByScanResults.png')
    
    #Calculate aggregate averages
    volDF=MLdf.groupby(['Volume','Method']).mean()
    volDF.reset_index(inplace=True)
    #Plot the barplots
    fig,ax=plt.subplots(1,1,figsize=(10.5,6))
    p1=sns.boxplot(x='Method',y='NIQI',data=volDF,ax=ax,palette=pal,width=.6,fliersize=0,order=['noise2Nyquist','noise2void','neighbor2neighbor'])
    sns.swarmplot(x='Method',y='NIQI',data=volDF,ax=ax,color=".25",size=8,order=['noise2Nyquist','noise2void','neighbor2neighbor'])
    ax.set_xticklabels(['noise2Nyquist.','noise2void','neighbor2neighbor'])
    ax.set_title(r'Naturalness Image Index$\downarrow$')
    ax.set_ylabel('NIQI ratio')
    plt.tight_layout()
    plt.savefig('../../results/oct/figures/aggregateResults.png')
    
    
    
    
    