#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 23 17:05:56 2022

@author: matthew
"""
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

##########Set matplotlib font sizes#########################
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
    
    #####################
    #Phantom L1 Random order directories
    #######################
    cleanTarg = pd.read_csv('../../results/phantom/2022-06-29--12-00-53/00/testResults_last.csv')
    nextTarg = pd.read_csv('../../results/phantom/2022-06-29--12-15-22/00/testResults_last.csv')
    noisyTarg = pd.read_csv('../../results/phantom/2022-06-29--12-50-44/00/testResults_last.csv')
    n2vTarg = pd.read_csv('../../results/phantom/2022-06-29--12-41-28/00/testResults_last.csv')
    neigh2neigh = pd.read_csv('../../results/phantom/neigh2neigh/2022-10-17-12-04/00/testResults_last.csv')
    n2nyqSingle = pd.read_csv('../../results/phantom/2022-06-29--12-32-07/00/testResults_last.csv')
    n2nyqO2 = pd.read_csv('../../results/phantom/2022-06-29--12-22-34/00/testResults_last.csv')
    n2nyqO3 = pd.read_csv('../../results/phantom/2022-06-29--12-26-37/00/testResults_last.csv')
    n2nyqO4 = pd.read_csv('../../results/phantom/2022-06-29--12-29-37/00/testResults_last.csv')
    n2nyqx2 = pd.read_csv('../../results/phantom/2022-08-03--17-15-42/00/testResults_last.csv')
    n2nyqx4 = pd.read_csv('../../results/phantom/2022-08-03--17-27-48/00/testResults_last.csv')
    
    #Read the data into dataframes with "Method"
    supervisedDF = pd.DataFrame({'Method':'Supervised','PSNR':cleanTarg[' PSNR'], 'SSIM':cleanTarg[' SSIM'],'MSE':cleanTarg[' MSE']})
    n2NyquistDF = pd.DataFrame({'Method':'noise2Nyq','PSNR':nextTarg[' PSNR'], 'SSIM':nextTarg[' SSIM'],'MSE':nextTarg[' MSE']})
    n2nDF = pd.DataFrame({'Method':'noise2noise','PSNR':noisyTarg[' PSNR'], 'SSIM':noisyTarg[' SSIM'],'MSE':noisyTarg[' MSE']})
    n2vDF = pd.DataFrame({'Method':'noise2void','PSNR':n2vTarg[' PSNR'], 'SSIM':n2vTarg[' SSIM'],'MSE':n2vTarg[' MSE']})
    ne2neDF = pd.DataFrame({'Method':'ne2ne','PSNR':neigh2neigh[' PSNR'], 'SSIM':neigh2neigh[' SSIM'], 'MSE':neigh2neigh[' MSE']})
    n2NyqO2DF = pd.DataFrame({'Method':'Nyq./2','PSNR':n2nyqO2[' PSNR'], 'SSIM':n2nyqO2[' SSIM'],'MSE':n2nyqO2[' MSE']})
    n2NyqO3DF = pd.DataFrame({'Method':'Nyq./3','PSNR':n2nyqO3[' PSNR'], 'SSIM':n2nyqO3[' SSIM'],'MSE':n2nyqO3[' MSE']})
    n2NyqO4DF = pd.DataFrame({'Method':'Nyq./4','PSNR':n2nyqO4[' PSNR'], 'SSIM':n2nyqO4[' SSIM'],'MSE':n2nyqO4[' MSE']})
    #n2Nyqx2DF = pd.DataFrame({'Method':'Nyq.x2','PSNR':n2nyqx2[' PSNR'], 'SSIM':n2nyqx2[' SSIM'],'MSE':n2nyqx2[' MSE']})
    #n2Nyqx4DF = pd.DataFrame({'Method':'Nyq.x4','PSNR':n2nyqx4[' PSNR'], 'SSIM':n2nyqx4[' SSIM'],'MSE':n2nyqx4[' MSE']})
    phantomDF = pd.concat((supervisedDF,n2NyquistDF,n2nDF,n2vDF,n2NyqO2DF,n2NyqO3DF,n2NyqO4DF))#,n2Nyqx2DF,n2Nyqx4DF))
   
    #Try plotting with jitter (like a strip plot)
    datShape = cleanTarg[' PSNR'].shape[0]
    o2Shape = n2nyqO2[' PSNR'].shape[0]
    o3Shape = n2nyqO3[' PSNR'].shape[0]
    o4Shape = n2nyqO4[' PSNR'].shape[0]
    jitterStd = 0.04
    fig,ax = plt.subplots(1,2,figsize=(16,8))
  
    
    p1=ax[0].plot(np.ones(datShape)+np.random.normal(0,jitterStd,datShape),cleanTarg[' SSIM'],'.')
    p2=ax[0].plot(np.ones(datShape+1)*2+np.random.normal(0,jitterStd,datShape+1),noisyTarg[' SSIM'],'.')
    p5=ax[0].plot(np.ones(datShape)*3+np.random.normal(0,jitterStd,datShape),nextTarg[' SSIM'],'.')
    p6=ax[0].plot(np.ones(o2Shape)*4+np.random.normal(0,jitterStd,o2Shape),n2nyqO2[' SSIM'],'.')
    p7=ax[0].plot(np.ones(o3Shape)*5+np.random.normal(0,jitterStd,o3Shape),n2nyqO3[' SSIM'],'.')
    p8=ax[0].plot(np.ones(o4Shape)*6+np.random.normal(0,jitterStd,o4Shape),n2nyqO4[' SSIM'],'.')
    p3=ax[0].plot(np.ones(datShape)*7+np.random.normal(0,jitterStd,datShape),n2vTarg[' SSIM'],'.')
    p4=ax[0].plot(np.ones(datShape)*8+np.random.normal(0,jitterStd,datShape),n2nyqSingle[' SSIM'],'.')

   
    ax[0].plot([0.8,1.2],[np.mean(cleanTarg[' SSIM']),np.mean(cleanTarg[' SSIM'])],color=p1[0].get_color(),linewidth=3)
    ax[0].plot([1.8,2.2],[np.mean(noisyTarg[' SSIM']),np.mean(noisyTarg[' SSIM'])],color=p2[0].get_color(),linewidth=3)
    ax[0].plot([2.8,3.2],[np.mean(nextTarg[' SSIM']),np.mean(nextTarg[' SSIM'])],color=p5[0].get_color(),linewidth=3)
    ax[0].plot([3.8,4.2],[np.mean(n2nyqO2[' SSIM']),np.mean(n2nyqO2[' SSIM'])],color=p6[0].get_color(),linewidth=3)
    ax[0].plot([4.8,5.2],[np.mean(n2nyqO3[' SSIM']),np.mean(n2nyqO3[' SSIM'])],color=p7[0].get_color(),linewidth=3)
    ax[0].plot([5.8,6.2],[np.mean(n2nyqO4[' SSIM']),np.mean(n2nyqO4[' SSIM'])],color=p8[0].get_color(),linewidth=3)
    ax[0].plot([6.8,7.2],[np.mean(n2vTarg[' SSIM']),np.mean(n2vTarg[' SSIM'])],color=p3[0].get_color(),linewidth=3)
    ax[0].plot([7.8,8.2],[np.mean(n2nyqSingle[' SSIM']),np.mean(n2nyqSingle[' SSIM'])],color=p4[0].get_color(),linewidth=3)
    
    ax[0].set_ylim([0.65,1])
    ax[0].set_ylabel('SSIM')
    ax[0].set_xticks([1,2,3,4,5,6,7,8])
    ax[0].set_xticklabels(['Supervised','noise2noise', 'noise2Nyq','Nyq/2','Nyq/3','Nyq/4','noise2void','line2line'],rotation=20,ha='right')
    ax[0].set_title('Phantom Denoising Similarity')
    
    p1=ax[1].plot(np.ones(datShape)+np.random.normal(0,jitterStd,datShape),cleanTarg[' MSE'],'.')
    p2=ax[1].plot(np.ones(datShape+1)*2+np.random.normal(0,jitterStd,datShape+1),noisyTarg[' MSE'],'.')
    p5=ax[1].plot(np.ones(datShape)*3+np.random.normal(0,jitterStd,datShape),nextTarg[' MSE'],'.')
    p6=ax[1].plot(np.ones(o2Shape)*4+np.random.normal(0,jitterStd,o2Shape),n2nyqO2[' MSE'],'.')
    p7=ax[1].plot(np.ones(o3Shape)*5+np.random.normal(0,jitterStd,o3Shape),n2nyqO3[' MSE'],'.')
    p8=ax[1].plot(np.ones(o4Shape)*6+np.random.normal(0,jitterStd,o4Shape),n2nyqO4[' MSE'],'.')
    p3=ax[1].plot(np.ones(datShape)*7+np.random.normal(0,jitterStd,datShape),n2vTarg[' MSE'],'.')
    p4=ax[1].plot(np.ones(datShape)*8+np.random.normal(0,jitterStd,datShape),n2nyqSingle[' MSE'],'.')
   
    ax[1].plot([0.8,1.2],[np.mean(cleanTarg[' MSE']),np.mean(cleanTarg[' MSE'])],color=p1[0].get_color(),linewidth=3)
    ax[1].plot([1.8,2.2],[np.mean(noisyTarg[' MSE']),np.mean(noisyTarg[' MSE'])],color=p2[0].get_color(),linewidth=3)
    ax[1].plot([2.8,3.2],[np.mean(nextTarg[' MSE']),np.mean(nextTarg[' MSE'])],color=p5[0].get_color(),linewidth=3)
    ax[1].plot([3.8,4.2],[np.mean(n2nyqO2[' MSE']),np.mean(n2nyqO2[' MSE'])],color=p6[0].get_color(),linewidth=3)
    ax[1].plot([4.8,5.2],[np.mean(n2nyqO3[' MSE']),np.mean(n2nyqO3[' MSE'])],color=p7[0].get_color(),linewidth=3)
    ax[1].plot([5.8,6.2],[np.mean(n2nyqO4[' MSE']),np.mean(n2nyqO4[' MSE'])],color=p8[0].get_color(),linewidth=3)
    ax[1].plot([6.8,7.2],[np.mean(n2vTarg[' MSE']),np.mean(n2vTarg[' MSE'])],color=p3[0].get_color(),linewidth=3)
    ax[1].plot([7.8,8.2],[np.mean(n2nyqSingle[' MSE']),np.mean(n2nyqSingle[' MSE'])],color=p4[0].get_color(),linewidth=3)
    
    ax[1].set_ylim([0,45])
    ax[1].set_ylabel('MSE')
    ax[1].set_xticks([1,2,3,4,5,6,7,8])
    ax[1].set_xticklabels(['Supervised','noise2noise', 'noise2Nyq','Nyq/2','Nyq/3','Nyq/4','noise2void','line2line'],rotation=20,ha='right')
    ax[1].set_title('Phantom Denoising Error')
    plt.savefig('../../results/phantom/figures/PhantomQuant_L1Random.png')
    
    #Now use seaborn to make cool swarmplots
    fig,ax = plt.subplots(1,2,figsize=(16,8))
    sns.swarmplot(x='Method',y='PSNR',data=phantomDF,ax=ax[0],size=3.5)
    # plot the mean line
    sns.boxplot(showmeans=True,
                meanline=True,
                meanprops={'color': 'k', 'ls': '-', 'lw': 4},
                medianprops={'visible': False},
                whiskerprops={'visible': False},
                zorder=10,
                x="Method",
                y="PSNR",
                data=phantomDF,
                showfliers=False,
                showbox=False,
                showcaps=False,
                width=0.6,
                ax=ax[0])
    sns.swarmplot(x='Method',y='SSIM',data=phantomDF,ax=ax[1],size=3.5)
    # plot the mean line
    sns.boxplot(showmeans=True,
                meanline=True,
                meanprops={'color': 'k', 'ls': '-', 'lw': 4},
                medianprops={'visible': False},
                whiskerprops={'visible': False},
                zorder=10,
                x="Method",
                y="SSIM",
                data=phantomDF,
                showfliers=False,
                showbox=False,
                showcaps=False,
                width=0.6,
                ax=ax[1])
    ax[0].set_xticklabels(ax[0].get_xticklabels(),rotation = 20,ha='right')
    ax[1].set_xticklabels(ax[1].get_xticklabels(),rotation = 20,ha='right')
    ax[0].set_title('Phantom Denoising Peak SNR')
    ax[1].set_title('Phantom Denoising Similarity')
    plt.tight_layout()
    plt.savefig('../../results/phantom/figures/phantomQuantSwarm.png')
    
    
   