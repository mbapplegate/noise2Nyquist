#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 17 15:33:21 2022

@author: matthew
"""

import utils.utilsOOP as utils
from matplotlib import pyplot as plt
from PIL import Image

import numpy as np

if __name__ == '__main__':
    datFile = '/home/matthew/Documents/datasets/Denoising_Planaria/fullFrames/condition2/03/frame059.npy'
    dat = np.load(datFile)
   
    modFile = '/home/matthew/Documents/noise2Nyquist_results/confocal/neigh2neigh/2022-10-14-14-35/05/modelWeights_last.pth'
    model = utils.loadModel(modFile,'neigh2neigh')
    
    model.to('cuda')
    despeckPatch=utils.despeckleMosaic(dat,64,32,model,'cuda')
    cleanFrame = utils.stitchMosaic(despeckPatch,32)
    fig,ax = plt.subplots(1,2)
    ax[0].imshow(dat)
    ax[1].imshow(cleanFrame)
    
    percs = np.percentile(cleanFrame,[2,99])
    
    normIm = (cleanFrame-percs[0])/(percs[1]-percs[0])
    im8bit = np.clip(np.round(cleanFrame*255),0,255).astype('uint8')
    
    im=Image.fromarray(im8bit)
    im.save('../communications/paper/figures/confocal/neighbor2neighborFrame.png')
    
    
    