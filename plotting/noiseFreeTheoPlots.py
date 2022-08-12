#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 26 16:42:16 2022

@author: matthew
"""
import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit

#Set up matplotlib font sizes
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

#1-D gaussian
def gauss(x,loc,std):
    a = 1/(np.sqrt(2*np.pi)*std)
    return a * np.exp(-((.5*(x-loc)**2)/(std**2)))

#Function to use with curve_fit to fit a gaussian
def fitGauss(x,offset,amp,loc,std):
    return offset+amp*np.exp(-((.5*(x-loc)**2)/(std**2)))

if __name__ == '__main__':
  #Set ground truth -- Doesn't change for each trial
  trueX = np.arange(1000)

  #Set up simulated point spread function
  nyquistSampling=16
  sampMult = 1
  #See Eq. 1 in the paper for why
  psfSz = 4*nyquistSampling
  psfFWHM = (nyquistSampling * 2 * 0.51)/(0.61)
  psfStd = psfFWHM / (2*np.sqrt(2*np.log(2)))
  rayleighDist = 0.51/0.61 * psfFWHM
  #1d psf
  psf = gauss(np.arange(-psfSz,(psfSz)),0,psfStd)
  
  #########################
  ########################
  #Analysis of Edge
  ########################
  #######################
  #Generate simulated image at high resolution
  trueY = np.zeros(trueX.shape)
  #Start with a step
  trueY[500::]=1
  #Make an image by convolving
  imgTrueY = np.convolve(trueY,psf,'valid')
  #Valid X values
  validX = trueX[psfSz-1:1000-psfSz]
  imgTrueY /= np.max(imgTrueY) #Normalize to 1
      
  for startLoc in [7,13]: #I found that these starting locations give the best and the worst results
      #Sample X and Y
      sampledX = validX[startLoc::nyquistSampling*sampMult]
      sampledY = imgTrueY[startLoc::nyquistSampling*sampMult]
      #Pre-allocate for running average/median
      truncX = np.zeros(len(sampledX)-2)
      avgY = np.zeros(len(sampledY)-2)
      medY = np.zeros(len(sampledY)-2)
      #Calculate running average and running median
      for x in range(1,len(sampledX)-1):
          truncX[x-1] = sampledX[x]
          avgY[x-1] = np.mean([sampledY[x-1],sampledY[x],sampledY[x+1]])
          medY[x-1] = np.median([sampledY[x-1],sampledY[x],sampledY[x+1]])
      #Get the maximum error (pixel-wise)
      maxErrAvg = np.max(np.abs((avgY-sampledY[1:-1])))
      maxErrMed = np.max(np.abs(medY-sampledY[1:-1]))
      
      print('Avg %.3f, Med %.3f'%(maxErrAvg,maxErrMed))
      #Plot the step
      fig,ax=plt.subplots(1,1,figsize=(6.6,4.8))
      ax.plot(trueX,trueY,'--',label='True Structure',linewidth=3)
      l=ax.plot(validX,imgTrueY,label='Image',linewidth=3)
      ax.plot(sampledX,sampledY,'3',label='Sampled Image',markersize=12,markeredgewidth=2,color=l[0].get_color())
      ax.plot(truncX,avgY,'o',label='Avg.',markerfacecolor='none',markersize=10,markeredgewidth=2)
      ax.plot(truncX,medY,'4',label='Median',markersize=12,markeredgewidth=2)
      ax.set_xlim([350,650])
      ax.set_xlabel('Position (AU)')
      ax.set_ylabel('Intensity (AU)')
      ax.set_title('Expected Performance for Edge')
      ax.legend()
      # ax[1].plot(np.arange(-psfSz,(psfSz)),psf)
      # ax[1].set_xlabel('Position (AU)')
      # ax[1].set_ylabel('Intensity (AU)')
      # ax[1].set_title('Gaussian Point Spread Function')
      plt.tight_layout()
      if startLoc == 7:
          plt.savefig('../../communications/paper/figures/theoretical/noiseFreeImageOfEdge_maxErr.png')
      else:
          plt.savefig('../../communications/paper/figures/theoretical/noiseFreeImageOfEdge_minErr.png')  
  
  ###################
  ####################
  #Analysis of a point
  #####################
  #######################  
  #Generate simulated image at high resolution
  trueY = np.zeros(trueX.shape)
  #trueY[(trueX>=500-3*nyquistSampling) & (trueX <=500+3*nyquistSampling)]=1
  #Start with a step
  trueY[500]=1
 
  sampMult=1 #Nyquist rate
  imgTrueY = np.convolve(trueY,psf,'valid')
  validX = trueX[psfSz-1:1000-psfSz]
  imgTrueY /= np.max(imgTrueY) #Normalize to 1
  fitTrue,_= curve_fit(fitGauss,validX,imgTrueY,p0=[0,1,500,30])
  distFromPt = np.zeros(nyquistSampling*sampMult)
  avgStds = np.zeros(nyquistSampling*sampMult)
  medStds = np.zeros(nyquistSampling*sampMult)
  avgErr = np.zeros(nyquistSampling*sampMult)
  medErr = np.zeros(nyquistSampling*sampMult)
  startLoc = range(nyquistSampling*sampMult)
  #Iterate over all starting locations
  for s in startLoc:
      #Sample the image
      sampledX = validX[s::nyquistSampling*sampMult]
      sampledY = imgTrueY[s::nyquistSampling*sampMult]
      truncX = np.zeros(len(sampledX)-2)
      avgY = np.zeros(len(sampledY)-2)
      medY = np.zeros(len(sampledY)-2)
      #Do running average/median
      for x in range(1,len(sampledX)-1):
        truncX[x-1] = sampledX[x]
        avgY[x-1] = np.mean([sampledY[x-1],sampledY[x],sampledY[x+1]])
        medY[x-1] = np.median([sampledY[x-1],sampledY[x],sampledY[x+1]])
      #Get max error for this starting location
      maxErrAvg = np.max(np.sqrt((avgY-sampledY[1:-1])**2))     
      maxErrMed = np.max(np.sqrt((medY-sampledY[1:-1])**2))
      avgErr[s] = maxErrAvg
      medErr[s] = maxErrMed
     # Fit gaussian to the two methods
      fitAvg,_ = curve_fit(fitGauss,truncX,avgY,p0=[0,1,500,20])
      fitMed,_ = curve_fit(fitGauss,truncX,medY,p0=[0,1,500,20])
      
      #Save widths
      avgStds[s] = fitAvg[-1]
      medStds[s] = fitMed[-1]
      idx = (np.abs(truncX-500)).argmin()
      distFromPt[s] = truncX[idx]-500
  
  ##################
  #Same as above but at Nyquist/2
  sampMult=2
  distFromPt2 = np.zeros(int(nyquistSampling*sampMult))
  avgStds2 = np.zeros(int(nyquistSampling*sampMult))
  medStds2 = np.zeros(int(nyquistSampling*sampMult))
  avgErr2 = np.zeros(int(nyquistSampling*sampMult))
  medErr2 = np.zeros(int(nyquistSampling*sampMult))
  startLoc2 = range(int(nyquistSampling*sampMult))
  for s in startLoc2:
      sampledX = validX[s::int(nyquistSampling*sampMult)]
      sampledY = imgTrueY[s::int(nyquistSampling*sampMult)]
      truncX = np.zeros(len(sampledX)-2)
      avgY = np.zeros(len(sampledY)-2)
      medY = np.zeros(len(sampledY)-2)
      for x in range(1,len(sampledX)-1):
        truncX[x-1] = sampledX[x]
        avgY[x-1] = np.mean([sampledY[x-1],sampledY[x],sampledY[x+1]])
        medY[x-1] = np.median([sampledY[x-1],sampledY[x],sampledY[x+1]])
      maxErrAvg = np.max(np.sqrt((avgY-sampledY[1:-1])**2))     
      maxErrMed = np.max(np.sqrt((medY-sampledY[1:-1])**2))
      avgErr2[s] = maxErrAvg
      medErr2[s] = maxErrMed
     # print('SL: %02d Avg %.3f, Med %.3f'%(startLoc,maxErrAvg,maxErrMed))
      fitAvg,_ = curve_fit(fitGauss,truncX,avgY,p0=[0,1,500,20])
      fitMed,_ = curve_fit(fitGauss,truncX,medY,p0=[0,1,500,20])
      #print('SL: %02d Avg Std: %.3f, Med Std: %3f'%(startLoc,fitAvg[-1],fitMed[-1]))
      avgStds2[s] = fitAvg[-1]
      medStds2[s] = fitMed[-1]
      idx = (np.abs(truncX-500)).argmin()
      distFromPt2[s] = truncX[idx]-500
  ################################
  #Same as above, but at 2x Nyquist   
  sampMult=.5
  distFromPt3 = np.zeros(int(nyquistSampling*sampMult))
  avgStds3 = np.zeros(int(nyquistSampling*sampMult))
  medStds3 = np.zeros(int(nyquistSampling*sampMult))
  avgErr3 = np.zeros(int(nyquistSampling*sampMult))
  medErr3 = np.zeros(int(nyquistSampling*sampMult))
  startLoc3 = range(int(nyquistSampling*sampMult))
  for s in startLoc3:
      sampledX = validX[s::int(nyquistSampling*sampMult)]
      sampledY = imgTrueY[s::int(nyquistSampling*sampMult)]
      truncX = np.zeros(len(sampledX)-2)
      avgY = np.zeros(len(sampledY)-2)
      medY = np.zeros(len(sampledY)-2)
      for x in range(1,len(sampledX)-1):
          truncX[x-1] = sampledX[x]
          avgY[x-1] = np.mean([sampledY[x-1],sampledY[x],sampledY[x+1]])
          medY[x-1] = np.median([sampledY[x-1],sampledY[x],sampledY[x+1]])
          maxErrAvg = np.max(np.sqrt((avgY-sampledY[1:-1])**2))     
          maxErrMed = np.max(np.sqrt((medY-sampledY[1:-1])**2))
          avgErr3[s] = maxErrAvg
          medErr3[s] = maxErrMed
      # print('SL: %02d Avg %.3f, Med %.3f'%(startLoc,maxErrAvg,maxErrMed))
      fitAvg,_ = curve_fit(fitGauss,truncX,avgY,p0=[0,1,500,20])
      fitMed,_ = curve_fit(fitGauss,truncX,medY,p0=[0,1,500,20])
       #print('SL: %02d Avg Std: %.3f, Med Std: %3f'%(startLoc,fitAvg[-1],fitMed[-1]))
      avgStds3[s] = fitAvg[-1]
      medStds3[s] = fitMed[-1]
      idx = (np.abs(truncX-500)).argmin()
      distFromPt3[s] = truncX[idx]-500
  
  #Gather data to plot error as a function of distance true point is from sampled point
  nyqDatMed = np.vstack((distFromPt,medStds))
  nyqDatAvg = np.vstack((distFromPt,avgStds))
  nyq2DatMed = np.vstack((distFromPt2,medStds2))
  nyq2DatAvg = np.vstack((distFromPt2,avgStds2))
  nyq3DatMed = np.vstack((distFromPt3,medStds3))
  nyq3DatAvg = np.vstack((distFromPt3,avgStds3))
  
  ind1 = np.argsort(nyqDatMed[0,:])
  ind2 = np.argsort(nyq2DatMed[0,:])
  ind3 = np.argsort(nyq3DatMed[0,:])
  
  nyqDatMed = nyqDatMed[:,ind1]
  nyqDatAvg = nyqDatAvg[:,ind1]
  nyq2DatMed = nyq2DatMed[:,ind2]
  nyq2DatAvg = nyq2DatAvg[:,ind2]
  nyq3DatMed = nyq3DatMed[:,ind3]
  nyq3DatAvg = nyq3DatAvg[:,ind3]
  
  #Plot Reconstructed PSF width as function of sampling rate and L2/L2 loss
  fig,ax=plt.subplots(1,1,figsize=(6.6,4.8))
  nm=ax.plot(nyqDatMed[0,:],nyqDatMed[1,:],'o-',markersize=8,markeredgewidth=2,linewidth=3,color='tab:blue')
  n2m=ax.plot(nyq2DatMed[0,:],nyq2DatMed[1,:],'o-',markersize=8,markeredgewidth=2,linewidth=3,color='firebrick')
  n3m=ax.plot(nyq3DatMed[0,:],nyq3DatMed[1,:],'o-',markersize=8,markeredgewidth=2,linewidth=3,color='tab:green')
  ax.plot(nyqDatAvg[0,:],nyqDatAvg[1,:],'x-',markersize=8,markeredgewidth=2,linewidth=3,color=nm[0].get_color())
  ax.plot(nyq2DatAvg[0,:],nyq2DatAvg[1,:],'x-',markersize=8,markeredgewidth=2,linewidth=3,color=n2m[0].get_color())
  ax.plot(nyq3DatAvg[0,:],nyq3DatAvg[1,:],'x-',markersize=8,markeredgewidth=2,linewidth=3,color=n3m[0].get_color())
  ax.plot([None],[None],'kx',markersize=8,markeredgewidth=2,label='$L_2$')
  ax.plot([None],[None],'ko',markersize=8,markeredgewidth=2,label='$L_1$')
  ax.plot([None],[None],'s',markersize=12,color=n2m[0].get_color(),label='Nyquist/2')
  ax.plot([None],[None],'s',markersize=12,color=nm[0].get_color(),label='Nyquist')
  ax.plot([None],[None],'s',markersize=12,color=n3m[0].get_color(),label = '2x Nyquist')
  ax.plot([-17,17],[fitTrue[-1],fitTrue[-1]],'k--',linewidth=3,label='True')
  ax.legend(bbox_to_anchor=(0.65,0.925),loc='upper left')
  ax.set_xlim([-17,17])
  ax.set_xlabel('Distance of sample to point (AU)')
  ax.set_ylabel('Fit Std. Dev. (AU)')
  ax.set_title('Point Spread Function Width')
  plt.tight_layout()
  plt.savefig('../../communications/paper/figures/theoretical/PSF_width.png')
  
  #Make plots of best and worst performing point at the Nyquist rate
  sampMult=1
  bestShift = np.argmin(medStds2)
  worstShift = np.argmax(medStds2)
  for s in [worstShift,bestShift]:
      sampledX = validX[s::nyquistSampling*sampMult]
      sampledY = imgTrueY[s::nyquistSampling*sampMult]
      truncX = np.zeros(len(sampledX)-2)
      avgY = np.zeros(len(sampledY)-2)
      medY = np.zeros(len(sampledY)-2)
      for x in range(1,len(sampledX)-1):
        truncX[x-1] = sampledX[x]
        avgY[x-1] = np.mean([sampledY[x-1],sampledY[x],sampledY[x+1]])
        medY[x-1] = np.median([sampledY[x-1],sampledY[x],sampledY[x+1]])
      maxErrAvg = np.max(np.sqrt((avgY-sampledY[1:-1])**2))     
      maxErrMed = np.max(np.sqrt((medY-sampledY[1:-1])**2))
      print('Avg %.3f, Med %.3f'%(maxErrAvg,maxErrMed))
      fitAvg,_ = curve_fit(fitGauss,truncX,avgY,p0=[0,1,500,20])
      fitMed,_ = curve_fit(fitGauss,truncX,medY,p0=[0,1,500,20])
      print('Avg Std: %.3f, Med Std: %3f'%(fitAvg[-1],fitMed[-1]))
      fig,ax=plt.subplots(1,1,figsize=(6.6,4.8))
      ax.plot(trueX,trueY,'--',label='True Structure',linewidth=3)
      l=ax.plot(validX,imgTrueY,label='Image',linewidth=3)
      ax.plot(sampledX,sampledY,'3',label='Sampled Image',markersize=10,markeredgewidth=2,color=l[0].get_color())
      a=ax.plot(truncX,avgY,'o',label='Avg.',markerfacecolor='none',markersize=10,markeredgewidth=2)
      m=ax.plot(truncX,medY,'4',label='Median',markersize=10,markeredgewidth=2)
      ax.plot(trueX,fitGauss(trueX,*fitAvg),color=a[0].get_color(),linewidth=3)
      ax.plot(trueX,fitGauss(trueX,*fitMed),color=m[0].get_color(),linewidth=3)
      ax.set_xlim([400,600])
      ax.set_xlabel('Position (AU)')
      ax.set_ylabel('Intensity (AU)')
     
      # ax[1].plot(np.arange(-psfSz,(psfSz)),psf)
      # ax[1].set_xlabel('Position (AU)')
      # ax[1].set_ylabel('Intensity (AU)')
      # ax[1].set_title('Gaussian Point Spread Function')
     
      if s==worstShift:
          ax.set_title('Worst Performance for Point')
          ax.legend()
          plt.tight_layout()
          plt.savefig('../../communications/paper/figures/theoretical/noiseFreeImageOfPoint_maxErr.png')
      else:
          ax.set_title('Best Performance for Point')
          ax.legend()
          plt.tight_layout()
          plt.savefig('../../communications/paper/figures/theoretical/noiseFreeImageOfPoint_minErr.png')
          
  ###############################
  #Here I want to make a plot of structure width vs Error
  #################################################
  structureWidthsPx=np.arange(1,201,step=4)
  structureWidthFWHM = np.array(structureWidthsPx)/psfFWHM
  sampMults = [0.5,1,2]
  errAvg = np.zeros((len(sampMults),len(structureWidthsPx)))
  errMed = np.zeros(errAvg.shape)
  for i,w in enumerate(structureWidthsPx):
      trueY = np.zeros(trueX.shape)
      trueY[500-int(w/2):500+int(np.ceil(w/2))]=1
      sampMult=1
      imgTrueY = np.convolve(trueY,psf,'valid')
      validX = trueX[psfSz-1:1000-psfSz]
      imgTrueY /= np.max(imgTrueY) #Normalize to 1
      for j,m in enumerate(sampMults):
          sampErrAvg = np.zeros(int(nyquistSampling*m))
          sampErrMed = np.zeros(int(nyquistSampling*m))
          for s in range(int(nyquistSampling*m)):
              sampledX = validX[s::int(nyquistSampling*m)]
              sampledY = imgTrueY[s::int(nyquistSampling*m)]
              truncX = np.zeros(len(sampledX)-2)
              avgY = np.zeros(len(sampledY)-2)
              medY = np.zeros(len(sampledY)-2)
              for x in range(1,len(sampledX)-1):
                  truncX[x-1] = sampledX[x]
                  avgY[x-1] = np.mean([sampledY[x-1],sampledY[x],sampledY[x+1]])
                  medY[x-1] = np.median([sampledY[x-1],sampledY[x],sampledY[x+1]])
              sampErrAvg[s] = np.mean((avgY-sampledY[1:-1])**2)
              sampErrMed[s] = np.mean((medY-sampledY[1:-1])**2)
          errAvg[j,i] = np.max(sampErrAvg)
          errMed[j,i] = np.max(sampErrMed)
  fig,ax = plt.subplots(1,1)
  l1=ax.semilogy(structureWidthFWHM,errAvg[1,:],label='L2')
  l2=ax.plot(structureWidthFWHM,errMed[1,:],label='L1')
  ax.plot(structureWidthFWHM,errAvg[0,:],'--',color=l1[0].get_color())
  ax.plot(structureWidthFWHM,errMed[0,:],'--',color=l2[0].get_color())
  ax.plot(structureWidthFWHM,errAvg[2,:],':',color=l1[0].get_color())
  ax.plot(structureWidthFWHM,errMed[2,:],':',color=l2[0].get_color())
  ax.legend()
          
  ###############################
  #Here I want to make a plot of sampling rate vs error
  ##################################
  structureWidthsPx = [1]
  structureWidthFWHM = np.array(structureWidthsPx)/psfFWHM
  sampleEvery = np.array(range(5,34))
  errAvg = np.zeros((2,len(sampleEvery)))
  errMed = np.zeros(errAvg.shape)
  allAvgErr = []
  allMedErr = []
  #Only executes once
  for i,w in enumerate(structureWidthsPx):
      trueY = np.zeros(trueX.shape)
      trueY[500-int(w/2):500+int(np.ceil(w/2))]=1
      imgTrueY = np.convolve(trueY,psf,'valid')
      validX = trueX[psfSz-1:1000-psfSz]
      imgTrueY /= np.max(imgTrueY) #Normalize to 1
      #Change the sampling rate from every 5 HR pixels to every 33 HR pixels
      for j,m in enumerate(sampleEvery):
          sampErrAvg = np.zeros(m)
          sampErrMed = np.zeros(m)
          for s in range(m):
              sampledX = validX[s::m] #Sample
              sampledY = imgTrueY[s::m]
              truncX = np.zeros(len(sampledX)-2)
              avgY = np.zeros(len(sampledY)-2)
              medY = np.zeros(len(sampledY)-2)
              #Process sampled image
              for x in range(1,len(sampledX)-1):
                  truncX[x-1] = sampledX[x]
                  avgY[x-1] = np.mean([sampledY[x-1],sampledY[x],sampledY[x+1]])
                  medY[x-1] = np.median([sampledY[x-1],sampledY[x],sampledY[x+1]])
              #Get the standard deviation of the fit Gaussian
              fitAvg,_ = curve_fit(fitGauss,truncX,avgY,p0=[0,1,500,20])
              fitMed,_ = curve_fit(fitGauss,truncX,medY,p0=[0,1,500,20])
              sampErrAvg[s] = fitAvg[-1]#np.mean((avgY-sampledY[1:-1])**2)
              sampErrMed[s] = fitMed[-1]#np.mean((medY-sampledY[1:-1])**2)
          allAvgErr.append(sampErrAvg)
          allMedErr.append(sampErrMed)
          errAvg[0,j] = np.max(sampErrAvg)
          errMed[0,j] = np.max(sampErrMed)
          errAvg[1,j] = np.min(sampErrAvg)
          errMed[1,j] = np.min(sampErrMed)

  ###################
  #Plot the sample rate vs reconstructed PSF width for L2 and L1
  ################################
  fig,ax = plt.subplots(2,1,sharex=True,sharey=True,figsize=(6.6,4.8))
  for q in range(len(allAvgErr)):
      ax[0].plot(np.ones(len(allAvgErr[q]))*nyquistSampling/sampleEvery[q],allAvgErr[q]/psfStd,'.',color='lightgray',markersize=4)
      ax[1].plot(np.ones(len(allMedErr[q]))*nyquistSampling/sampleEvery[q],allMedErr[q]/psfStd,'.',color='lightgray',markersize=4)
  l1=ax[0].plot(nyquistSampling/sampleEvery,(errAvg[0,:])/psfStd,'-',color='tab:green',linewidth=3)
  l2=ax[1].plot(nyquistSampling/sampleEvery,(errMed[0,:])/psfStd,'-',color='firebrick',linewidth=3)
  ax[0].plot(nyquistSampling/sampleEvery,(errAvg[1,:])/psfStd,'--',color='tab:green',linewidth=3)
  ax[1].plot(nyquistSampling/sampleEvery,(errMed[1,:])/psfStd,'--',color='firebrick',linewidth=3)
  ax[0].plot([None],[None],'k--',label='Best Case',linewidth=3)
  ax[0].plot([None],[None],'k-',label='Worst Case',linewidth=3)
  ax[0].plot([None],[None],'.',color='lightgray',markersize=4,label='Other cases')
 
  ax[0].grid()
  ax[1].grid()
 
  ax[1].set_xlabel('(Samp. Rate)/(Nyq. Rate)')
  fig.supylabel(r'(FWHM)/(PSF$_{FWHM}$)')
  ax[0].set_title(r'L$_2$ Loss')
  ax[1].set_title(r'L$_1$ Loss')
  ax[0].legend()
  plt.tight_layout()
  plt.savefig('../../communications/paper/figures/theoretical/samplingVsError.png')
