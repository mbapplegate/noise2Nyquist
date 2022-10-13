#!/bin/bash

#Script to train multiple versions of the denoising algorithm sequentially (good for overnight training)
#
#Here are some less, used options that might be helpful later
#--nextImFlag - train with next (or previous) image
#--singleImage - train with just a single image
#--noise2VoidFlag - Use masking method a la noise2Void
#--phantomNoiseLevel - SD of noise to add to phantom measurements (ignored if datatype isn't phantom)
#--nyquistSampling - Number of pixels that is the nyquist frequency
#--sampMult - Multiply the nyquist rate 2=half the rate, 0.5 = twice the rate
#--initLR - Initial learning rate

echo "Running training script..."
##CT
#Supervised
python trainOOP_CL.py --dataType ct --dataPath /home/matthew/Documents/datasets/lowDoseCT/fullFrames/ \
--csvDir ~/Documents/n2NyquistGit/noise2Nyquist/results/ct/dataSplits/ --saveDir ../results/ct/neigh2neigh/2022-10-12-18-47 --randomOrder --loss_fn l1 --batch_size 32 --numTrainingIms 65536 --numValidIms 1024 --numTestIms 0 --numEpochs 150 --startSplit 1 --splitsToTrain 9 --nfolds 10 --testOnly --singleImage

echo "Done. Hopefully there were no errors"
