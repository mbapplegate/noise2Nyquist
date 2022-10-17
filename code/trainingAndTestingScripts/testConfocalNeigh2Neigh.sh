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
python trainOOP_CL.py --dataType confocal --dataPath /home/matthew/Documents/datasets/Denoising_Planaria/patches064/ \
--csvDir ../results/confocal/dataSplits/ --saveDir ../results/confocal/neigh2neigh/2022-10-14-14-35 --randomOrder --loss_fn l1 --batch_size 32 --numTrainingIms 65536 --numValidIms 1024 --numTestIms 0 --numEpochs 150 --startSplit 1 --splitsToTrain 9 --nfolds 10 --testOnly --singleImage

echo "Done. Hopefully there were no errors"
