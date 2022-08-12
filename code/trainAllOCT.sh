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

#Noise2Void
python trainOOP_CL.py --dataType oct --dataPath ~/Documents/datasets/OCT_Denoise/Data/patches064 \
--csvDir ./dataSplits/oct --saveDir ../results/oct/ --nextImFlag \
--randomOrder --loss_fn l1 --batch_size 64 --numTrainingIms 65536 --numValidIms 1024 --numTestIms 0 \
--numEpochs 150 --startSplit 0 --splitsToTrain 10 --nfolds 10 --calcStats --singleImage --noise2Void

#Noise2Nyquist (next image as target)
python trainOOP_CL.py --dataType oct --dataPath ~/Documents/datasets/OCT_Denoise/Data/patches064 \
--csvDir ./dataSplits/oct --saveDir ../results/oct/ --nextImFlag \
--randomOrder --loss_fn l1 --batch_size 64 --numTrainingIms 65536 --numValidIms 1024 --numTestIms 0 \
--numEpochs 150 --startSplit 0 --splitsToTrain 10 --nfolds 10 --calcStats

#line2line
python trainOOP_CL.py --dataType oct --dataPath ~/Documents/datasets/OCT_Denoise/Data/patches064 \
--csvDir ./dataSplits/oct --saveDir ../results/oct/ --nextImFlag --singleImage \
--randomOrder --loss_fn l1 --batch_size 64 --numTrainingIms 65536 --numValidIms 1024 --numTestIms 0 \
--numEpochs 150 --startSplit 0 --splitsToTrain 10 --nfolds 10 --calcStats

echo "Done. Hopefully there were no errors"
