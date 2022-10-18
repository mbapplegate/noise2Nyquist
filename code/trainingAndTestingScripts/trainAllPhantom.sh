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

#Clean Targets
python ../trainOOP_CL.py --dataType phantom --saveDir ../../results/phantom --nextImFlag \
--cleanTargets --loss_fn l1 --batch_size 4 --calcStats --phantomNoiseLevel 45 --sampMult 1 --randomOrder \
--trainPhantomFile ../HRPhantomData/SheppLoganPhan.mat --valPhantomFile ../HRPhantomData/YuYeWangPhan.mat --numEpochs 150 --splitsToTrain 10 --startSplit 0 --nfolds 10 

#Noisy Targets
python ../trainOOP_CL.py --dataType phantom --saveDir ../../results/phantom \
--loss_fn l1 --batch_size 4 --calcStats --phantomNoiseLevel 45 --sampMult 1 --randomOrder \
--trainPhantomFile ../HRPhantomData/SheppLoganPhan.mat --valPhantomFile ../HRPhantomData/YuYeWangPhan.mat --numEpochs 150 --splitsToTrain 10 --startSplit 0 --nfolds 10

#Next Targets (Nyquist)
python ../trainOOP_CL.py --dataType phantom --saveDir ../../results/phantom --nextImFlag \
--loss_fn l1 --batch_size 4 --calcStats --phantomNoiseLevel 45 --sampMult 0.5 --randomOrder \
--trainPhantomFile ../HRPhantomData/SheppLoganPhan.mat --valPhantomFile ../HRPhantomData/YuYeWangPhan.mat --numEpochs 150 --splitsToTrain 10 --startSplit 0 --nfolds 10

#Next Targets (Nyquist/2)
python ../trainOOP_CL.py --dataType phantom --saveDir ../../results/phantom --nextImFlag \
--loss_fn l1 --batch_size 4 --calcStats --phantomNoiseLevel 45 --sampMult 2 --randomOrder \
--trainPhantomFile ../HRPhantomData/SheppLoganPhan.mat --valPhantomFile ../HRPhantomData/YuYeWangPhan.mat --numEpochs 150 --splitsToTrain 10 --startSplit 0 --nfolds 10

#Next Targets (Nyquist/3)
python ../trainOOP_CL.py --dataType phantom --saveDir ../../results/phantom --nextImFlag \
--loss_fn l1 --batch_size 4 --calcStats --phantomNoiseLevel 45 --sampMult 3 --randomOrder \
--trainPhantomFile ./HRPhantomData/SheppLoganPhan.mat --valPhantomFile ./HRPhantomData/YuYeWangPhan.mat --numEpochs 150 --splitsToTrain 10 --startSplit 0 --nfolds 10

#Next Targets (Nyquist/4)
python ../trainOOP_CL.py --dataType phantom --saveDir ../../results/phantom --nextImFlag \
--loss_fn l1 --batch_size 4 --calcStats --phantomNoiseLevel 45 --sampMult 4 --randomOrder\
--trainPhantomFile ../HRPhantomData/SheppLoganPhan.mat --valPhantomFile ../HRPhantomData/YuYeWangPhan.mat --numEpochs 150 --splitsToTrain 10 --startSplit 0 --nfolds 10

#line2line
#python trainOOP_CL.py --dataType phantom --saveDir ../results/phantom --nextImFlag \
#--loss_fn l1 --batch_size 4 --calcStats --phantomNoiseLevel 45 --sampMult 1 --singleImage --randomOrder\
#--trainPhantomFile ./HRPhantomData/SheppLoganPhan.mat --valPhantomFile ./HRPhantomData/YuYeWangPhan.mat --numEpochs 150 --splitsToTrain 10 --startSplit 0

#noise2void
python ../trainOOP_CL.py --dataType phantom --saveDir ../../results/phantom --nextImFlag \
--loss_fn l1 --batch_size 4 --calcStats --phantomNoiseLevel 45 --sampMult 1 --singleImage --noise2VoidFlag --randomOrder --trainPhantomFile ../HRPhantomData/SheppLoganPhan.mat --valPhantomFile ../HRPhantomData/YuYeWangPhan.mat --numEpochs 150 --splitsToTrain 10 --startSplit 0 --nfolds 10

echo "Done. Hopefully there were no errors"
