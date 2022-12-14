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
python trainNeigh2Neigh.py --noisetype none --data_dir /home/matthew/Documents/datasets/Denoising_Planaria/patches064/condition2/current --val_dir /home/matthew/Documents/datasets/Denoising_Planaria/patches064/clean --save_model_path ../results/confocal/neigh2neigh --log_name Conf_split0 --n_channel 1 --n_snapshot 10  --n_epoch 60 --patchsize 64 --dataType confocal --fold_number 0

echo "Done. Hopefully there were no errors"
