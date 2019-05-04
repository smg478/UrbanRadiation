#!/bin/bash
###!/usr/bin/env python3
##!/bin/bash python3
######################################################
# Competition data
#####################################################
# bash test.sh /data/testing/ solution.csv

# Data preparation and statistics
#python 01_make_slice_data.py "$@" &
#wait
#python 02_make_features.py "$@" &
#wait



# Download train statistics file from google drive, if required (e.g. first time prediction)
if [ -e /wdata/train_feature_bin_30_slice.csv ]
then
    echo "train statistics file exists, skipping download..."
else
    echo "train statistics file does not exist, downloading from google drive..."
    gdown "https://drive.google.com/uc?id=16X16QadR-hPwjcaykZs0VwrfxqVP95m5"
    mv /work/train_feature_bin_30_slice.csv /wdata
fi


# Prediction
python 06_predict_25.py "$@" &
python 07_predict_3000.py "$@" &
python 08_predict_6000.py "$@" &
wait

# ensemble predictions
python 09_vote_ensemble.py "$@" &
wait

# Fimetune source location
python 10_timeProcess.py "$@"
