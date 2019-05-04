#!/bin/bash

# bash train.sh /data/training/ /data/trainingAnswers.csv

# Data preparation
python 01_make_slice_data.py "$@" &
wait
python 02_make_features.py "$@" &
wait

# training
python 03_train_ANN_CNN.py "$@"

