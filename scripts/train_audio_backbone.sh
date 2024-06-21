#!/bin/bash

# This script is used to train the audio backbone model using the VGGSound/Audioset dataset.
config=${1:-"config/vggsound/train.yaml"}
leaveout_fold=${2:-"fold1"}

python main.py train \
    --config $config \
    --leaveout_fold $leaveout_fold