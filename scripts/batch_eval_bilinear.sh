#!/bin/bash

# This script is used to evaluate the baseline performance of a batch of models on leaveout fold
# The models are trained with different seeds

base_dir=$1
# base_dir like **/fold1 and contains folds like **/fold1/Seed1, **/fold1/Seed2, ...

python main.py batch_zero_shot_bilinear_vector \
    --base_dir $base_dir \
    --test_h5 /hpc_stor03/sjtu_home/pingyue.zhang/work/zero-shot/data/vggsound/features/vggsound_test_classname.hdf5