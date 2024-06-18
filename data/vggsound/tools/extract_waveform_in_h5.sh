#!/bin/bash

audio_dir=$1
split=${2:-"train"}
sr=${3:-32000}
length=${4:-10}
label=${5:-"vggsound.csv"}
output=${6:-"vggsound_${split}.hdf5"}

SCRIPT_DIR=$(dirname $(realpath $0))
OUTPUT_DIR=$(dirname $output)

python $SCRIPT_DIR/extract_waveform_in_h5.py \
    -c 1 \
    --audio_path $audio_dir \
    --split $split \
    --sr $sr \
    --length $length \
    --label $label \
    --output $output 2>&1 | tee $OUTPUT_DIR/extract_waveform_in_h5.log
