#!/bin/bash

video_dir=$1
wav_dir=$2
sr=${3:-32000}

SCRIPT_DIR=$(dirname $(realpath $0))

python $SCRIPT_DIR/convert_mp4_to_wav.py \
    -c 8 \
    --video_path $video_dir \
    --output_path $wav_dir \
    --sr $sr 2>&1 | tee $wav_dir/convert_mp4_to_wav.log
