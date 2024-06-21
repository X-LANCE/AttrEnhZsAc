#!/bin/bash

# This script is used to run the random attributes method using multiple seeds and a specified leaveout fold.
# It is designed to be used in the context of zero-shot learning experiments.

# Usage: ./run_random_attr.sh <audio_model_path> <fold>
#   <audio_model_path> - The path to the folder containing audio model to be used in the experiment.
#       This should be a audio backbone model pretrained using the same leaveout fold as well.
#       The pretrained model has a name like "eval_best_checkpoint_*.pt"
#   <fold> - The leaveout fold to be used in the experiment.

# Example usage: ./run_random_attr.sh vggsound/train/fold1/ fold1

# Note: This script assumes that the necessary data and dependencies are already set up.
# Make sure to configure the script accordingly before running it.

export TOKENIZERS_PARALLELISM=false
audio_model_path=$1
fold=${2:-"fold1"}

for seed in {0..4}
do
    python main.py --seed $seed pretrain_random_attr_vector \
        --dataset vggsound \
        --audio_model_path $audio_model_path \
        --config config/vggsound/pretrain_random_attr_vector.yaml \
        --outputdir experiment/ZeroShot/VGGSound/ContrastiveOneLinear/CNN14/AllAttributes/$fold/Seed$seed \
        --leaveout_fold $fold \
        --dataloader_args "{'random_strategy': 'all'}" \
        --attributes "['class', 'emotion', 'onomatopoeia', 'pitch', 'simile', 'timbre']"
    echo "Contrastive method with all attributes done";
done

