# !/bin/bash
export TOKENIZERS_PARALLELISM=false
audio_model_path=$1
fold=$2


for seed in {0..4}
do
    # python main.py --seed $seed pretrain_bilinear_vector \
    #     --dataset vggsound \
    #     --audio_model_path $audio_model_path \
    #     --config config/vggsound/bilinear_vector.yaml \
    #     --outputdir experiment/ZeroShot/VGGSound/Bilinear/OneAttribute/$fold/Seed$seed \
    #     --leaveout_fold $fold \
    #     --attributes "['class']"
    # echo "Baseline method with seed $seed and fold $fold done";

    python main.py --seed $seed pretrain_bilinear_vector \
        --dataset vggsound \
        --audio_model_path $audio_model_path \
        --config config/vggsound/bilinear_vector.yaml \
        --outputdir experiment/ZeroShot/VGGSound/BilinearTrueBaseline/AllAttributes/$fold/Seed$seed \
        --leaveout_fold $fold \
        --criterion WeightedRankPairwiseLoss2 \
        --attributes "['class', 'emotion', 'onomatopoeia', 'pitch', 'simile', 'timbre']"
    echo "Baseline method with seed $seed and fold $fold done";

    # python main.py --seed $seed pretrain_bilinear_vector \
    #     --dataset audioset \
    #     --audio_model_path $audio_model_path \
    #     --config config/audioset/bilinear_vector.yaml \
    #     --outputdir experiment/NewDebug/Audioset/NewBilinearTrueBaseline/OneAttribute/$fold/Seed$seed \
    #     --leaveout_fold $fold \
    #     --criterion "WeightedRankPairwiseLoss2" \
    #     --attributes "['class', 'description']"
    # echo "Baseline (all attributes) method 2 with seed $seed and fold $fold done";

    # python main.py --seed $seed pretrain_bilinear_vector \
    #     --dataset audioset \
    #     --audio_model_path $audio_model_path \
    #     --config config/audioset/bilinear_vector.yaml \
    #     --outputdir experiment/Debug/Audioset/NewBilinearTrueBaseline/AllAttributes/$fold/Seed$seed \
    #     --leaveout_fold $fold \
    #     --text_kwargs "{'max_text_length': 100}" \
    #     --criterion "WeightedRankPairwiseLoss2" \
    #     --attributes "['class', 'description', 'emotion', 'onomatopoeia', 'pitch', 'simile', 'timbre']"
    # echo "Baseline (all attributes) method 2 with seed $seed and fold $fold done";

done

