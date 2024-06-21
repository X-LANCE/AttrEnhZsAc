# !/bin/bash
export TOKENIZERS_PARALLELISM=false
audio_model_path=$1
fold=$2

for seed in {0..4}
do
    # python main.py --seed $seed pretrain_random_attr_vector \
    #     --dataset vggsound \
    #     --audio_model_path $audio_model_path \
    #     --config config/vggsound/pretrain_random_attr_vector.yaml \
    #     --outputdir experiment/ZeroShot/VGGSound/ContrastiveOneLinear/CNN14/RandomWithClassMPNet/$fold/Seed$seed \
    #     --leaveout_fold $fold \
    #     --dataloader_args "{'random_strategy': 'random_with_class'}" \
    #     --model_kwargs "{'text_model_kwargs': {'text_embedding_model': 'sentence-transformers/all-mpnet-base-v2'}}" \
    #     --tokenizer "{'tokenizer_type': 'sentence-transformers/all-mpnet-base-v2'}" \
    #     --attributes "['class', 'emotion', 'onomatopoeia', 'pitch', 'simile', 'timbre']"
    # echo "Contrastive method with random_with_class done";
    

    # python main.py --seed $seed pretrain_random_attr_vector \
    #     --dataset vggsound \
    #     --audio_model_path $audio_model_path \
    #     --config config/vggsound/pretrain_random_attr_vector.yaml \
    #     --outputdir experiment/ZeroShot/VGGSound/ContrastiveOneLinear/CNN14/Ablation/WithEmotion/$fold/Seed$seed \
    #     --leaveout_fold $fold \
    #     --dataloader_args "{'random_strategy': 'random_with_class'}" \
    #     --attributes "['class', 'emotion']"
    # echo "Contrastive method with emotion done";

    # python main.py --seed $seed pretrain_random_attr_vector \
    #     --dataset vggsound \
    #     --audio_model_path $audio_model_path \
    #     --config config/vggsound/pretrain_random_attr_vector.yaml \
    #     --outputdir experiment/ZeroShot/VGGSound/ContrastiveOneLinear/CNN14/Ablation/WithOno/$fold/Seed$seed \
    #     --leaveout_fold $fold \
    #     --dataloader_args "{'random_strategy': 'random_with_class'}" \
    #     --attributes "['class', 'onomatopoeia']"
    # echo "Contrastive method with ono done";

    python main.py --seed $seed pretrain_random_attr_vector \
        --dataset vggsound \
        --audio_model_path $audio_model_path \
        --config config/vggsound/pretrain_random_attr_vector.yaml \
        --outputdir experiment/ZeroShot/VGGSound/ContrastiveOneLinear/CNN14/Ablation/WithPitch/$fold/Seed$seed \
        --leaveout_fold $fold \
        --dataloader_args "{'random_strategy': 'random_with_class'}" \
        --attributes "['class', 'pitch']"
    echo "Contrastive method with pitch done";

    # python main.py --seed $seed pretrain_random_attr_vector \
    #     --dataset vggsound \
    #     --audio_model_path $audio_model_path \
    #     --config config/vggsound/pretrain_random_attr_vector.yaml \
    #     --outputdir experiment/ZeroShot/VGGSound/ContrastiveOneLinear/CNN14/Ablation/WithSimile/$fold/Seed$seed \
    #     --leaveout_fold $fold \
    #     --dataloader_args "{'random_strategy': 'random_with_class'}" \
    #     --attributes "['class', 'simile']"
    # echo "Contrastive method with simile done";

    # python main.py --seed $seed pretrain_random_attr_vector \
    #     --dataset vggsound \
    #     --audio_model_path $audio_model_path \
    #     --config config/vggsound/pretrain_random_attr_vector.yaml \
    #     --outputdir experiment/ZeroShot/VGGSound/ContrastiveOneLinear/CNN14/Ablation/WithTimbre/$fold/Seed$seed \
    #     --leaveout_fold $fold \
    #     --dataloader_args "{'random_strategy': 'random_with_class'}" \
    #     --attributes "['class', 'timbre']"
    # echo "Contrastive method with timbre done";


    # python main.py --seed $seed pretrain_random_attr_vector \
    #     --dataset audioset \
    #     --audio_model_path $audio_model_path \
    #     --config config/audioset/pretrain_random_attr_vector.yaml \
    #     --outputdir experiment/Debug/Audioset/ContrastiveOneLinear/CNN14/OneAttribute/$fold/Seed$seed \
    #     --dataloader_args "{'random_strategy': 'all', 'max_text_length': 80}" \
    #     --leaveout_fold $fold \
    #     --attributes "['class', 'description']"
    #     # --attributes "['class', 'emotion', 'onomatopoeia', 'pitch', 'simile', 'timbre']"
    # echo "Contrastive method with one attribute done";

    # python main.py --seed $seed pretrain_random_attr_vector \
    #     --dataset audioset \
    #     --audio_model_path $audio_model_path \
    #     --config config/audioset/pretrain_random_attr_vector.yaml \
    #     --outputdir experiment/NewDebug/Audioset/ContrastiveOneLinear/CNN14/AllAttributes/$fold/Seed$seed \
    #     --tokenizer "{'tokenizer_kwargs': {'max_length': 100}}" \
    #     --leaveout_fold $fold \
    #     --attributes "['class', 'description', 'emotion', 'onomatopoeia', 'pitch', 'simile', 'timbre']"
    # echo "Contrastive method with all attributes done";
done

