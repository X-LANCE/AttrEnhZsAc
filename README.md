# Source codes

This repository contains the implementation of:

* train: train a base model with one fold excluded as the embedding model.
* pretrain_bilinear_vector: bilinear method as the baseline
* pretrain_random_attr_vector: our contrastive method
* batch_zero_shot_bilinear_vector: evaluate a batch of bilinear model using different seeds but the same leaveout fold
* batch_zero_shot_random_attr_vector: evaluate a batch of contrastive model using different seeds but the same leaveout fold

### Data Preparation

index file (vggsound_train_index_classname.hdf5): 
- label: contains label name (b'waterfall burbling') for each sample with shape (B,)
- file: contains file name for each sample with shape (B,)
  - file name indicates the data file which contains the waveform of this sample
- index: contains index for each sample with shape (B,)
  - index is the position of the sample in the data file

data file:
- waveform: contains all waveforms swith shape (B1, 320000) (sr = 32000, 10 seconds)
- NOTE that waveforms are separated into different data files

fold file: contains the fold information about class belongs to which fold
- check `folds/class_wise.json` for vggsound datasets
```json
{
  "all_classes": [
    "waterfall burbling",
    "car engine starting",
    ...
  ],
  "folds": [
    "fold1": [
      "playing drum kit",
      "playing bass guitar",
      ...
    ],
    "fold2": [
      ...
    ],
    ...
  ],
}
```

description file: description for each label stored in *tsv* file
- check `description/vgg_desc.tsv`

### Training Instruction

* Train on VGGSound dataset with fold1 excluded
```bash
./scripts/train_audio_backbone.sh  \
  config/vggsound/train.yaml \
  fold1
```

* Train Baseline model with the trained base model using different seeds
```bash
./scripts/run_bilinear.sh \
  <pretrained_audio_path> \
  fold1
```

* Zero-shot evaluation, `<bilinear_base_dir>` like `**/fold1` and contains folds like `**/fold1/Seed1`, `**/fold1/Seed2`, etc
```bash
./scripts/batch_eval_bilinear.sh \
  <bilinear_base_dir>
```