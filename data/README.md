# Data Preparation Pipeline

In this work, we use two datasets: VGGSound and AudioSet. Here we provide the pipeline to pre-process VGGSound data. The process for AudioSet is similar. 

On VGGSound, pre-process includes:
* Convert .mp4 file to .wav file
* Extract waveform (numpy.array) into hdf5 file
* Create index on hdf5 file and distribute data

## Pipeline scripts

* Assume .mp4 files are store at `vggsound/videos` and .wav will be stored in `vggsound/audios`
```bash
cd vggsound
./tools/convert_mp4_to_wav.sh  \
    ./videos/ \
    ./audios/ \
    32000
```

* Extract waveform and store in hdf5 file, with [vggsound.csv](https://github.com/hche11/VGGSound/blob/master/data/vggsound.csv) downloaded here:
```bash
./tools/extract_waveform_in_h5.sh \
    ./audios/ \
    train \
    32000 \ # sample rate
    10 \ # 10 seconds
    vggsound.csv \
    features/vggsound_train.hdf5
```

* Build index and distribute data, data will be in `features/Train`
```bash
python tools/create_index_distribute_data.py \
    --input_h5 features/vggsound_train.hdf5 \
    --index_file features/vggsound_train_index.hdf5 \
    --data_dir Train \
    -n 20
```