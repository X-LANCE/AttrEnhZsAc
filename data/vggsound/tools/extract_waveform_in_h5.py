import librosa
import h5py
import pypeln.process as pr
import numpy as np
import pandas as pd
import argparse
import os
import json
from glob import glob
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('-c',
                    type=int,
                    default=20,
                    help='number of workers')
parser.add_argument('-a',
                    '--audio_path',
                    type=str,
                    default='audios/',
                    help='path of raw data')
parser.add_argument('--split',
                    type=str,
                    default='train',
                    choices=['train', 'test'],
                    help='train or test split')
parser.add_argument('--sr',
                    type=int, 
                    default=32000,
                    help='sample rate')
parser.add_argument('--length',
                    type=int,
                    default=10,
                    help='fixed length (seconds) of each wav')
parser.add_argument('-l', '--label_file',
                    type=str,
                    default='vggsound.csv',
                    help='the label of raw data')
parser.add_argument('-o', '--output',
                    type=str,
                    default='vggsound_train.hdf5',
                    help='path of output raw wav')
args = parser.parse_args()

print(f"""
Extracting waveform in .wav from {args.audio_path} \
to {args.output}. \
Using sample rate {args.sr}, \
data from {args.split} split \
and fixed length {args.length}. \
""")

sr = args.sr 
n_frames = sr * args.length
audios = glob(os.path.join(args.audio_path, '*.wav'))

df = pd.read_csv(args.label_file, sep=',', header=None)
df.columns = ['ID', 'start_time', 'label', 'split']

names, start_times, labels, splits = zip(*(df.values))
names = [f'{names[i]}_{start_times[i]}' for i in range(len(names))]
label_dict = dict(zip(names, labels))
split_dict = dict(zip(names, splits))
audios = list(filter(lambda x: split_dict[os.path.basename(x).split('.')[0]] == args.split, audios))

print(f"{len(audios)} audios in {args.split} split.") 

def pad_or_truncate(waveform):
    """
    Pad or truncate audio to n_frames
    ================================
    Parameters:
    waveform (np.array): The audio waveform

    """

    if len(waveform) > n_frames:
        return waveform[:n_frames]
    tmp = np.zeros(n_frames, dtype=waveform.dtype)
    tmp[:len(waveform)] = waveform
    return tmp

def read_wav(audio_file):
    waveform, _ = librosa.load(audio_file, sr=sr, dtype=np.float32)
    audio_name = os.path.basename(audio_file).split('.')[0]
    return audio_name, pad_or_truncate(waveform)


with h5py.File(args.output, 'w') as output, tqdm(total=len(audios), desc='Reading wav') as pbar:
    output.create_dataset('audio_name',
                          shape=((len(audios),)),
                          dtype='S40')
    output.create_dataset('waveform', 
                          shape=((len(audios), n_frames)),
                          dtype=np.float32)
    output.create_dataset('label',
                          shape=((len(audios),)),
                          dtype='S40')
    output.attrs.create('sample_rate',
                        data=sr,
                        dtype=np.int16)
    output.attrs.create('n_class',
                        data=309,
                        dtype=np.int16)

    for audio_name, waveform in pr.map(read_wav, audios, workers=args.c, maxsize=2*args.c):
        n = pbar.n
        label = label_dict[audio_name]
        output['audio_name'][n] = audio_name.encode()
        output['label'][n] = label.encode()
        output['waveform'][n] = waveform
        pbar.update()


