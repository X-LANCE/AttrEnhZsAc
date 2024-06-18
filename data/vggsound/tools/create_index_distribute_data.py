import h5py
import argparse
import os
from tqdm import tqdm
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input_h5',
                    type=str,
                    default='vggsound_train.hdf5',
                    help='input h5 file')
parser.add_argument('--index_file',
                    type=str,
                    default='vggsound_train_index.hdf5',
                    help='index file')
parser.add_argument('-d', '--data_dir', 
                    type=str,
                    default='Train',
                    help='data directory')
parser.add_argument('-n',
                    type=int,
                    default=20,
                    help='number of data files')
args = parser.parse_args()

basedir = os.path.dirname(args.index_file)
data_dir = os.path.join(basedir, args.data_dir)
os.makedirs(data_dir, exist_ok=False)
with h5py.File(args.input_h5, 'r') as input_file,\
        h5py.File(args.index_file, 'w') as index_file:
    indices = list(range(input_file['label'].shape[0]))
    size = len(indices) // args.n
    pbar = tqdm(total=len(indices), desc='Distributing data')

    index_file.create_dataset('label', shape=((len(indices),)), dtype='S40')
    index_file.create_dataset('audio_name', shape=((len(indices),)), dtype='S40')
    index_file.create_dataset('file', shape=((len(indices),)), dtype='S40')
    index_file.create_dataset('index', shape=((len(indices),)), dtype=np.int32)
    index_file.attrs.create('sample_rate', data=input_file.attrs['sample_rate'], dtype=np.int32)
    index_file.attrs.create('n_class', data=input_file.attrs['n_class'], dtype=np.int16)
    index_file['label'][:] = input_file['label'][:]
    index_file['audio_name'][:] = input_file['audio_name'][:]
    for i in range(args.n):
        file_path = os.path.join(data_dir, f'VGGSound_{(i+1):0>2d}.hdf5')
        rel_path = os.path.join(args.data_dir, f'VGGSound_{(i+1):0>2d}.hdf5')
        with h5py.File(file_path, 'w') as data_file:
            if (i == args.n - 1):
                waveforms = input_file['waveform'][size * i:]
                index_file['file'][size * i:] = rel_path.encode()
                index_file['index'][size * i:] = np.array(list(range(len(waveforms))))
            else:
                waveforms = input_file['waveform'][size * i: size * (i + 1)]
                index_file['file'][size * i: size * (i + 1)] = rel_path.encode()
                index_file['index'][size * i: size * (i + 1)] = np.array(list(range(len(waveforms))))
            data_file.create_dataset('waveform', shape=(waveforms.shape), dtype=np.float32)
            data_file['waveform'][:] = waveforms
        pbar.update(len(waveforms))
    
    pbar.close()