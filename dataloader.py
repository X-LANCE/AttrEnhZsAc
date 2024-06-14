import h5py
import numpy as np
from pathlib import Path
import random
import librosa
import pandas as pd
import os

import torch 
from torch.utils.data import Dataset, DataLoader

class AudioBaseDataset(Dataset):
    def __init__(self, audio_file: str):
        with h5py.File(audio_file, 'r') as input:
            if 'waveform' in input:
                self.is_index = False
            else:
                self.is_index = True
                self.files = input['file'][:]
                self.indices_in_file = input['index'][:]
                self.base_dir = os.path.dirname(audio_file)
            self.labels = input['label'][:]
            self.audio_names = input['audio_name'][:]
        self.audio_file = audio_file
    def _get_waveform(self, idx):
        """
        Read waveform from index h5 file
        index h5 file contains:
            - file: list of file names
            - index: list of index in file
            - label: list of label name (encoded string)
            - audio_name: list of audio name (encoded string)
        data file (indicated in <file> and indexed by <index> above) contains:
            - waveform: numpy array of N x T
        """

        if not self.is_index:
            with h5py.File(self.audio_file) as input:
                waveform = input['waveform'][idx]
        else:
            file = os.path.join(self.base_dir, self.files[idx].decode())
            index_in_file = self.indices_in_file[idx]
            
            with h5py.File(file, 'r') as input:
                waveform = input['waveform'][index_in_file]
        label = self.labels[idx]
        audio_name = self.audio_names[idx]
        
        waveform = waveform.astype(np.float32)

        return {
            "waveform": waveform,
            "label": label.decode(),
            "audio_name": audio_name.decode()
        }


class AudioDataset(AudioBaseDataset):
    """
    Dataset of classifying audios

    ========================================
    Parameters:
    audio_file: h5 file containing waveform
        - waveform: numpy array of N x T
        - label: numpy array of label name (encoded string)
    """
    
    def __init__(self,
                 audio_file: str,
                 label2int: dict,
                 audio_transform,
                 indices: list):
        super(AudioDataset, self).__init__(audio_file)
        self.label2int = label2int
        self.indices = indices
        self.audio_transform = audio_transform

    def __getitem__(self, i):
        index = self.indices[i]
        data = self._get_waveform(index)
        waveform, label = data['waveform'], data['label']
        audio_name = data['audio_name']
        target = self.label2int[label]
        waveform = self.audio_transform(waveform)
        return waveform, target, label, audio_name

    def __len__(self):
        return len(self.indices)


class RandomAttributeVectorDataset(Dataset):
    """
    Dataset of audio embeddings and RAW text
    Generate random attributes for each audio

    ========================================
    Initialization Parameters:
    attr_lists (List[List[str]]): list of attributes
    targets (List[int]): list of target
    audio_embeddings (torch.tensor): audio embeddings
    random_strategy (str): 
        - all: all attributes
        - random_with_class: random attributes including class name
        - random: random attributes

    ========================================
    Returns:
    audio_embedding (torch.tensor): D
    desc (str): description text (randomly picked)
    target (int): target index
    """
    
    def __init__(self,
                 audio_embeddings,
                 attr_list: list,
                 targets: list,
                 random_strategy: str = 'all'):
        self.audio_embeddings = audio_embeddings
        self.targets = targets
        self.attr_list = attr_list
        assert random_strategy in ['all', 'random_with_class', 'random']
        self.random_strategy = random_strategy

    def __getitem__(self, idx):
        target = self.targets[idx]
        attr_list = self.attr_list[target]
        audio_embedding = self.audio_embeddings[idx]
        if self.random_strategy == 'all':
            desc = '; '.join(attr_list)
        elif self.random_strategy == 'random_with_class':
            selected_attrs = [attr_list[0]]
            num_attr = len(attr_list) - 1 # first element already selected
            if num_attr:
                num_elements = torch.randint(0, num_attr + 1, (1,)).item()
                indices = torch.randperm(num_attr)[0: num_elements] + 1
                selected_attrs.extend([attr_list[i] for i in indices])
            desc = '; '.join(selected_attrs)
        else:
            selected_attrs = []
            num_attr = len(attr_list) 
            if num_attr:
                num_elements = torch.randint(1, num_attr + 1, (1,)).item()
                indices = torch.randperm(num_attr)[0: num_elements]
                selected_attrs.extend([attr_list[i] for i in indices])
            desc = '; '.join(selected_attrs)
        # print(desc)
        return audio_embedding, desc, target
    
    def __len__(self):
        return len(self.targets)


class VectorDataset(Dataset):
    """
    Bilinear Vector Dataset of audio embeddings

    ========================================
    Parameters:
    audio_embeddings (torch.tensor): B x D
    targets (List[int]): B

    ========================================
    Returns:
    audio_embed (torch.tensor): D
    target (int)
    """
    
    def __init__(self,
                 audio_embeddings,
                 targets):
        self.audio_embeddings = audio_embeddings
        self.targets = targets

    def __getitem__(self, idx):
        return self.audio_embeddings[idx], self.targets[idx]

    def __len__(self):
        return len(self.targets)

def get_collate_fn(task, **kwargs):
    if task == 'vector':
        def vector_collate_fn(batch):
            """
            collate_fn for bilinear vector dataset

            ================
            Returns:
            Batch dict:
                - audio_embed (torch.tensor): B x D
                - target (torch.tensor): B
            """
            embeddings, targets = zip(*batch)
            embeddings = torch.stack(embeddings, dim=0) # B x D
            targets = torch.tensor(targets)
            return {
                'audio_embed': embeddings,
                'target': targets
            }
        return vector_collate_fn
    

    if task == 'audio':
        def audio_collate_fn(batch):
            """
            collate_fn for single audio task

            ================
            Returns:
            Batch dict:
                - waveform (torch.tensor): B x T
                - target (torch.tensor): B
                - label (List[str]): B
                - audio_name (List[str]): B
            """
            waveform_list, targets, labels, audio_names = zip(*batch)
            waveforms = torch.stack(waveform_list, dim=0) # B x T
            targets = torch.tensor(targets)
            
            return {
                'waveform': waveforms,
                'target': targets,
                'label': labels,
                'audio_name': audio_names
            }
        return audio_collate_fn
    

    if task == 'audio-text':
        tokenize_fn = kwargs['tokenize_fn']
        def audio_text_collate_fn(batch):
            """
            collate_fn for audio-text task (random attribute)
            Use tokenizer to generate token

            ================
            Returns:
            Batch dict:
                - waveform (torch.tensor): B x T
                - target (torch.tensor): B
                - input_ids (torch.tensor): token IDs
                - attention_mask (torch.tensor): real tokens (1) and padding tokens (0)
            """
            audio_data_list, desc_list, targets = zip(*batch)
            audio_data = torch.stack(audio_data_list, dim=0) # B x T
            targets = torch.tensor(targets)
            desc_tokens = tokenize_fn(desc_list)
            return {
                'audio_data': audio_data,
                'target': targets,
                **desc_tokens}
        return audio_text_collate_fn
    
def create_random_attr_dataloader(audio_embeddings,
                                  attr_list,
                                  targets,
                                  tokenize_fn,
                                  random_strategy: str = 'all',
                                  is_train: bool = True,
                                  **kwargs):
    kwargs.setdefault('batch_size', 64)
    kwargs.setdefault('num_workers', 8)
    print(kwargs)
    if not is_train:
        random_strategy = 'all'
    dataset = RandomAttributeVectorDataset(audio_embeddings=audio_embeddings,
                                           attr_list=attr_list,
                                           random_strategy=random_strategy,
                                           targets=targets)
    collate_fn = get_collate_fn(task='audio-text',
                                tokenize_fn=tokenize_fn)
    if is_train:
        dataloader = DataLoader(dataset=dataset,
                                collate_fn=collate_fn,
                                drop_last=True,
                                shuffle=True,
                                **kwargs)
    else:
        dataloader = DataLoader(dataset=dataset,
                                collate_fn=collate_fn,
                                drop_last=False,
                                shuffle=False,
                                **kwargs)
    return dataloader


def create_bilinear_vector_dataloader(audio_embeddings,
                                   targets,
                                   is_train: bool = True,
                                   **kwargs):
    kwargs.setdefault('batch_size', len(targets))
    kwargs.setdefault('num_workers', 8)
    collate_fn = get_collate_fn(task='vector')
    dataset = VectorDataset(audio_embeddings=audio_embeddings,
                            targets=targets)
    if is_train:
        dataloader = DataLoader(dataset=dataset,
                                collate_fn=collate_fn,
                                drop_last=True,
                                shuffle=True,
                                **kwargs)
    else:
        dataloader = DataLoader(dataset=dataset,
                                collate_fn=collate_fn,
                                drop_last=False,
                                shuffle=False,
                                **kwargs)
    return dataloader

def create_train_cls_dataloader(audio_file: str,
                               label2int: dict,
                               indices: list,
                               audio_transform,
                               **kwargs):
    kwargs.setdefault('batch_size', 64)
    kwargs.setdefault('num_workers', 8)
    collate_fn = get_collate_fn(task='audio')
    dataset = AudioDataset(audio_file=audio_file,
                           label2int=label2int,
                           audio_transform=audio_transform,
                           indices=indices)
    dataloader = DataLoader(dataset=dataset,
                            collate_fn=collate_fn,
                            drop_last=True,
                            shuffle=True,
                            **kwargs)
    return dataloader

def create_val_cls_dataloader(audio_file: str,
                              label2int: dict,
                              indices: list,
                              audio_transform,
                              **kwargs):
    kwargs.setdefault('batch_size', 64)
    kwargs.setdefault('num_workers', 8)
    dataset = AudioDataset(audio_file=audio_file,
                           label2int=label2int,
                           audio_transform=audio_transform,
                           indices=indices)
    collate_fn = get_collate_fn(task='audio')
    dataloader = DataLoader(dataset=dataset,
                            collate_fn=collate_fn,
                            drop_last=False,
                            shuffle=False,
                            **kwargs)
    return dataloader
