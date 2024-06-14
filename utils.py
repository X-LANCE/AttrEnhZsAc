import h5py
import json
import numpy as np
import pandas as pd
import torch
import pandas as pd
import h5py 
import yaml
import sys, os
from loguru import logger
from models import models
from glob import glob
from tqdm import tqdm


def get_classes(fold_file: str,
                select_leaveout: bool = False,
                leaveout_fold: str = 'fold1'):
    """
    Get classes from fold file, classes from folds except leaveout fold or leaveout fold

    =================================
    Parameters:
    fold_file (str): fold file
    select_leaveout (bool): whether to select leaveout classes
    leaveout_fold (str): leaveout fold
    """
    with open(fold_file, 'r') as input:
        input = json.load(input) # category/fold -> class name
    leaveout_classes = input['folds'][leaveout_fold]
    if select_leaveout:
        return leaveout_classes
    all_classes = input['all_classes']
    selected_classes = list(filter(lambda x: x not in leaveout_classes, all_classes))
    return selected_classes

def get_indices(h5: str,
                classes: list):
    """
    Get indices of samples from picked classes from h5 file
    =================================
    Parameters:
    h5 (str): input h5 file
        - label: numpy array of labels (D,)
    classes: picked classes (dog, cat, ...)

    =================================
    Returns:
    indices (list): indices of samples which belong to picked classes
    classes (list): picked classes
    """

    classes = list(map(lambda x: x.encode(), classes))
    # labels are bytes in h5 file
    with h5py.File(h5, 'r') as input:
        all_labels = input['label'][:]
        indices = np.where(np.isin(all_labels, classes))[0]
    return indices, classes

@torch.no_grad()
def get_audio_embedding(h5: str,
                        indices: list,
                        audio_transform,
                        label2int: dict,
                        config: dict = {},
                        pretrained_path: str = None,
                        embedding_file: str = None,
                        **dataloader_kwargs):
    """
    Extract audio embeddings using pretrained model
    Load embeddings from h5 file if embedding_file is already existed
    Save embeddings to h5 file if embedding_file is provided

    =================================
    Parameters:
    h5 (str): input h5 file
    audio_transform: transform function for audio
    label2int (dict): class -> integer
    config (dict): configurtion for audio model
    pretrained_path (str): path to pretrained model
    embedding_file (str): path to save embeddings
    dataloader_kwargs (dict): arguments for dataloader
        - batch_size
        - num_workers

    =================================
    Returns:
    embeddings (torch.Tensor): audio embeddings
    targets (list): list of targets
    """


    if embedding_file and os.path.exists(embedding_file):
        with h5py.File(embedding_file, 'r') as input:
            embeddings = input['embedding'][:]
            labels = input['label'][:]
        targets = [label2int[label.decode()] for label in labels]
        embeddings = torch.from_numpy(embeddings)
        assert len(embeddings) == len(indices)
        return embeddings, targets
    
    model, _, _ = get_model_from_pretrain(model_path=pretrained_path,
                                          config=config,
                                          resume=(pretrained_path is not None))
    model.eval().cuda(0)

    from dataloader import create_val_cls_dataloader
    dataloader_kwargs.setdefault('batch_size', 16)
    dataloader_kwargs.setdefault('num_workers', 4)
    ValDataloader = create_val_cls_dataloader(audio_file=h5,
                                              label2int=label2int,
                                              indices=indices,
                                              audio_transform=audio_transform,
                                              **dataloader_kwargs)
    
    audio_embeddings, targets = [], []
    labels, audio_names = [], []
    for data in tqdm(ValDataloader, desc='Extracting audio embeddings', ncols=85):
        waveform, target, label, audio_name = data['waveform'].cuda(0),\
                                              data['target'].cuda(0),\
                                              data['label'],\
                                              data['audio_name']
        embeddings = model.encode_audio(waveform).cpu()
        audio_embeddings.append(embeddings)
        targets.extend(target.cpu().tolist())
        labels.extend(label)
        audio_names.extend(audio_name)
    audio_embeddings = torch.cat(audio_embeddings, dim=0)
    # import torch.nn.functional as F
    # audio_embeddings = F.normalize(audio_embeddings, dim=-1)
    if embedding_file:
        with h5py.File(embedding_file, 'w') as output:
            output['embedding'] = audio_embeddings.numpy()
            output['audio_name'] = [name.encode() for name in audio_names]
            output['label'] = [label.encode() for label in labels]
    return audio_embeddings, targets

def get_tokenize_fn(tokenizer_type: str, tokenizer_kwargs: dict):
    """
    Get tokenizer for BERT-like model

    =================================
    Parameters:
    tokenizer_type: e.g. 'sentence-transformers/all-MiniLM-L6-v2'
    tokenizer_kwargs: e.g. {'padding': "max_length", 'max_length': 64, 'truncation': True}

    =================================
    Returns:
    tokenize (function): tokenize function
    """
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_type)
    def tokenize(text):
        tokens = tokenizer(text, **tokenizer_kwargs)
        return tokens
    return tokenize


@torch.no_grad()
def get_text_embedding(texts: list,
                       tokenize_fn: callable,
                       text_embedding_model: str = 'sentence-transformers/all-MiniLM-L6-v2',
                       text_embedding_type: str = 'BERT'):
    """
    Extract text embedding from descriptions

    =================================
    Parameters:
    texts: list of text
        - ',' separated attribute
        - e.g. "cow, a low pitch, moderate intensity"
    tokenize_fn: tokenize function
    text_embedding_type: type of embedding, e.g. 'BERT'
    text_embedding_model: e.g. 'sentence-transformers/all-MiniLM-L6-v2'

    =================================
    Returns:
    text_embeddings (torch.Tensor): T x D text embeddings
    """

    if text_embedding_type == 'BERT':
        from models import text_models

        text_model = getattr(text_models, text_embedding_type)(text_embedding_model=text_embedding_model)
        text_model.eval().cuda(0)

        def get_embedding(text, embed_type='mean_emb'):
            tokens = tokenize_fn(text)
            tokens = {k: v.cuda(0) for k, v in tokens.items()}
            model_out = text_model(**tokens)
            return model_out[embed_type].cpu() # T' x D

        text_embeddings = get_embedding(texts)
        return text_embeddings
    
    else:
        raise NotImplementedError

def split_dataset(h5: str,
                  label2int: dict,
                  indices: list,
                  train_ratio: float = 0.8,
                  seed: int = 0):
    """
    Split dataset into train and val set while keeping the ratio of each class
    =================================
    Parameters:
    h5 (str): input h5 file
        - label: numpy array of labels (D,)
    label2int (dict): class -> integer
    indices (list): indices of samples
    train_ratio (float): ratio of train set
    seed (int): random seed

    =================================
    Returns:
    train (list): indices of train set
    val (list): indices of val set
    """

    with h5py.File(h5, 'r') as input:
        all_labels = input['label'][:][indices]
    targets = list(map(lambda x: label2int[x.decode()], all_labels))
    target2indices = {}
    for index, target in zip(indices, targets):
        target2indices.setdefault(target, []).append(index)

    random_state = np.random.RandomState(seed=0)
    # NOTE the seed is set to 0 to fix the train and val set
    
    train, val = [], []
    for target, indices in target2indices.items():
        random_state.shuffle(indices)
        train_size = int(train_ratio * len(indices))
        train.extend(indices[:train_size])
        val.extend(indices[train_size:])
    return train, val
    
        

def get_label2int(desc_file: str = None, classes: list = None):
    """
    mapping (sorted) classes to integer [0, 1, ...]

    =================================
    Parameters:
    desc_file (str): tsv file containing description
        column: class, pitch, timbre, ...
    attr_list (list): picked attributes

    =================================
    Returns:
    label2int (dict): class -> integer
    """
    assert desc_file or classes
    if classes:
        classes = sorted(classes)
    else:
        df = pd.read_csv(desc_file, sep='\t')
        classes = sorted(df.loc[:, 'class'].values)
    label2int = {_class: idx for idx, _class in enumerate(classes)}
    return label2int


def get_label2desc(desc_file: str,
                   attr_list: list = []):
    """
    mapping classes to description

    =================================
    Parameters:
    desc_file (str): tsv file containing description
        column: class, pitch, timbre, ...
    attr_list (list): picked attributes

    =================================
    Returns:
    label2desc (dict): class -> description
    """

    df = pd.read_csv(desc_file, sep='\t')
    df['CLASS'] = df['class'].copy()
    df['class'] = df['class'].apply(lambda x: x.replace(',', ''))
    if 'pitch' in attr_list:
        df['pitch'] = df['pitch'].apply(lambda x: x.replace('pitch', 'frequency'))

    df = df.set_index('CLASS').loc[:, attr_list]
    df['combined'] = df.apply(lambda row: '; '.join(row.values.astype(str)), axis=1)
    label2desc = df.combined.to_dict()
    return label2desc


def parse_config(config_file, debug=False, **kwargs):
    """
    Convert yaml file to dictionary

    =================================
    Parameters:
    config_file (str): yaml file
    debug (bool): debug mode
    kwargs: additional arguments given in command line which will overwrite the config file
    """
    with open(config_file) as con_read:
        config = yaml.load(con_read, Loader=yaml.FullLoader)
    # for k, v in kwargs.items():
    #     config[k] = v
    def merge_dict(target_dict, input_dict):
        if not isinstance(target_dict, dict) or not isinstance(input_dict, dict):
            raise NotImplementedError
        for k, input_v in input_dict.items():
            if k not in target_dict:
                target_dict[k] = input_v
                continue
            target_v = target_dict[k]
            if not isinstance(input_v, dict) and not isinstance(target_v, dict):
                target_dict[k] = input_v
                continue
            if isinstance(input_v, dict) and isinstance(target_v, dict):
                merge_dict(target_v, input_v)
            else:
                raise NotImplementedError
    merge_dict(config, kwargs)
    if debug:
        config['dataloader_args']['batch_size'] = 32
        config['dataloader_args']['num_workers'] = 4
        config['outputdir'] = 'experiment/Debug'
        config['n_epochs'] = 5
        config['iters_per_epoch'] = 10
    return config

def genlogger(file):
    log_format = "[<green>{time:YYYY-MM-DD HH:mm:ss}</green>] {message}"
    logger.configure(handlers=[{"sink": sys.stderr, "format": log_format}])
    if file:
        logger.add(file, enqueue=True, format=log_format)
    return logger

class Logger():
    def __init__(self, file, rank=0):
        self.logger = None
        self.rank = rank
        if not rank:
            self.logger = genlogger(file)
    def info(self, msg):
        if not self.rank:
            self.logger.info(msg)
    
def get_model_from_pretrain(
    model_path: str,
    config: dict = {},
    resume: bool = False,
    **kwargs
):
    if not resume:
        # must provide config file to construct model
        assert len(config)
    else:
        # use the config file of pretrained model
        pretrain_config = torch.load(
            glob(os.path.join(model_path, '*config*'))[0], map_location='cpu')
        config = pretrain_config

    model = getattr(models, config['model'])(**config['model_kwargs'], **kwargs)
    if model_path is None:
        return model, {}, {}
    
    saved = torch.load(
        glob(os.path.join(model_path, '*best*.pt'))[0], map_location='cpu')
    params, optim_params, scheduler_params =\
        saved['model'], saved['optimizer'], saved['scheduler']

    if resume:
        print("Load all parameters")
        model.load_state_dict(params, strict=True)
        return model, optim_params, scheduler_params
    return model, optim_params, scheduler_params

def get_output_func(pattern='supcon', **kwargs):
    """
    output function between model output and loss calculation

    =================================
    model_out:
        - audio_embed
        - text_embed
        - audio_proj
        - text_proj
        - audio_target
        - text_target
    """
    def output_fn_supcon(model_out):
        """
        Calculate in supcon manner

        =================================
        Parameters:
        model_out:
            - audio_embed
            - text_embed
            - audio_proj
            - text_proj
            - audio_target
            - text_target

        =================================
        Returns:
        scores: similarity scores between audio and text
        masks: mask where 1 for positive and 0 for negative
        """

        audio_proj = model_out['audio_proj']
        text_proj = model_out['text_proj']
        audio_targets = model_out['audio_target']
        text_targets = model_out['text_target']

        text_proj_detach = text_proj.clone().detach()
        scores = audio_proj @ text_proj_detach.T
        masks = audio_targets.unsqueeze(-1) == text_targets.unsqueeze(0)

        return scores, masks
    
    if pattern == 'supcon':
        return output_fn_supcon
    else:
        raise lambda x: x

def get_transform():
    """
    Get audio transform function

    =================================
    x: waveform
    """
    return lambda x: torch.from_numpy(x)
