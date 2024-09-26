import torch
from xtuner.parallel.sequence import (
    pad_for_sequence_parallel,
    get_sequence_parallel_group,
    split_for_sequence_parallel,
    reduce_sequence_parallel_loss,
)
from ..config import Config
from ..config.config_utils import get_sp_size
from loguru import logger

def count_consecutive_zeros(tensor, left=True):
    assert tensor.dim() == 2
    column_zeros = (tensor == 0).all(dim=0)
    if not left:
        column_zeros = torch.flip(column_zeros, dims=[-1])
    num_consecutive_zeros = 0
    for i in range(column_zeros.shape[0]):
        if column_zeros[i]:
            num_consecutive_zeros = num_consecutive_zeros + 1
        else:
            break
    return num_consecutive_zeros

def remove_paddings(batch, labels):
    left_zeros = count_consecutive_zeros(batch['input_ids'],left=True)
    if left_zeros != 0:
        batch['input_ids'] = batch['input_ids'][:,left_zeros:]
        batch['attention_mask'] = batch['attention_mask'][:,left_zeros:]
        batch['position_ids'] = batch['position_ids'][:,left_zeros:]
        if 'input_ids' in labels:
            labels['input_ids'] = labels['input_ids'][:,left_zeros:]
    
    right_zeros = count_consecutive_zeros(batch['input_ids'],left=False)
    if right_zeros != 0:
        for key in batch.keys():
            batch[key]= batch[key][:,:-right_zeros]
        for key in labels.keys():
            labels[key] = labels[key][:,:-right_zeros]

    return batch, labels

def labels_remove_paddings(labels):
    pad_keys = ['old_logprobs','advantages','mask','old_values','returns']
    no_pad_keys = ['input_ids']
    new_labels={}
    for key in pad_keys:
        if key in labels:
            new_labels[key] = labels[key][:,1:]
    for key in no_pad_keys:
        if key in labels:
            new_labels[key] = labels[key]
    return new_labels


def labels_add_paddings(input_ids, labels):
    padding_left = input_ids.shape[1] - labels['mask'].shape[1]
    if padding_left == 0:
        return labels
    pad_keys = ['old_logprobs','advantages','mask','old_values','returns']
    no_pad_keys = ['input_ids']
    new_labels={}
    for key in pad_keys:
        if key in labels:
            new_labels[key] = torch.nn.functional.pad(labels[key], (padding_left, 0), mode='constant', value=0)
    for key in no_pad_keys:
        if key in labels:
            new_labels[key] = labels[key]
    return new_labels


def split_for_sp(batch, labels, padding_value_dict={'tensor_label': -100, 'default': 0}):
    batch, labels = pad_for_sp(batch, labels, padding_value_dict)
    sp_group = get_sequence_parallel_group()
    for key in batch.keys():
        if key == 'attention_mask':
            continue
        if batch[key] is not None:
            batch[key] = split_for_sequence_parallel(batch[key], dim=1, sp_group=sp_group)
    if isinstance(labels, torch.Tensor):
        labels = split_for_sequence_parallel(labels, dim=1, sp_group=sp_group)
    elif isinstance(labels, dict):
        for key in labels.keys():
            labels[key] = split_for_sequence_parallel(labels[key], dim=1, sp_group=sp_group)
    return batch, labels

def pad_for_sp(batch, labels, padding_value_dict):
    for key in batch.keys():
        if batch[key] is not None:
            batch[key] = pad_for_sequence_parallel(batch[key], padding_value=padding_value_dict.get(key, padding_value_dict['default']), dim=-1)
    if isinstance(labels, torch.Tensor):
        labels = pad_for_sequence_parallel(labels, padding_value=padding_value_dict.get('tensor_label', -100), dim=-1)
    elif isinstance(labels, dict):
        for key in labels.keys():
            labels[key] = pad_for_sequence_parallel(labels[key], padding_value=padding_value_dict.get(key, padding_value_dict['default']), dim=-1)
    return batch, labels

def add_dispatch_config_if_needed(conf: Config) -> Config:
    train_config = conf.get('train_config',{})
    use_varlen_attn = train_config.get('use_varlen_attn', False)
    model_configs = conf.get('model_configs',[])
    enable_sp = False
    for _, model_config in model_configs.items():
        if 'trainer_config' not in model_config:
            continue
        sp_size = get_sp_size(model_config['trainer_config'])
        if sp_size > 1:
            enable_sp = True
            break
    enable_xtuner_dispatch = use_varlen_attn or enable_sp
    if enable_xtuner_dispatch:
        logger.info(f"set enable_xtuner_dispatch = True")
        for _, model_config in model_configs.items():
            if 'trainer_config' not in model_config:
                continue
            else:
                model_config['trainer_config']['enable_xtuner_dispatch'] = True
    return conf
