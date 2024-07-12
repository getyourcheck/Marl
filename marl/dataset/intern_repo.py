# adapted from https://github.com/InternLM/xtuner/blob/main/xtuner/dataset/intern_repo.py

from xtuner.dataset.intern_repo import *


def build_dataset_rank0(dataset_cfg, packed=True, max_length=8192, seed=1024):
    if isinstance(dataset_cfg, dict):
        datasets = BUILDER.build(dataset_cfg)
    else:
        datasets = dataset_cfg

    if not isinstance(datasets, list):
        datasets = [datasets]

    new_datasets = []

    for dataset in datasets:
        if packed:
            ds = PackedDataset(dataset, max_length, seed=seed)
        else:
            ds = dataset # UnPackedDataset(dataset, max_length, seed=seed)
        new_datasets.append(ds)

    dataset = ConcatDataset(datasets=new_datasets)

    return dataset


def build_dataset(*args, **kwargs):
    if not (dist.is_available() and dist.is_initialized()):
        return build_dataset_rank0(*args, **kwargs)

    if dist.get_rank() == 0:
        dataset = build_dataset_rank0(*args, **kwargs)
        objects = [dataset]
    else:
        objects = [None]
    dist.broadcast_object_list(objects, src=0)
    return objects[0]


def packed_collate_fn(batch, max_length, batch_size):
    input_ids, labels, cumulative_len, position_ids, ts = [], [], [], [], []
    for b in batch:
        assert (len(b["input_ids"]) == max_length)
        assert (len(b["labels"]) == max_length)
        assert (len(b["position_ids"]) == max_length)

        input_id = [abs(w) for w in b["input_ids"]]
        label = [w if w > 0 else -100 for w in b["labels"]]

        input_ids.append(torch.LongTensor(input_id))
        # The labels have been shifted here, so they are aligned with the output corresponding to the token
        labels.append(torch.LongTensor(label))
        cumulative_len.append(b["cumulative_len"])
        position_ids.append(torch.LongTensor(b["position_ids"]))

    input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True)
    labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)
    position_ids = torch.stack(position_ids, dim=0)
    if len(set(map(len, cumulative_len))) == 1:  # if has uniform length, then stack to save device transfer time
        cumulative_len = torch.stack(cumulative_len, dim=0)

    assert input_ids.shape[1] == max_length, (input_ids.shape[1], max_length)
    assert input_ids.shape[0] == batch_size, (input_ids.shape[0], batch_size)

    return {"input_ids": input_ids, "cumulative_len": cumulative_len, "position_ids": position_ids, "labels": labels}

def batch_collate_fn(batch, max_length, batch_size):
    input_ids, labels, attention_mask = [], [], []
    done = False
    while not done:
        for b in batch:
            input_id = [abs(w) for w in b["input_ids"]]
            label = list(input_id[1:]) + [-100]
            # labels = deepcopy(input_ids)
            attn = [True for _ in b["input_ids"]]
            attn[-1] = False
            # print("============", len(input_id))
            if len(input_id) >= max_length:
                continue
            input_ids.append(torch.LongTensor(input_id))
            # The labels have been shifted here, so they are aligned with the output corresponding to the token
            labels.append(torch.LongTensor(label))
            attention_mask.append(torch.BoolTensor(attn))
            if len(input_ids) == batch_size:
                done = True
                break
    input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True)
    labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)
    attention_mask = torch.nn.utils.rnn.pad_sequence(attention_mask, batch_first=True, padding_value=False)
    return {"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask}


def get_pretrain_data(folder=None, packed=True, max_length=8192, batch_size=32, seed=1024, file_type='.bin', min_length=0, shuffle=True):
    if (folder is None) or (batch_size <= 0):
        return None
    import functools
    from torch.utils.data import DataLoader
    dataset_cfg = dict(
        type=load_intern_repo_tokenized_dataset,
        folder=folder,
        min_length=min_length,
        file_type=file_type
    )
    
    pretrain_dataset = build_dataset(dataset_cfg, packed=packed, max_length=max_length, seed=seed)

    collate_fn = packed_collate_fn if packed else batch_collate_fn

    pretrain_dataloader = DataLoader(
            shuffle=shuffle,
            dataset=pretrain_dataset,
            batch_size=batch_size,
            collate_fn=functools.partial(collate_fn, max_length=max_length, batch_size=batch_size),
        )

    pretrain_data_iterator = pretrain_dataloader.__iter__()
    return pretrain_data_iterator


if __name__ == "__main__":
    # dataset_cfg = dict(
    #     type=load_intern_repo_tokenized_dataset,
    #     folder='/fs-computility/llm/shared/zhaoqian/dataset/pretrain/1226-mix-v13-complete-watermark-pjx50/train',
    #     min_length=0,
    #     file_type='.bin'
    # )

    # dataset = build_dataset(dataset_cfg, packed=True, max_length=1024, seed=1024)
    # print("=============")
    # i = next(iter(dataset))

    # for k, v in i.items():
    #     try:
    #         print(k, v.shape)
    #     except:
    #         print(k, len(v))
    # exit()
    # pretrain_data_iterator = get_pretrain_data(folder='/fs-computility/llm/shared/zhaoqian/dataset/pretrain/1226-mix-v13-complete-watermark-pjx50/train',
    #                                             packed=True,
    #                                             max_length=8192,
    #                                             batch_size=32,
    #                                             )
    pretrain_dataset_config = dict(
            folder='/fs-computility/llm/shared/zhaoqian/dataset/pretrain/1226-mix-v13-complete-watermark-pjx50/train',
            packed=False,
            max_length=8192,
            batch_size=32,
    )

    # pretrain_dataset_config = {}
    pretrain_data_iterator = get_pretrain_data(**pretrain_dataset_config)

    pretrain_data = next(pretrain_data_iterator) if pretrain_data_iterator is not None else None

    import numpy as np
    import torch
    torch.set_printoptions(threshold=np.inf)

    print(len(pretrain_data))
    for k in pretrain_data.keys():
        if k != 'cumulative_len':
            print(k, pretrain_data[k].shape,)# pretrain_data[k][0])

    # print(pretrain_data['cumulative_len'])
    print(pretrain_data.keys())
