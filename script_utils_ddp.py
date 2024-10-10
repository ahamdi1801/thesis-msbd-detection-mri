import argparse
import gc
import json
import logging
import sys
import time
import os

import numpy as np
import pandas as pd
import torch
import json
import html
from copy import deepcopy
from functools import partial
from multiprocessing import Value
from sklearn.model_selection import train_test_split, GroupShuffleSplit, GroupKFold
from generate_transforms import (
    generate_detection_train_transform,
    generate_detection_val_transform,
)
from torch.utils.data import get_worker_info

import monai
from monai.apps.detection.metrics.coco import COCOMetric
from monai.apps.detection.metrics.matching import matching_batch
from monai.apps.detection.networks.retinanet_detector import RetinaNetDetector
from monai.apps.detection.networks.retinanet_network import (
    RetinaNet,
    resnet_fpn_feature_extractor,
)
from monai.apps.detection.utils.anchor_utils import AnchorGeneratorWithAnchorShape
from monai.data import DataLoader, ThreadDataLoader, CacheDataset, SmartCacheDataset, Dataset, box_utils, load_decathlon_datalist
from monai.data.utils import no_collation, partition_dataset
from monai.networks.nets import resnet
from monai.transforms import ScaleIntensityRanged, NormalizeIntensityd

import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

class DDP_multinode_CacheDataset(CacheDataset):
    """CacheDataset for DistributedDataParallel training."""
    def __init__(self, world_size: int, data_list: list, transform, cache_rate: float = 0, num_workers: int=0):
        """
        Args:
            data: data to cache.
            transform: transform to apply to data.
            cache_rate: cache rate for data.
            num_workers: number of workers for data.
        """

        part_data = self._get_part_data_list(data_list)
        
        print(f"Data count: {len(part_data)} for local rank {dist.get_rank()}, world size: {dist.get_world_size()}, total number: {len(data_list)}")
    
        super().__init__(part_data, transform, cache_rate=cache_rate, num_workers=num_workers)

    def _get_part_data_list(self, data_list: list) -> list:
        """
        Get part of data list for each rank.
        Args:
            data_list: list of data.
        """
        return partition_dataset(
            data=data_list,
            num_partitions=dist.get_world_size(),
            shuffle=True,
            seed=42,
            drop_last=False,
            even_divisible=True,
        )[dist.get_rank()]

# https://github.com/Lightning-AI/pytorch-lightning/discussions/11763
class DDP_CacheDataset(CacheDataset):
    """CacheDataset for DistributedDataParallel training."""
    def __init__(self, data_list: list, transform, cache_rate: float = 0, num_workers: int=0):
        """
        Args:
            data: data to cache.
            transform: transform to apply to data.
            cache_rate: cache rate for data.
            num_workers: number of workers for data.
        """
        if dist.is_initialized():
            part_data = self._get_part_data_list(data_list)
            print(f"Data count: {len(part_data)} for local rank {dist.get_rank()}, world size: {dist.get_world_size()}, total number: {len(data_list)}")
        else:
            part_data = data_list
        
        super().__init__(part_data, transform, cache_rate=cache_rate, num_workers=num_workers)

    def _get_part_data_list(self, data_list: list) -> list:
        """
        Get part of data list for each rank.
        Args:
            data_list: list of data.
        """
        return partition_dataset(
            data=data_list,
            num_partitions=dist.get_world_size(),
            shuffle=True,
            seed=42,
            drop_last=False,
            even_divisible=True,
        )[dist.get_rank()]
        

def load_data_ddp(args, rank, world_size, training_key="training_chunks", validation_key="validation", cache_rate=1.0, amp=True):
    #1. prepare transforms
    intensity_transform = NormalizeIntensityd(
        keys="image",                            
        nonzero=True,                             
        channel_wise=True                                    
    )
    
    train_transforms = generate_detection_train_transform(
        "image",
        "box",
        "label",
        args.gt_box_mode,
        intensity_transform,
        args.patch_size,
        args.batch_size,
        affine_lps_to_ras=True,
        amp=amp,
        with_attention_transform=args.with_attention_transform,
        with_binary=args.with_binary
    )

    val_transforms = generate_detection_val_transform(
        "image",
        "box",
        "label",
        args.gt_box_mode,
        intensity_transform,
        affine_lps_to_ras=True,
        amp=amp,
        with_attention_transform=args.with_attention_transform,
        with_binary=args.with_binary
    )

    # 2. prepare training data
    # create a training data loader
    train_data = load_decathlon_datalist(
        args.data_list_file_path,
        is_segmentation=True,
        data_list_key=training_key,
        base_dir=args.data_base_dir,
    )

    validation_data = load_decathlon_datalist(
        args.data_list_file_path,
        is_segmentation=True,
        data_list_key=validation_key,
        base_dir=args.data_base_dir,
    )

    print('Caching data ...')
    start_time = time.time()
    
    # ------------------------------------- Create datasets -------------------------------------
    train_ds = DDP_CacheDataset(
        data_list=train_data,
        transform=train_transforms, 
        cache_rate=cache_rate,
        num_workers=10
    )

    if rank == 0:
        val_ds = CacheDataset(
            data=validation_data,
            transform=val_transforms, 
            cache_rate=cache_rate,
            num_workers=10
        )
    else:
        val_ds = None

    # ------------------------------------- Create dataloaders -------------------------------------
    # train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank)
    # val_sampler = DistributedSampler(val_ds, num_replicas=world_size, rank=rank)
    
    train_loader = DataLoader(
        train_ds,
        batch_size=1,
        shuffle=True,
        num_workers=10,
        pin_memory=torch.cuda.is_available(),
        collate_fn=no_collation,
        persistent_workers=True
    )

    if rank == 0:
        val_loader = DataLoader(
            val_ds,
            batch_size=1,
            num_workers=5,
            pin_memory=torch.cuda.is_available(),
            collate_fn=no_collation,
            persistent_workers=True
        )
    else:
        val_loader = None

    end_time = time.time()

    print(f"Caching time: {end_time-start_time}s\n")

    epoch_len = len(train_ds) // train_loader.batch_size

    return train_loader, val_loader, epoch_len, train_ds, val_ds

def load_data_ddp_multinode(args, rank, world_size, training_key="training_chunks", validation_key="validation", cache_rate=1.0, amp=True):
    #1. prepare transforms
    intensity_transform = NormalizeIntensityd(
        keys="image",                            
        nonzero=True,                             
        channel_wise=True                                    
    )
    
    train_transforms = generate_detection_train_transform(
        "image",
        "box",
        "label",
        args.gt_box_mode,
        intensity_transform,
        args.patch_size,
        args.batch_size,
        affine_lps_to_ras=True,
        amp=amp,
        with_attention_transform=args.with_attention_transform,
        with_binary=args.with_binary
    )

    val_transforms = generate_detection_val_transform(
        "image",
        "box",
        "label",
        args.gt_box_mode,
        intensity_transform,
        affine_lps_to_ras=True,
        amp=amp,
        with_attention_transform=args.with_attention_transform,
        with_binary=args.with_binary
    )

    # 2. prepare training data
    # create a training data loader
    train_data = load_decathlon_datalist(
        args.data_list_file_path,
        is_segmentation=True,
        data_list_key=training_key,
        base_dir=args.data_base_dir,
    )

    validation_data = load_decathlon_datalist(
        args.data_list_file_path,
        is_segmentation=True,
        data_list_key=validation_key,
        base_dir=args.data_base_dir,
    )

    print('Caching data ...')
    start_time = time.time()
    
    # ------------------------------------- Create datasets -------------------------------------
    train_ds = DDP_CacheDataset(
        world_size=world_size,
        data_list=train_data,
        transform=train_transforms, 
        cache_rate=cache_rate,
        num_workers=10
    )

    if rank == 0:
        val_ds = CacheDataset(
            data_list=validation_data,
            transform=val_transforms, 
            cache_rate=cache_rate,
            num_workers=10
        )
    else:
        val_ds = None

    # ------------------------------------- Create dataloaders -------------------------------------
    # train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank)
    # val_sampler = DistributedSampler(val_ds, num_replicas=world_size, rank=rank)
    
    train_loader = DataLoader(
        train_ds,
        batch_size=1,
        shuffle=False,
        num_workers=9,
        pin_memory=torch.cuda.is_available(),
        collate_fn=no_collation,
        persistent_workers=True
    )

    if rank == 0:
        val_loader = DataLoader(
            val_ds,
            batch_size=1,
            num_workers=5,
            pin_memory=torch.cuda.is_available(),
            collate_fn=no_collation,
            persistent_workers=True
        )
    else:
        val_loader = None

    end_time = time.time()

    print(f"Caching time: {end_time-start_time}s\n")

    epoch_len = len(train_ds) // train_loader.batch_size

    return train_loader, val_loader, epoch_len, train_ds, val_ds

def ddp_setup(rank, world_size):    
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '8888'    
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    return

def ddp_multinode_setup():    
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    return



def cleanup():
    dist.destroy_process_group()
