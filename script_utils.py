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
import SimpleITK as sitk
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
from monai.data.utils import no_collation
from monai.networks.nets import resnet
from monai.transforms import ScaleIntensityRanged, NormalizeIntensityd
from monai.utils import set_determinism

from torch.utils.data.distributed import DistributedSampler

class CustomCacheDataset(CacheDataset):
    def __init__(
        self,
        data,
        train_transforms,
        val_transforms,
        cache_rate: float = 1.0,
        num_workers: int | None = 1,
        *args,
        **kwargs
    ):
        super().__init__(data, train_transforms, cache_rate, num_workers, *args, **kwargs)
        self.train_transforms = train_transforms
        self.val_transforms = val_transforms
        self._mode = 1

    def get_mode():
        return self._mode

    def __getitem__(self, index):
        if self._mode == 1:
            self.train_transforms(self.data[index])
        elif self._mode == 0:
            self.val_transforms(self.data[index])


def init_fn(worker_id, mode):
    info = get_worker_info()
    info.dataset._mode = mode.value
        
    

def load_data(args, training_key="training_chunks", validation_key="validation", cache_rate=1.0, smart_cache_ds=False, amp=True, use_thread=False):
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
    
    dataloader = ThreadDataLoader if use_thread else DataLoader 
    
    # ------------------------------------- Create datasets -------------------------------------
    if smart_cache_ds:
        train_ds = SmartCacheDataset(
                data=train_data,
                transform=train_transforms, 
                cache_rate=cache_rate,
                replace_rate=.2,
                num_init_workers=10,
                num_replace_workers=4,
                copy_cache = False,
        )

        val_ds = SmartCacheDataset(
                data=validation_data,
                transform=val_transforms, 
                cache_rate=cache_rate,
                replace_rate=.2,
                num_init_workers=10,
                num_replace_workers=4,
                copy_cache = False,
        )
    
    else:
        train_ds = CacheDataset(
            data=train_data,
            transform=train_transforms, 
            cache_rate=cache_rate,
            num_workers=10
        )

        val_ds = CacheDataset(
            data=validation_data,
            transform=val_transforms, 
            cache_rate=cache_rate,
            num_workers=10
        )

    # ------------------------------------- Create dataloaders -------------------------------------
    train_loader = dataloader(
        train_ds,
        batch_size=1,
        shuffle=True,
        num_workers=9,
        pin_memory=torch.cuda.is_available(),
        collate_fn=no_collation,
        persistent_workers=True
    )
    
    val_loader = dataloader(
        val_ds,
        batch_size=1,
        num_workers=5,
        pin_memory=torch.cuda.is_available(),
        collate_fn=no_collation,
        persistent_workers=True
    )

    end_time = time.time()

    print(f"Caching time: {end_time-start_time}s\n")

    epoch_len = len(train_ds) // train_loader.batch_size

    return train_loader, val_loader, epoch_len, train_ds, val_ds

def load_train_data(args, training_val_key="training", cache_rate=1.0, smart_cache_ds=False, amp=True, use_thread=False):
    #1. prepare transforms
    intensity_transform = NormalizeIntensityd(
        keys="image",                            
        nonzero=True,                             
        channel_wise=True                                    
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

    train_val_data = load_decathlon_datalist(
        args.data_list_file_path,
        is_segmentation=True,
        data_list_key=training_val_key,
        base_dir=args.data_base_dir,
    )

    print('Caching data ...')
    start_time = time.time()
    
    dataloader = ThreadDataLoader if use_thread else DataLoader 
    
    # ------------------------------------- Create datasets -------------------------------------
    if smart_cache_ds:
        train_val_ds = SmartCacheDataset(
                data=train_val_data,
                transform=val_transforms, 
                cache_rate=cache_rate,
                replace_rate=.2,
                num_init_workers=10,
                num_replace_workers=4,
                copy_cache = False,
        )
    elif cache_rate == None:
        train_val_ds = Dataset(
            data=train_val_data,
            transform=val_transforms
        )
    
    else:
        train_val_ds = CacheDataset(
            data=train_val_data,
            transform=val_transforms, 
            cache_rate=cache_rate,
            num_workers=10
        )
    

    # ------------------------------------- Create dataloader -------------------------------------
    
    train_val_loader = dataloader(
        train_val_ds,
        batch_size=1,
        shuffle=True,
        num_workers=1,
        pin_memory=torch.cuda.is_available(),
        collate_fn=no_collation,
        persistent_workers=True,
    )
    

    end_time = time.time()

    print(f"Caching time: {end_time-start_time}s\n")

    return train_val_loader, train_val_ds

def load_train_data2(args, training_val_key="training", cache_rate=1.0, smart_cache_ds=False, amp=True, use_thread=False):
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
    # 2. prepare training data
    # create a training data loader

    train_val_data = load_decathlon_datalist(
        args.data_list_file_path,
        is_segmentation=True,
        data_list_key=training_val_key,
        base_dir=args.data_base_dir,
    )

    print('Caching data ...')
    start_time = time.time()
    
    dataloader = ThreadDataLoader if use_thread else DataLoader 
    
    # ------------------------------------- Create datasets -------------------------------------
    if smart_cache_ds:
        train_val_ds = SmartCacheDataset(
                data=train_val_data,
                transform=train_transforms, 
                cache_rate=cache_rate,
                replace_rate=.2,
                num_init_workers=10,
                num_replace_workers=4,
                copy_cache = False,
        )
    elif cache_rate == None:
        train_val_ds = Dataset(
            data=train_val_data,
            transform=train_transforms
        )
    
    else:
        train_val_ds = CacheDataset(
            data=train_val_data,
            transform=train_transforms, 
            cache_rate=cache_rate,
            num_workers=10
        )
    

    # ------------------------------------- Create dataloader -------------------------------------
    
    train_val_loader = dataloader(
        train_val_ds,
        batch_size=1,
        shuffle=True,
        num_workers=12,
        pin_memory=torch.cuda.is_available(),
        collate_fn=no_collation
    )
    

    end_time = time.time()

    print(f"Caching time: {end_time-start_time}s\n")

    return train_val_loader, train_val_ds

def create_split(df, cv_split=None):
    splitter = GroupShuffleSplit(test_size=.2, n_splits=2, random_state = 0)
    split = splitter.split(df, groups=df.index.get_level_values(0)) #allows us to split the data according to the patient ID
    train_inds, valtest_inds = next(split)
    
    df_valtest = df.iloc[valtest_inds].copy()
    
    splitter = GroupShuffleSplit(test_size=.5, n_splits=2, random_state = 2)
    split = splitter.split(df_valtest, groups=df_valtest.index.get_level_values(0)) #allows us to split the data according to the patient ID
    
    val_inds, test_inds = next(split)
    
    df_train = df.iloc[train_inds].copy()
    df_val = df_valtest.iloc[val_inds].copy()
    df_test = df_valtest.iloc[test_inds].copy()
    
    if isinstance(cv_split, int):
        df_cv = pd.concat([df_train, df_val], copy=True)
        kfold = GroupKFold(n_splits=cv_split)

        df_cv_dict = dict([])
        for i, (train_index, val_index) in enumerate(kfold.split(df_cv, groups=df_cv.index.get_level_values(0))): 
            df_cv_dict[f"fold{i}"] = {
                "train": df_cv.iloc[train_index].copy(), 
                "val": df_cv.iloc[val_index].copy()
            }
            
            # check for duplicates
            a = df_cv.iloc[train_index].index.get_level_values(0).unique().tolist()
            b = df_cv.iloc[val_index].index.get_level_values(0).unique().tolist()
            if a in b:
                print("\n"+"-"*40+" WARNING "+"-"*40+"\n")
                print("Duplicate patient found\n")
                print("-"*90+"\n")

        return df_cv_dict, df_test
        
    else:
        return df_train, df_val, df_test

def create_split_parts(df_parts, df_chunks, cv_split=None):    
    df = df_chunks
    splitter = GroupShuffleSplit(test_size=.2, n_splits=2, random_state = 0)
    split = splitter.split(df, groups=df.index.get_level_values(0)) #allows us to split the data according to the patient ID
    train_inds, valtest_inds = next(split)
    
    df_valtest = df.iloc[valtest_inds].copy()
    
    splitter = GroupShuffleSplit(test_size=.5, n_splits=2, random_state = 2)
    split = splitter.split(df_valtest, groups=df_valtest.index.get_level_values(0)) #allows us to split the data according to the patient ID
    
    val_inds, test_inds = next(split)
    
    df_train = df.iloc[train_inds].copy()
    df_val = df_valtest.iloc[val_inds].copy()
    df_test = df_valtest.iloc[test_inds].copy()

    train_indices = df_train.index.get_level_values(0).unique()
    val_indices = df_val.index.get_level_values(0).unique()
    test_indices = df_test.index.get_level_values(0).unique()

    df_parts_train = df_parts.loc[train_indices]
    df_parts_val = df_parts.loc[val_indices]
    df_parts_test = df_parts.loc[test_indices]
    
    if isinstance(cv_split, int):
        df_cv = pd.concat([df_train, df_val], copy=True)
        kfold = GroupKFold(n_splits=cv_split)

        df_cv_dict = dict([])
        for i, (train_index, val_index) in enumerate(kfold.split(df_cv, groups=df_cv.index.get_level_values(0))): 
            df_cv_dict[f"fold{i}"] = {
                "train": df_cv.iloc[train_index].copy(), 
                "val": df_cv.iloc[val_index].copy()
            }
            
            # check for duplicates
            a = df_cv.iloc[train_index].index.get_level_values(0).unique().tolist()
            b = df_cv.iloc[val_index].index.get_level_values(0).unique().tolist()
            if a in b:
                print("\n"+"-"*40+" WARNING "+"-"*40+"\n")
                print("Duplicate patient found\n")
                print("-"*90+"\n")

        df_parts_cv_dict = dict([])

        for fold in df_cv_dict.keys():
            df_parts_cv_dict[fold] = dict([])
            for k in ['train', 'val']:
                indices = df_cv_dict[fold][k].index.get_level_values(0).unique()
                df_parts_cv_dict[fold][k] = df_parts.loc[indices].copy()

        return df_parts_cv_dict, df_parts_test

    else:
        return df_parts_train, df_parts_val, df_parts_test
        
def set_label(x, threshold):
    x['label'] = 0 if x['size'] < threshold else 1
    return x

def create_json_dataset(modalities, sizes=None, use_2=False):
    filename = 'dataset_ph_bounding_boxes_lps_chunks.csv'
    df = pd.read_csv(filename).set_index(['Folder', 'chunk'])
    df['label'] = np.zeros(len(df))
        
    if sizes == 'small':
        threshold = df['size'].describe()['50%']
        df = df[df['size'] <= threshold]

    elif sizes == 'large':
        threshold = df['size'].describe()['50%']
        df = df[df['size'] > threshold] 
        
    elif sizes == 'split':
        threshold = df['size'].describe()['50%']
        df = df.apply(axis=1, func=set_label, threshold=threshold)
        
    elif isinstance(sizes, list):
        threshold = str(sizes[1])
        operator = sizes[0]
        comparison = [f"{x} {operator} {threshold}" for x in df['size']]
        indices = [eval(html.unescape(x)) for x in comparison]
        df = df[indices]
        
    if use_2:
        df_train, df_val, df_test = create_split_2(df)
    else:
        df_train, df_val, df_test = create_split(df)
        
    training = {
        "training": [],
        "training_chunks": [],
        "validation": [],
        "validation_chunks": [],
        "test": [],
        "test_chunks": []
    }
    
    for df_x, training_key in zip([df_train, df_val, df_test], ["training_chunks", "validation_chunks", "test_chunks"]):
        for patient, chunk_nr in df_x.index.unique():
            dict_entry = {'box': [], 'label': []}
            
            if type(df_x.loc[patient, chunk_nr]) == pd.DataFrame:
                for _, row in df_x.loc[patient, chunk_nr].iterrows():
                    dict_entry['box'].append(row["x_center":"l"].tolist())
                    dict_entry["label"].append(int(row['label']))
            else:
                row = df_x.loc[patient, chunk_nr]
                dict_entry['box'].append(row["x_center":"l"].tolist())
                dict_entry["label"].append(int(row['label']))
                
            dict_entry['image'] = [f"{patient}/chunks/{modality}_resampled_chunk{chunk_nr}.nii.gz" for modality in modalities]
            training[training_key].append(dict_entry)

    for df_x, training_key in zip([df_train, df_val, df_test], ["training", "validation", "test"]):
        for patient in df_x.index.get_level_values(0).unique():
            dict_entry = {'box': [], 'label': []}
            
            if type(df_x.loc[patient]) == pd.DataFrame:
                for _, row in df_x.loc[patient].iterrows():
                    dict_entry['box'].append(row['x_center':'l'].tolist())
                    dict_entry["label"].append(int(row['label']))
                
            else:
                row = df_x.loc[patient]
                dict_entry['box'].append(row['x_center':'l'].tolist())
                dict_entry["label"].append(int(row['label']))
                
            dict_entry['image'] = [f"{patient}/{modality}_resampled.nii.gz" for modality in modalities]
            training[training_key].append(dict_entry)

    return training

def create_json_dataset_parts(modalities, sizes=None, part=None, use_2=False):
    filename = 'dataset_ph_bounding_boxes_lps_parts.csv'
    df = pd.read_csv(filename).set_index(['Folder', 'part'])
    empty_df = pd.read_csv('dataset_empty_bounding_boxes_parts.csv').set_index('Folder')
        
    df['label'] = np.zeros(len(df))
        
    if sizes == 'small':
        threshold = df['size'].describe()['50%']
        df = df[df['size'] <= threshold]

    elif sizes == 'large':
        threshold = df['size'].describe()['50%']
        df = df[df['size'] > threshold] 
        
    elif sizes == 'split':
        threshold = df['size'].describe()['50%']
        df = df.apply(axis=1, func=set_label, threshold=threshold)
        
    elif isinstance(sizes, list):
        threshold = str(sizes[1])
        operator = sizes[0]
        comparison = [f"{x} {operator} {threshold}" for x in df['size']]
        indices = [eval(html.unescape(x)) for x in comparison]
        df = df[indices]

    if use_2:
        df_train, df_val, df_test = create_split_2(df)
    else:
        df_train, df_val, df_test = create_split(df)

    if part!=None:
        df = df.xs(part, level=1, drop_level=False)
        empty_df = empty_df[(empty_df==part).values]

    df_train.sort_index()
    df_val.sort_index()
    df_test.sort_index()
        
    training = {
        "training": [],
        "training_parts": [],
        "validation": [],
        "validation_parts": [],
        "test": [],
        "test_parts": []
    }
    
    for df_x, training_key in zip([df_train, df_val, df_test], ["training_parts", "validation_parts", "test_parts"]):
        for patient, part in df_x.index.unique():
            dict_entry = {'box': [], 'label': []}
            
            if type(df_x.loc[patient, part]) == pd.DataFrame:
                for _, row in df_x.loc[patient, part].iterrows():
                    dict_entry['box'].append(row["x_center":"l"].tolist())
                    dict_entry["label"].append(int(row['label']))
            else:
                row = df_x.loc[patient, part]
                dict_entry['box'].append(row["x_center":"l"].tolist())
                dict_entry["label"].append(int(row['label']))
                
            dict_entry['image'] = [f"{patient}/parts/{modality}_{part}.nii.gz" for modality in modalities]
            training[training_key].append(dict_entry)

    #fill training_parts, validation_parts, and test_parts with images without bone metastases
    for df_x, training_key in zip([df_train, df_val, df_test], ["training_parts", "validation_parts", "test_parts"]):
        for patient in df_x.index.get_level_values(0).unique():
            if patient in empty_df.index.unique().tolist():
                empty_parts = empty_df.loc[patient]
                for empty_part in empty_parts:
                    dict_entry = {'box': [], 'label': []}
                    dict_entry['image'] = [f"{patient}/parts/{modality}_{empty_part}.nii.gz" for modality in modalities]
                    training[training_key].append(dict_entry)

    for df_x, training_key in zip([df_train, df_val, df_test], ["training", "validation", "test"]):
        for patient in df_x.index.get_level_values(0).unique():
            dict_entry = {'box': [], 'label': []}
            
            if type(df_x.loc[patient]) == pd.DataFrame:
                for _, row in df_x.loc[patient].iterrows():
                    dict_entry['box'].append(row['x_center':'l'].tolist())
                    dict_entry["label"].append(int(row['label']))
                
            else:
                row = df_x.loc[patient]
                dict_entry['box'].append(row['x_center':'l'].tolist())
                dict_entry["label"].append(int(row['label']))
                
            dict_entry['image'] = [f"{patient}/{modality}_resampled.nii.gz" for modality in modalities]
            training[training_key].append(dict_entry)

    return training

def find_empty_chunks(df):
    df_empty_chunks = pd.DataFrame(columns=['Folder', 'chunk'])
    for patient in df.index.get_level_values(0).unique():
        chunks = set(df.loc[patient].index.unique())
        empty_chunks = list({1, 2, 3, 4}.difference(chunks))
    
        for empty_chunk in empty_chunks:
            df_empty_chunks.loc[len(df_empty_chunks), :] = (patient, empty_chunk)

    return df_empty_chunks

def find_empty_parts(df):
    df_empty_parts = pd.DataFrame(columns=['Folder', 'part'])
    for patient in df.index.get_level_values(0).unique():
        parts = set(df.loc[patient].index.unique())

        if patient not in ['JSW-025a', 'JSW-025b']:
            empty_parts = list({'r_femur', 'l_femur', 'pelvis', 'spine'}.difference(parts))
        else:
            empty_parts = list({'r_femur', 'pelvis', 'spine'}.difference(parts))
    
        for part in empty_parts:
            df_empty_parts.loc[len(df_empty_parts), :] = (patient, part)

    return df_empty_parts
    

def create_json_cv_dataset(modalities, sizes=None, cv_split=5, use_2=False):
    filename = 'dataset_ph_bounding_boxes_lps_chunks.csv'
    df = pd.read_csv(filename).set_index(['Folder', 'chunk'])
    df['label'] = np.zeros(len(df))
        
    if sizes == 'small':
        threshold = df['size'].describe()['50%']
        df = df[df['size'] <= threshold]

    elif sizes == 'large':
        threshold = df['size'].describe()['50%']
        df = df[df['size'] > threshold] 
        
    elif sizes == 'split':
        threshold = df['size'].describe()['50%']
        df = df.apply(axis=1, func=set_label, threshold=threshold)
        
    elif isinstance(sizes, list):
        threshold = str(sizes[1])
        operator = sizes[0]
        comparison = [f"{x} {operator} {threshold}" for x in df['size']]
        indices = [eval(html.unescape(x)) for x in comparison]
        df = df[indices]
        
    if use_2:
        df_cv_dict, df_test = create_split_2(df, cv_split)
        empty_df = find_empty_chunks(df).set_index('Folder')
    else:
        df_cv_dict, df_test = create_split(df, cv_split)

    training = {
        "training": [],
        "training_chunks": [],
        "validation": [],
        "validation_chunks": []
    }

    cv_training = {fold: deepcopy(training) for fold in df_cv_dict.keys()}

    for fold in cv_training.keys():
        df_train, df_val = df_cv_dict[fold]['train'], df_cv_dict[fold]['val']
        
        for df_x, training_key in zip([df_train, df_val], ["training_chunks", "validation_chunks"]):
            for patient, chunk_nr in df_x.index.unique():
                dict_entry = {'box': [], 'label': []}
                
                if type(df_x.loc[patient, chunk_nr]) == pd.DataFrame:
                    for _, row in df_x.loc[patient, chunk_nr].iterrows():
                        dict_entry['box'].append(row["x_center":"l"].tolist())
                        dict_entry["label"].append(int(row['label']))
                else:
                    row = df_x.loc[patient, chunk_nr]
                    dict_entry['box'].append(row["x_center":"l"].tolist())
                    dict_entry["label"].append(int(row['label']))
                    
                dict_entry['image'] = [f"{patient}/chunks/{modality}_resampled_chunk{chunk_nr}.nii.gz" for modality in modalities]
                cv_training[fold][training_key].append(dict_entry)
                

        if use_2:
            #fill training_chunks with empty chunks
            for patient in df_train.index.get_level_values(0).unique():
                if patient in empty_df.index.unique().tolist():
                    empty_chunks = empty_df.loc[[patient]]
                    for _, row in empty_chunks.iterrows():
                        dict_entry = {'box': [], 'label': []}
                        dict_entry['image'] = [f"{patient}/chunks/{modality}_resampled_chunk{row['chunk']}.nii.gz" for modality in modalities]
                        cv_training[fold]["training_chunks"].append(dict_entry)
                        
    
        for df_x, training_key in zip([df_train, df_val], ["training", "validation"]):
            for patient in df_x.index.get_level_values(0).unique():
                dict_entry = {'box': [], 'label': []}
                
                if type(df_x.loc[patient]) == pd.DataFrame:
                    for _, row in df_x.loc[patient].iterrows():
                        dict_entry['box'].append(row['x_center':'l'].tolist())
                        dict_entry["label"].append(int(row['label']))
                    
                else:
                    row = df_x.loc[patient]
                    dict_entry['box'].append(row['x_center':'l'].tolist())
                    dict_entry["label"].append(int(row['label']))
                    
                dict_entry['image'] = [f"{patient}/{modality}_resampled.nii.gz" for modality in modalities]
                cv_training[fold][training_key].append(dict_entry)

    cv_training['test_chunks'] = []
    for patient, chunk_nr in df_test.index.unique():
        dict_entry = {'box': [], 'label': []}
        
        if type(df_test.loc[patient, chunk_nr]) == pd.DataFrame:
            for _, row in df_test.loc[patient, chunk_nr].iterrows():
                dict_entry['box'].append(row["x_center":"l"].tolist())
                dict_entry["label"].append(int(row['label']))
        else:
            row = df_test.loc[patient, chunk_nr]
            dict_entry['box'].append(row["x_center":"l"].tolist())
            dict_entry["label"].append(int(row['label']))
            
        dict_entry['image'] = [f"{patient}/chunks/{modality}_resampled_chunk{chunk_nr}.nii.gz" for modality in modalities]
        cv_training['test_chunks'].append(dict_entry)

    cv_training['test'] = []    
    for patient in df_test.index.get_level_values(0).unique():
        dict_entry = {'box': [], 'label': []}
        
        if type(df_test.loc[patient]) == pd.DataFrame:
            for _, row in df_test.loc[patient].iterrows():
                dict_entry['box'].append(row['x_center':'l'].tolist())
                dict_entry['label'].append(int(row['label']))
            
        else:
            row = df_train.loc[patient]
            dict_entry['box'].append(row['x_center':'l'].tolist())
            dict_entry['label'].append(int(row['label']))
            
        dict_entry['image'] = [f"{patient}/{modality}_resampled.nii.gz" for modality in modalities]
        cv_training['test'].append(dict_entry)

    return cv_training

def create_json_cv_dataset_parts(modalities, sizes=None, part=None, cv_split=5, use_2=False):
    if use_2:
        filename = 'dataset_ph_bounding_boxes_lps_parts_2.csv'
    else:
        filename = 'dataset_ph_bounding_boxes_lps_parts.csv'
        
    df = pd.read_csv(filename).set_index(['Folder', 'part'])
    df['label'] = np.zeros(len(df))

    empty_df = pd.read_csv('dataset_empty_bounding_boxes_parts.csv').set_index('Folder')
    
    df_chunks = pd.read_csv('./dataset_ph_bounding_boxes_lps_chunks.csv')
    df_chunks = df_chunks.set_index(['Folder', 'chunk'])
    df_chunks.insert(0, 'id', np.arange(len(df_chunks)))
    df_chunks['label'] = np.zeros(len(df_chunks))
    
    if sizes == 'small':
        threshold = df['size'].describe()['50%']
        df = df[df['size'] <= threshold]

    elif sizes == 'large':
        threshold = df['size'].describe()['50%']
        df = df[df['size'] > threshold] 
        
    elif sizes == 'split':
        threshold = df['size'].describe()['50%']
        df = df.apply(axis=1, func=set_label, threshold=threshold)
        
    elif isinstance(sizes, list):
        threshold = str(sizes[1])
        operator = sizes[0]
        comparison = [f"{x} {operator} {threshold}" for x in df['size']]
        indices = [eval(html.unescape(x)) for x in comparison]
        df = df[indices]
        
        comparison = [f"{x} {operator} {threshold}" for x in df_chunks['size']]
        indices = [eval(html.unescape(x)) for x in comparison]
        df_chunks = df_chunks[indices]

    if use_2:
        df_cv_dict, df_test = create_split_parts_2(df, df_chunks, cv_split)
        empty_df = find_empty_parts(df).set_index('Folder')
    else:
        df_cv_dict, df_test = create_split_parts(df, df_chunks, cv_split)

    if part!=None:
        for fold in df_cv_dict.keys():
            df_cv_dict[fold]['train'] = df_cv_dict[fold]['train'].xs(part, level=1, drop_level=False)
            df_cv_dict[fold]['val'] = df_cv_dict[fold]['val'].xs(part, level=1, drop_level=False)
            
        empty_df = empty_df[(empty_df==part).values]

    training = {
        "training": [],
        "training_parts": [],
        "validation": [],
        "validation_parts": []
    }

    cv_training = {fold: deepcopy(training) for fold in df_cv_dict.keys()}
    
    for fold in cv_training.keys():
        df_train, df_val = df_cv_dict[fold]['train'], df_cv_dict[fold]['val']
        df_train.sort_index()
        df_val.sort_index()
        
        for df_x, training_key in zip([df_train, df_val], ["training_parts", "validation_parts"]):
            for patient, part in df_x.index.unique():
                dict_entry = {'box': [], 'label': [], 'id': []}
                
                if type(df_x.loc[patient, part]) == pd.DataFrame:
                    for _, row in df_x.loc[patient, part].iterrows():
                        dict_entry['box'].append(row["x_center":"l"].tolist())
                        dict_entry["label"].append(int(row['label']))
                        dict_entry["id"].append(int(row['id']))
                else:
                    row = df_x.loc[patient, part]
                    dict_entry['box'].append(row["x_center":"l"].tolist())
                    dict_entry["label"].append(int(row['label']))
                    dict_entry["id"].append(int(row['id']))
                    
                dict_entry['image'] = [f"{patient}/parts/{modality}_{part}.nii.gz" for modality in modalities]
                cv_training[fold][training_key].append(dict_entry)

        #fill training_parts with images without bone metastases
        # for patient in df_train.index.get_level_values(0).unique():
        #     if patient in empty_df.index.unique().tolist():
        #         empty_parts = empty_df.loc[[patient]]
        #         for _, row in empty_parts.iterrows():
        #             dict_entry = {'box': [], 'label': []}
        #             dict_entry['image'] = [f"{patient}/parts/{modality}_{row['part']}.nii.gz" for modality in modalities]
        #             cv_training[fold]["training_parts"].append(dict_entry)

        #fill training_parts, validation_parts, and test_parts with images without bone metastases
        for df_x, training_key in zip([df_train, df_val], ["training_parts", "validation_parts"]):
            for patient in df_x.index.get_level_values(0).unique():
                if patient in empty_df.index.unique().tolist():
                    empty_parts = empty_df.loc[[patient]]
                    for _, row in empty_parts.iterrows():
                        dict_entry = {'box': [], 'label': [], 'id': []}
                        dict_entry['image'] = [f"{patient}/parts/{modality}_{row['part']}.nii.gz" for modality in modalities]
                        cv_training[fold][training_key].append(dict_entry)
                
            
    
        for df_x, training_key in zip([df_train, df_val], ["training", "validation"]):
            for patient in df_x.index.get_level_values(0).unique():
                dict_entry = {'box': [], 'label': []}
                
                if type(df_x.loc[patient]) == pd.DataFrame:
                    for _, row in df_x.loc[patient].iterrows():
                        dict_entry['box'].append(row['x_center':'l'].tolist())
                        dict_entry["label"].append(int(row['label']))
                    
                else:
                    row = df_x.loc[patient]
                    dict_entry['box'].append(row['x_center':'l'].tolist())
                    dict_entry["label"].append(int(row['label']))
                    
                dict_entry['image'] = [f"{patient}/{modality}_resampled.nii.gz" for modality in modalities]
                cv_training[fold][training_key].append(dict_entry)


    cv_training['test_parts'] = []
    for patient, part in df_test.index.unique():
        dict_entry = {'box': [], 'label': [], 'id': []}
        
        if type(df_test.loc[patient, part]) == pd.DataFrame:
            for _, row in df_test.loc[patient, part].iterrows():
                dict_entry['box'].append(row["x_center":"l"].tolist())
                dict_entry["label"].append(int(row['label']))
                dict_entry["id"].append(int(row['id']))
        else:
            row = df_test.loc[patient, part]
            dict_entry['box'].append(row["x_center":"l"].tolist())
            dict_entry["label"].append(int(row['label']))
            dict_entry["id"].append(int(row['id']))
            
        dict_entry['image'] = [f"{patient}/parts/{modality}_{part}.nii.gz" for modality in modalities]
        cv_training['test_parts'].append(dict_entry)

    cv_training['test'] = []
    for patient in df_test.index.get_level_values(0).unique():
        dict_entry = {'box': [], 'label': []}
        
        if type(df_test.loc[patient]) == pd.DataFrame:
            for _, row in df_test.loc[patient].iterrows():
                dict_entry['box'].append(row['x_center':'l'].tolist())
                dict_entry['label'].append(int(row['label']))
            
        else:
            row = df_train.loc[patient]
            dict_entry['box'].append(row['x_center':'l'].tolist())
            dict_entry['label'].append(int(row['label']))
            
        dict_entry['image'] = [f"{patient}/{modality}_resampled.nii.gz" for modality in modalities]
        cv_training['test'].append(dict_entry)

    #fill test_parts with images without bone metastases
    for patient in df_test.index.get_level_values(0).unique():
        if patient in empty_df.index.unique().tolist():
            empty_parts = empty_df.loc[[patient]]
            for _, row in empty_parts.iterrows():
                dict_entry = {'box': [], 'label': [], 'id': []}
                dict_entry['image'] = [f"{patient}/parts/{modality}_{row['part']}.nii.gz" for modality in modalities]
                cv_training["test_parts"].append(dict_entry)

    return cv_training

def prepare_json_files(args, modalities=["T1"], with_attention_transform=None, sizes=None, cv_split=None, base_anchor_shapes=None, additional_config_params=None, use_2=False):
    env_dict = json.load(open(args.environment_file, "r"))
    config_dict = json.load(open(args.config_file, "r"))

    config_dict['n_input_channels'] = len(modalities)
    config_dict['modalities'] = modalities
    config_dict['with_attention_transform'] = with_attention_transform
    config_dict['sizes'] = sizes
    
    if sizes == 'split':
        config_dict['fg_labels'] = [0, 1]
    else:
        config_dict['fg_labels'] = [0]
    
    if base_anchor_shapes != None:
        if isinstance(base_anchor_shapes[0], list) or isinstance(base_anchor_shapes[0], np.array):
            config_dict['base_anchor_shapes'].extend(base_anchor_shapes)
            
        elif isinstance(base_anchor_shapes[0], int):
            config_dict['base_anchor_shapes'].append(base_anchor_shapes)

    if cv_split == None:
        train_val = create_json_dataset(modalities=modalities, sizes=sizes)
        
        with open('train_val_script.json', 'w') as f:
            json.dump(train_val, f)
    else:
        train_val = create_json_cv_dataset(modalities=modalities, sizes=sizes, cv_split=cv_split, use_2=use_2)

        with open('train_val_folds_script.json', 'w') as f:
            json.dump(train_val, f)

    return env_dict, config_dict, train_val

def prepare_json_files_parts(args, modalities=["T1"], part=None, with_attention_transform=None, sizes=None, cv_split=None, base_anchor_shapes=None, additional_config_params=None, use_2=False):
    env_dict = json.load(open(args.environment_file, "r"))
    config_dict = json.load(open(args.config_file, "r"))

    config_dict['n_input_channels'] = len(modalities)
    config_dict['modalities'] = modalities
    config_dict['with_attention_transform'] = with_attention_transform
    config_dict['sizes'] = sizes
    
    if sizes == 'split':
        config_dict['fg_labels'] = [0, 1]
    else:
        config_dict['fg_labels'] = [0]
    
    if base_anchor_shapes != None:
        if isinstance(base_anchor_shapes[0], list) or isinstance(base_anchor_shapes[0], np.array):
            config_dict['base_anchor_shapes'].extend(base_anchor_shapes)
            
        elif isinstance(base_anchor_shapes[0], int):
            config_dict['base_anchor_shapes'].append(base_anchor_shapes)

    if cv_split == None:
        train_val = create_json_dataset_parts(modalities=modalities, sizes=sizes, part=part)
        
        with open('train_val_script.json', 'w') as f:
            json.dump(train_val, f)
    else:
        train_val = create_json_cv_dataset_parts(modalities=modalities, sizes=sizes, part=part, cv_split=cv_split, use_2=use_2)

        with open('train_val_folds_script.json', 'w') as f:
            json.dump(train_val, f)

    return env_dict, config_dict, train_val


def load_data2(args, amp=True, validation_key="validation"):
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
        with_attention_transform=args.with_attention_transform
    )
    
    val_transforms = generate_detection_val_transform(
        "image",
        "box",
        "label",
        args.gt_box_mode,
        intensity_transform,
        affine_lps_to_ras=True,
        amp=amp,
    )
    
    # 2. prepare training data
    # create a training data loader
    train_data = load_decathlon_datalist(
        args.data_list_file_path,
        is_segmentation=True,
        data_list_key="training",
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
    
    train_ds = Dataset(
        data=train_data,
        transform=train_transforms, 
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=1,
        shuffle=True,
        num_workers=7,
        pin_memory=torch.cuda.is_available(),
        collate_fn=no_collation,
        persistent_workers=True,
    )
    
    val_ds = Dataset(
        data=validation_data,
        transform=val_transforms, 
    )
    
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        num_workers=8,
        pin_memory=torch.cuda.is_available(),
        collate_fn=no_collation,
        persistent_workers=True,
    )
    
    end_time = time.time()
    
    print(f"Caching time: {end_time-start_time}s\n")
    
    epoch_len = len(train_ds) // train_loader.batch_size
    
    return train_loader, val_loader, epoch_len

def create_split_luna(df_train, df_val, cv_split):
    df_cv = pd.concat([df_train, df_val], copy=True)
    kfold = GroupKFold(n_splits=cv_split)

    df_cv_dict = dict([])
    for i, (train_index, val_index) in enumerate(kfold.split(df_cv, groups=df_cv.index.get_level_values(0))): 
        df_cv_dict[f"fold{i}"] = {
            "train": df_cv.iloc[train_index].copy(), 
            "val": df_cv.iloc[val_index].copy()
        }
        
        # check for duplicates
        a = df_cv.iloc[train_index].index.get_level_values(0).unique().tolist()
        b = df_cv.iloc[val_index].index.get_level_values(0).unique().tolist()
        if a in b:
            print("\n"+"-"*40+" WARNING "+"-"*40+"\n")
            print("Duplicate patient found\n")
            print("-"*90+"\n")

    return df_cv_dict

def create_json_cv_dataset_luna(train_val, cv_split):
    train = train_val['training']
    val = train_val['validation']
    
    df_train = pd.DataFrame(columns=['Folder', 'x_center', 'y_center', 'z_center', 'w', 'h', 'l'])
    df_val = pd.DataFrame(columns=['Folder', 'x_center', 'y_center', 'z_center', 'w', 'h', 'l'])
    
    for x in train:
        folder = x['image'].split('.nii')[0]
        for bbox in x['box']:
            df_train.loc[len(df_train), :] = (folder, *bbox)
    
    df_train = df_train.set_index('Folder')
    
    for x in val:
        folder = x['image'].split('.nii')[0]
        for bbox in x['box']:
            df_val.loc[len(df_val), :] = (folder, *bbox)
    
    df_val = df_val.set_index('Folder')

    df_train['label'] = np.zeros(len(df_train))
    df_val['label'] = np.zeros(len(df_val))

    
    df_cv_dict = create_split_luna(df_train, df_val, cv_split)

    training = {
        "training": [],
        "validation": [],
    }

    cv_training = {fold: deepcopy(training) for fold in df_cv_dict.keys()}
    
    for fold in cv_training.keys():
        df_train, df_val = df_cv_dict[fold]['train'], df_cv_dict[fold]['val']            
    
        for df_x, training_key in zip([df_train, df_val], ["training", "validation"]):
            for patient in df_x.index.get_level_values(0).unique():
                dict_entry = {'box': [], 'label': []}
                
                if type(df_x.loc[patient]) == pd.DataFrame:
                    for _, row in df_x.loc[patient].iterrows():
                        dict_entry['box'].append(row['x_center':'l'].tolist())
                        dict_entry["label"].append(int(row['label']))
                    
                else:
                    row = df_x.loc[patient]
                    dict_entry['box'].append(row['x_center':'l'].tolist())
                    dict_entry["label"].append(int(row['label']))
                    
                dict_entry['image'] = patient+'.nii.gz'
                cv_training[fold][training_key].append(dict_entry)

    return cv_training


def prepare_json_files_luna(args, cv_split=None):
    env_dict = json.load(open(args.environment_file, "r"))
    config_dict = json.load(open(args.config_file, "r"))
    train_val = json.load(open('./LUNA16_datasplit/dataset_fold_mbd.json', "r"))
    
    train_val_fold = create_json_cv_dataset_luna(train_val, cv_split=cv_split)
    
    return env_dict, config_dict, train_val_fold


def load_luna16(args, cache_rate=1.0, amp=True):
        
    # 1. define transform
    intensity_transform = ScaleIntensityRanged(
        keys=["image"],
        a_min=-1024,
        a_max=300.0,
        b_min=0.0,
        b_max=1.0,
        clip=True,
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
        with_attention_transform=None
    )
    
    val_transforms = generate_detection_val_transform(
        "image",
        "box",
        "label",
        args.gt_box_mode,
        intensity_transform,
        affine_lps_to_ras=True,
        amp=amp,
        with_attention_transform=None
    )
    
    # 2. prepare training data
    # create a training data loader
    train_data = load_decathlon_datalist(
        args.data_list_file_path,
        is_segmentation=True,
        data_list_key=args.training_key,
        base_dir=args.data_base_dir,
    )
    
    validation_data = load_decathlon_datalist(
        args.data_list_file_path,
        is_segmentation=True,
        data_list_key=args.validation_key,
        base_dir=args.data_base_dir,
    )
    
    print('Caching data ...')
    start_time = time.time()
    
    train_ds = CacheDataset(
        data=train_data,
        transform=train_transforms,
        cache_rate=1.0,
        num_workers=8
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=1,
        shuffle=True,
        num_workers=9,
        pin_memory=torch.cuda.is_available(),
        collate_fn=no_collation,
        persistent_workers=True,
    )
    
    val_ds = CacheDataset(
        data=validation_data,
        transform=val_transforms, 
        cache_rate=1.0,
        num_workers=8
    )
    
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        num_workers=5,
        pin_memory=torch.cuda.is_available(),
        collate_fn=no_collation,
        persistent_workers=True,
    )
    
    end_time = time.time()
    
    print(f"Caching time: {end_time-start_time}s\n")
    
    epoch_len = len(train_ds) // train_loader.batch_size
    
    return train_loader, val_loader, epoch_len

def load_luna16_val(args, cache_rate=1.0, amp=True):
        
    # 1. define transform
    intensity_transform = ScaleIntensityRanged(
        keys=["image"],
        a_min=-1024,
        a_max=300.0,
        b_min=0.0,
        b_max=1.0,
        clip=True,
    )
    val_transforms = generate_detection_val_transform(
        "image",
        "box",
        "label",
        args.gt_box_mode,
        intensity_transform,
        affine_lps_to_ras=True,
        amp=amp,
        with_attention_transform=None
    )
    
    # 2. prepare training data
    # create a training data loader
    validation_data = load_decathlon_datalist(
        args.data_list_file_path,
        is_segmentation=True,
        data_list_key=args.validation_key,
        base_dir=args.data_base_dir,
    )
    
    print('Caching data ...')
    start_time = time.time()
    
    val_ds = CacheDataset(
        data=validation_data,
        transform=val_transforms, 
        cache_rate=1.0,
        num_workers=8
    )
    
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        num_workers=12,
        pin_memory=torch.cuda.is_available(),
        collate_fn=no_collation,
        persistent_workers=True,
    )
    
    end_time = time.time()
    
    print(f"Caching time: {end_time-start_time}s\n")
    
    return val_loader, val_ds

def assemble_parts(preds, gts):
    patients = [x['image'][0].split('/')[0] for x in gts]
    
    new_preds = {patient: None for patient in patients}
    new_gts = {patient: None for patient in patients}
    
    for pred, gt in zip(preds, gts):
        patient = gt['image'][0].split('/')[0]
    
        if new_preds[patient] != None:
            new_gts[patient]['box'] = new_gts[patient]['box'] + gt['box']
            new_gts[patient]['id'] = new_gts[patient]['id'] + gt['id']
            new_gts[patient]['label'] = new_gts[patient]['label'] + gt['label']
            for key in pred.keys():
                if len(pred[key].shape) > 1:
                    new_preds[patient][key] = torch.vstack([new_preds[patient][key].cpu().detach(), pred[key].cpu().detach()])
                else:
                    new_preds[patient][key] = torch.cat((new_preds[patient][key].cpu().detach(), pred[key].cpu().detach()))
        else:
            new_preds[patient] = pred
            new_gts[patient] = gt
            del new_gts[patient]['image_path']
            new_gts[patient]['image'] = [f"{x.split('/')[0]}/{x.split('/')[2].split('_')[0]}_resampled.nii.gz" for x in gt['image']]
    
    new_preds = list(new_preds.values())
    new_gts = list(new_gts.values())
    
    for i, x in enumerate(new_preds):
        _, index = np.unique(x['box'].numpy(), axis=0, return_index=True)
        sorted_index = x['label_scores'][index].argsort(descending=True)
        for key in x.keys():
            new_preds[i][key] = x[key][index][sorted_index][:150]
    
    for i, x in enumerate(new_gts):
        _, index = np.unique(x['id'], return_index=True)
        new_gts[i]['box'] = np.array(x['box'])[index].tolist()
        new_gts[i]['id'] = np.array(x['id'])[index].tolist()
        new_gts[i]['label'] = np.array(x['label'])[index].tolist()

    return new_preds, new_gts


def create_split_2(df, cv_split=None):
    splitter = GroupShuffleSplit(test_size=.2, n_splits=2, random_state = 0)
    split = splitter.split(df, groups=df.index.get_level_values(0).to_series().apply(lambda x: x[:-1])) #allows us to split the data according to the patient ID
    train_inds, valtest_inds = next(split)
    
    df_valtest = df.iloc[valtest_inds].copy()
    
    splitter = GroupShuffleSplit(test_size=.5, n_splits=2, random_state = 2)
    split = splitter.split(df_valtest, groups=df_valtest.index.get_level_values(0).to_series().apply(lambda x: x[:-1])) #allows us to split the data according to the patient ID
    
    val_inds, test_inds = next(split)
    
    df_train = df.iloc[train_inds].copy()
    df_val = df_valtest.iloc[val_inds].copy()
    df_test = df_valtest.iloc[test_inds].copy()
    
    if isinstance(cv_split, int):
        df_cv = pd.concat([df_train, df_val], copy=True)
        kfold = GroupKFold(n_splits=cv_split)

        df_cv_dict = dict([])
        for i, (train_index, val_index) in enumerate(kfold.split(df_cv, groups=df_cv.index.get_level_values(0).to_series().apply(lambda x: x[:-1]))): 
            df_cv_dict[f"fold{i}"] = {
                "train": df_cv.iloc[train_index].copy(), 
                "val": df_cv.iloc[val_index].copy()
            }
            
            # check for duplicates
            a = df_cv.iloc[train_index].index.get_level_values(0).unique().tolist()
            b = df_cv.iloc[val_index].index.get_level_values(0).unique().tolist()
            if a in b:
                print("\n"+"-"*40+" WARNING "+"-"*40+"\n")
                print("Duplicate patient found\n")
                print("-"*90+"\n")

        return df_cv_dict, df_test
        
    else:
        return df_train, df_val, df_test

def create_split_parts_2(df_parts, df_chunks, cv_split=None):    
    df = df_chunks
    splitter = GroupShuffleSplit(test_size=.2, n_splits=2, random_state = 0)
    split = splitter.split(df, groups=df.index.get_level_values(0).to_series().apply(lambda x: x[:-1])) #allows us to split the data according to the patient ID
    train_inds, valtest_inds = next(split)
    
    df_valtest = df.iloc[valtest_inds].copy()
    
    splitter = GroupShuffleSplit(test_size=.5, n_splits=2, random_state = 2)
    split = splitter.split(df_valtest, groups=df_valtest.index.get_level_values(0).to_series().apply(lambda x: x[:-1])) #allows us to split the data according to the patient ID
    
    val_inds, test_inds = next(split)
    
    df_train = df.iloc[train_inds].copy()
    df_val = df_valtest.iloc[val_inds].copy()
    df_test = df_valtest.iloc[test_inds].copy()

    train_indices = df_train.index.get_level_values(0).unique()
    val_indices = df_val.index.get_level_values(0).unique()
    test_indices = df_test.index.get_level_values(0).unique()

    df_parts_train = df_parts.loc[train_indices]
    df_parts_val = df_parts.loc[val_indices]
    df_parts_test = df_parts.loc[test_indices]
    
    if isinstance(cv_split, int):
        df_cv = pd.concat([df_train, df_val], copy=True)
        kfold = GroupKFold(n_splits=cv_split)

        df_cv_dict = dict([])
        for i, (train_index, val_index) in enumerate(kfold.split(df_cv, groups=df_cv.index.get_level_values(0).to_series().apply(lambda x: x[:-1]))): 
            df_cv_dict[f"fold{i}"] = {
                "train": df_cv.iloc[train_index].copy(), 
                "val": df_cv.iloc[val_index].copy()
            }
            
            # check for duplicates
            a = df_cv.iloc[train_index].index.get_level_values(0).unique().tolist()
            b = df_cv.iloc[val_index].index.get_level_values(0).unique().tolist()
            if a in b:
                print("\n"+"-"*40+" WARNING "+"-"*40+"\n")
                print("Duplicate patient found\n")
                print("-"*90+"\n")

        df_parts_cv_dict = dict([])

        for fold in df_cv_dict.keys():
            df_parts_cv_dict[fold] = dict([])
            for k in ['train', 'val']:
                indices = df_cv_dict[fold][k].index.get_level_values(0).unique()
                df_parts_cv_dict[fold][k] = df_parts.loc[indices].copy()

        return df_parts_cv_dict, df_parts_test

    else:
        return df_parts_train, df_parts_val, df_parts_test

# def load_luna16_2(args, cache_rate=1.0, amp=True):
#     setattr(args, 'environment_file', './config/environment_luna16_fold_mbd.json')
#     setattr(args, 'config_file', './config/config_train_luna16_16g.json')
    
#     env_dict = json.load(open(args.environment_file, "r"))
#     config_dict = json.load(open(args.config_file, "r"))

#     train_val = json.load(open(env_dict['data_list_file_path'], "r"))

#     with open('train_val_luna_script.json', 'w') as f:
#         json.dump(train_val, f)
        
#     for k, v in env_dict.items():
#         setattr(args, k, v)
#     for k, v in config_dict.items():
#         setattr(args, k, v)
        
#     # 1. define transform
#     intensity_transform = ScaleIntensityRanged(
#         keys=["image"],
#         a_min=-1024,
#         a_max=300.0,
#         b_min=0.0,
#         b_max=1.0,
#         clip=True,
#     )
#     train_transforms = generate_detection_train_transform(
#         "image",
#         "box",
#         "label",
#         args.gt_box_mode,
#         intensity_transform,
#         args.patch_size,
#         args.batch_size,
#         affine_lps_to_ras=True,
#         amp=amp,
#         with_attention_transform=None
#     )
    
#     val_transforms = generate_detection_val_transform(
#         "image",
#         "box",
#         "label",
#         args.gt_box_mode,
#         intensity_transform,
#         affine_lps_to_ras=True,
#         amp=amp,
#         with_attention_transform=None
#     )
    
#     # 2. prepare training data
#     # create a training data loader
#     train_data = load_decathlon_datalist(
#         args.data_list_file_path,
#         is_segmentation=True,
#         data_list_key="training",
#         base_dir=args.data_base_dir,
#     )
    
#     validation_data = load_decathlon_datalist(
#         args.data_list_file_path,
#         is_segmentation=True,
#         data_list_key="validation",
#         base_dir=args.data_base_dir,
#     )
    
#     print('Caching data ...')
#     start_time = time.time()
    
#     train_ds = CacheDataset(
#         data=train_data,
#         transform=train_transforms,
#         cache_rate=1.0,
#         num_workers=8
#     )
#     train_loader = DataLoader(
#         train_ds,
#         batch_size=1,
#         shuffle=True,
#         num_workers=5,
#         pin_memory=torch.cuda.is_available(),
#         collate_fn=no_collation,
#         persistent_workers=True,
#     )
    
#     val_ds = CacheDataset(
#         data=validation_data,
#         transform=val_transforms, 
#         cache_rate=1.0,
#         num_workers=8
#     )
    
#     val_loader = DataLoader(
#         val_ds,
#         batch_size=1,
#         num_workers=5,
#         pin_memory=torch.cuda.is_available(),
#         collate_fn=no_collation,
#         persistent_workers=True,
#     )
    
#     end_time = time.time()
    
#     print(f"Caching time: {end_time-start_time}s\n")
    
#     epoch_len = len(train_ds) // train_loader.batch_size
    
#     return args, env_dict, config_dict, train_val, train_loader, val_loader, epoch_len

