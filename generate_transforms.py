# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import fnmatch
import torch
import numpy as np
import SimpleITK as sitk
import monai
from copy import deepcopy
from monai.transforms import (
    Compose,
    CopyItemsd,
    DeleteItemsd,
    EnsureChannelFirstd,
    EnsureTyped,
    LoadImaged,
    Orientationd,
    RandAdjustContrastd,
    RandGaussianNoised,
    RandGaussianSmoothd,
    RandRotated,
    RandScaleIntensityd,
    RandShiftIntensityd,
    Lambda,
    RandLambda,
    LoadImage,
    MapTransform,
    Randomizable,
    Orientation
)
from functools import partial
from monai.config import KeysCollection
from collections.abc import Hashable, Sequence

from monai.apps.detection.transforms.dictionary import (
    AffineBoxToImageCoordinated,
    AffineBoxToWorldCoordinated,
    BoxToMaskd,
    ClipBoxToImaged,
    ConvertBoxToStandardModed,
    MaskToBoxd,
    RandCropBoxByPosNegLabeld,
    RandFlipBoxd,
    RandRotateBox90d,
    RandZoomBoxd,
    ConvertBoxModed,
    StandardizeEmptyBoxd,
)


def apply_attn_mask(sample):
    img_path = sample['image_path']
    img = sample['image']
    folder_path = './data/'+img_path[0].split('/')[1]
    loader = LoadImage()
    if 'chunks' in img_path[0].split('/'):
        chunk_nr = img_path[0].split('.nii')[0][-1]
        path = folder_path+'/chunks/attn_mask_resampled_chunk'+chunk_nr+'.nii.gz'
    else:
        path = folder_path+'/attn_mask_resampled.nii.gz'

    attn_mask = loader(path)
    attn_mask = Orientation(axcodes="RAS")(attn_mask[None, ...])[0]
    img_with_attn = img*attn_mask

    sample['image'] = img_with_attn
    
    return sample

def apply_attn_mask2(sample, train=False):
    img_path = sample['image_path']
    folder_path = './data/'+img_path[0].split('/')[1]
    loader = LoadImage()
    if 'chunks' in img_path[0].split('/'):
        chunk_nr = img_path[0].split('.nii')[0][-1]
        path = folder_path+'/chunks/attn_mask2_resampled_chunk'+chunk_nr+'.nii.gz'
    else:
        path = folder_path+'/attn_mask2_resampled.nii.gz'

    img_with_attn = sample['image']
    attn_mask2 = loader(path)
    attn_mask2 = Orientation(axcodes="RAS")(attn_mask2[None, ...])[0]
    
    attn_mask2 = torch.repeat_interleave(attn_mask2[None, ...], img_with_attn.shape[0], dim=0)
    img_with_attn[attn_mask2==0] = img_with_attn.min()

    sample['image'] = img_with_attn
    if train:
        sample['threshold'] = img_with_attn.min().item()
        
    return sample

class RandCropBoxByPosNegLabeld_with_threshold(Randomizable, MapTransform):

    def __init__(
        self,
        image_keys: KeysCollection,
        box_keys: str,
        label_keys: KeysCollection,
        spatial_size: Sequence[int] | int,
        num_samples: int = 1,
        whole_box: bool = True,
        pos: float = 1.0,
        neg: float = 1.0,
    ) -> None:

        self.image_keys = image_keys
        self.box_keys = box_keys
        self.label_keys = label_keys
        self.spatial_size = spatial_size
        self.num_samples = num_samples
        self.whole_box = whole_box
        self.pos = pos
        self.neg = neg

    def __call__(self, data: dict) -> list[dict[Hashable, torch.Tensor]]:
        threshold = data['threshold']
        randcrop = RandCropBoxByPosNegLabeld(
            image_keys = self.image_keys,
            box_keys = self.box_keys,
            label_keys =self.label_keys,
            spatial_size =self.spatial_size,
            whole_box = self.whole_box,
            num_samples = self.num_samples,
            pos = self.pos,
            neg = self.neg,
            thresh_image_key = self.image_keys[0],
            image_threshold = threshold
        )

        return randcrop(data)

        

def apply_attn_mask3(sample):
    img_path = sample['image_path']
    img = sample['image']
    folder_path = './data/'+img_path[0].split('/')[1]
    loader = LoadImage()
    if 'chunks' in img_path[0].split('/'):
        chunk_nr = img_path[0].split('.nii')[0][-1]
        path = folder_path+'/chunks/attn_mask3_resampled_chunk'+chunk_nr+'.nii.gz'
    else:
        path = folder_path+'/attn_mask3_resampled.nii.gz'

    attn_mask3 = loader(path)
    attn_mask3 = Orientation(axcodes="RAS")(attn_mask3[None, ...])[0]
    
    img_with_attn = img*attn_mask3

    sample['image'] = img_with_attn
    
    return sample

def apply_attn_mask4(sample):
    img_path = sample['image_path']
    folder_path = './data/'+img_path[0].split('/')[1]
    loader = LoadImage()
    if 'chunks' in img_path[0].split('/'):
        chunk_nr = img_path[0].split('.nii')[0][-1]
        path = folder_path+'/chunks/attn_mask2_resampled_chunk'+chunk_nr+'.nii.gz'
    else:
        path = folder_path+'/attn_mask2_resampled.nii.gz'

    img_with_attn = sample['image']
    attn_mask2 = loader(path)
    attn_mask2 = Orientation(axcodes="RAS")(attn_mask2[None, ...])[0]
    attn_mask2[attn_mask2==0] = .2
    
    img_with_attn = img_with_attn*attn_mask2

    sample['image'] = img_with_attn
    return sample

def get_modality(path):
    modality = path.split('/')[-1].split('_')[0]
    return modality

def copy_binary_channel(sample):
    modalities = [get_modality(x) for x in sample['image_path']]
    channels = len(fnmatch.filter(modalities, '*attn*'))
    channels += (np.array(modalities)=='skeleton').sum()
    binary_img = deepcopy(sample['image'][-channels:, ...].as_tensor())
    sample['binary_channel'] = binary_img

    return sample

def set_binary_channel(sample):
    channels = len(sample['binary_channel'])
    sample['image'][-channels:, ...] = sample['binary_channel']
    del sample['binary_channel']

    return sample


def get_new_transforms(transforms, indices, transformations, replace=False):
    if replace == None:
        replace = [False for _ in range(len(indices))]
    transforms = deepcopy(transforms)
    new_transforms = list(transforms.transforms)
    if isinstance(indices, list) or isinstance(indices, np.array):
        for index, transformation in zip(indices, transformations):
            if replace:
                new_transforms[index] = transformation
            else:
                new_transforms.insert(index, transformation)
    else:
        raise ValueError('Indices and transforms need to be put in a list.')
    transforms.transforms = tuple(new_transforms) 
    return transforms

def combine_chunks(sample, train=True):
    if train:
        meta_information = sample['image_1'].meta
    else:
        meta_information = sample['image_1_meta_dict']
        [sample.pop(f'image_{i}_meta_dict') for i in range(1, 5)]
        
    new_img_tensor = torch.concat([sample.pop(f'image_{i}') for i in range(1, 5)], axis=2)
    new_shape = new_img_tensor.shape
    meta_information['dim'][:len(new_shape)] = new_shape
    meta_information['spatial_shape'] = new_shape
    
    new_img_tensor.meta = meta_information
    sample['image'] = new_img_tensor
    
    if not train:
        sample['image_meta_dict'] = meta_information
        
    return sample


def generate_detection_train_transform(
    image_key,
    box_key,
    label_key,
    gt_box_mode,
    intensity_transform,
    patch_size,
    batch_size,
    affine_lps_to_ras=False,
    amp=True,
    with_attention_transform: str | None = None, 
    with_binary: bool = False,
):
    """
    Generate training transform for detection.

    Args:
        image_key: the key to represent images in the input json files
        box_key: the key to represent boxes in the input json files
        label_key: the key to represent box labels in the input json files
        gt_box_mode: ground truth box mode in the input json files
        intensity_transform: transform to scale image intensities,
            usually ScaleIntensityRanged for CT images, and NormalizeIntensityd for MR images.
        patch_size: cropped patch size for training
        batch_size: number of cropped patches from each image
        affine_lps_to_ras: Usually False.
            Set True only when the original images were read by itkreader with affine_lps_to_ras=True
        amp: whether to use half precision
        with_attention_transform: Choose with attention mask to apply (mask1, mask2, mask3, mask4)
        with_binary: Set to True if there is a binary channel and put it as the last channel in json file

    Return:
        training transform for detection
    """
    if amp:
        compute_dtype = torch.float16
    else:
        compute_dtype = torch.float32

    train_transforms = Compose(
        [
            CopyItemsd(keys=image_key, names='image_path'),
            LoadImaged(keys=[image_key], meta_key_postfix="meta_dict"),
            EnsureChannelFirstd(keys=[image_key]),
            EnsureTyped(keys=[image_key, box_key], dtype=torch.float32),
            EnsureTyped(keys=[label_key], dtype=torch.long),
            StandardizeEmptyBoxd(box_keys=[box_key], box_ref_image_keys=image_key),
            Orientationd(keys=[image_key], axcodes="RAS"),
            intensity_transform,
            EnsureTyped(keys=[image_key], dtype=torch.float16),
            ConvertBoxToStandardModed(box_keys=[box_key], mode=gt_box_mode),
            AffineBoxToImageCoordinated(
                box_keys=[box_key],
                box_ref_image_keys=image_key,
                image_meta_key_postfix="meta_dict",
                affine_lps_to_ras=affine_lps_to_ras,
            ),
            RandCropBoxByPosNegLabeld(
                image_keys=[image_key],
                box_keys=box_key,
                label_keys=label_key,
                spatial_size=patch_size,
                whole_box=True,
                num_samples=batch_size,
                pos=1,
                neg=1,
                allow_smaller=True,
            ),
            RandZoomBoxd(
                image_keys=[image_key],
                box_keys=[box_key],
                box_ref_image_keys=[image_key],
                prob=0.2,
                min_zoom=0.7,
                max_zoom=1.4,
                padding_mode="constant",
                keep_size=True,
            ),
            ClipBoxToImaged(
                box_keys=box_key,
                label_keys=[label_key],
                box_ref_image_keys=image_key,
                remove_empty=True,
            ),
            RandFlipBoxd(
                image_keys=[image_key],
                box_keys=[box_key],
                box_ref_image_keys=[image_key],
                prob=0.5,
                spatial_axis=0,
            ),
            RandFlipBoxd(
                image_keys=[image_key],
                box_keys=[box_key],
                box_ref_image_keys=[image_key],
                prob=0.5,
                spatial_axis=1,
            ),
            RandFlipBoxd(
                image_keys=[image_key],
                box_keys=[box_key],
                box_ref_image_keys=[image_key],
                prob=0.5,
                spatial_axis=2,
            ),
            RandRotateBox90d(
                image_keys=[image_key],
                box_keys=[box_key],
                box_ref_image_keys=[image_key],
                prob=0.75,
                max_k=3,
                spatial_axes=(0, 1),
            ),
            BoxToMaskd(
                box_keys=[box_key],
                label_keys=[label_key],
                box_mask_keys=["box_mask"],
                box_ref_image_keys=image_key,
                min_fg_label=0,
                ellipse_mask=True,
            ),
            RandRotated(
                keys=[image_key, "box_mask"],
                mode=["nearest", "nearest"],
                prob=0.2,
                range_x=np.pi / 6,
                range_y=np.pi / 6,
                range_z=np.pi / 6,
                keep_size=True,
                padding_mode="zeros",
            ),
            MaskToBoxd(
                box_keys=[box_key],
                label_keys=[label_key],
                box_mask_keys=["box_mask"],
                min_fg_label=0,
            ),
            DeleteItemsd(keys=["box_mask"]),
            RandGaussianNoised(keys=[image_key], prob=0.1, mean=0, std=0.1),
            RandGaussianSmoothd(
                keys=[image_key],
                prob=0.1,
                sigma_x=(0.5, 1.0),
                sigma_y=(0.5, 1.0),
                sigma_z=(0.5, 1.0),
            ),
            RandScaleIntensityd(keys=[image_key], prob=0.15, factors=0.25),
            RandShiftIntensityd(keys=[image_key], prob=0.15, offsets=0.1),
            RandAdjustContrastd(keys=[image_key], prob=0.3, gamma=(0.7, 1.5)),
            EnsureTyped(keys=[image_key, box_key], dtype=compute_dtype),
            EnsureTyped(keys=[label_key], dtype=torch.long),
        ]
    )
    
    if with_attention_transform in ['mask1', 'mask3']:
        apply_attn_func = {'mask1': apply_attn_mask, 'mask3': apply_attn_mask3}[with_attention_transform]
        attn_transform = Lambda(func=apply_attn_func)
        
        train_transforms = get_new_transforms(train_transforms, [8], [attn_transform])
        # new_train_transforms.insert(8, attn_transform)
    
    elif with_attention_transform == 'mask2':
        randcrop = RandCropBoxByPosNegLabeld_with_threshold(
            image_keys=[image_key],
            box_keys=box_key,
            label_keys=label_key,
            spatial_size=patch_size,
            whole_box=True,
            num_samples=batch_size,
            pos=1,
            neg=1,
        )
        
        attn_transform = Lambda(func=partial(apply_attn_mask2, train=True))
        train_transforms = get_new_transforms(train_transforms, [8, 11], [attn_transform, randcrop], replace=[False, True])
        
        # new_train_transforms[11] = RandCropBoxByPosNegLabeld_with_threshold(
        #     image_keys=[image_key],
        #     box_keys=box_key,
        #     label_keys=label_key,
        #     spatial_size=patch_size,
        #     whole_box=True,
        #     num_samples=batch_size,
        #     pos=1,
        #     neg=1,
        # )
        
        # new_train_transforms.insert(8, attn_transform)
        # new_train_transforms.insert(10, DeleteItemsd(keys='threshold'))
        
    elif with_attention_transform == 'mask4':
        attn_transform = Lambda(func=apply_attn_mask4)
        train_transforms = get_new_transforms(train_transforms, [8], [attn_transform])
        # new_train_transforms.insert(10, DeleteItemsd(keys='threshold'))
        # new_train_transforms.insert(8, attn_transform)
        
    elif ((isinstance(with_attention_transform, str) and with_attention_transform not in ['mask1', 'mask2', 'mask3', 'mask4'])) or with_attention_transform != None:
        raise ValueError('with_attention_transform needs to be in (None, mask1, mask2, mask3, mask4)')


    if with_binary and with_attention_transform in ['mask1', 'mask2', 'mask3', 'mask4']:
        # new_train_transforms.insert(7, Lambda(copy_binary_channel))
        # new_train_transforms.insert(10, Lambda(set_binary_channel))
        # new_train_transforms.insert(-8, Lambda(copy_binary_channel))
        # new_train_transforms.insert(-2, Lambda(set_binary_channel))
        train_transforms = get_new_transforms(train_transforms, [7, 10], [Lambda(copy_binary_channel), Lambda(set_binary_channel)])
    
    elif with_binary and with_attention_transform not in ['mask1', 'mask2', 'mask3', 'mask4']:
        # new_train_transforms.insert(7, Lambda(copy_binary_channel))
        # new_train_transforms.insert(9, Lambda(set_binary_channel))
        # new_train_transforms.insert(-8, Lambda(copy_binary_channel))
        # new_train_transforms.insert(-2, Lambda(set_binary_channel))
        train_transforms = get_new_transforms(train_transforms, [7, 9], [Lambda(copy_binary_channel), Lambda(set_binary_channel)])
  
    return train_transforms


def generate_detection_val_transform(
    image_key,
    box_key,
    label_key,
    gt_box_mode,
    intensity_transform,
    affine_lps_to_ras=True,
    amp=True,
    with_attention_transform: str | None = None,
    with_binary: bool = False,
):
    """
    Generate validation transform for detection.

    Args:
        image_key: the key to represent images in the input json files
        box_key: the key to represent boxes in the input json files
        label_key: the key to represent box labels in the input json files
        gt_box_mode: ground truth box mode in the input json files
        intensity_transform: transform to scale image intensities,
            usually ScaleIntensityRanged for CT images, and NormalizeIntensityd for MR images.
        affine_lps_to_ras: Usually False.
            Set True only when the original images were read by itkreader with affine_lps_to_ras=True
        amp: whether to use half precision

    Return:
        validation transform for detection
    """
    if amp:
        compute_dtype = torch.float16
    else:
        compute_dtype = torch.float32

    val_transforms = Compose(
        [
            CopyItemsd(keys=image_key, names='image_path'),
            LoadImaged(keys=[image_key], image_only=False, meta_key_postfix="meta_dict"),
            EnsureChannelFirstd(keys=[image_key]),
            EnsureTyped(keys=[image_key, box_key], dtype=torch.float32),
            EnsureTyped(keys=[label_key], dtype=torch.long),
            StandardizeEmptyBoxd(box_keys=[box_key], box_ref_image_keys=image_key),
            Orientationd(keys=[image_key], axcodes="RAS"),
            intensity_transform,
            ConvertBoxToStandardModed(box_keys=[box_key], mode=gt_box_mode),
            AffineBoxToImageCoordinated(
                box_keys=[box_key],
                box_ref_image_keys=image_key,
                image_meta_key_postfix="meta_dict",
                affine_lps_to_ras=affine_lps_to_ras,
            ),
            EnsureTyped(keys=[image_key, box_key], dtype=compute_dtype),
            EnsureTyped(keys=label_key, dtype=torch.long),
        ]
    )
    if with_attention_transform in ['mask1', 'mask2', 'mask3', 'mask4']:
        apply_attn_func = {'mask1': apply_attn_mask, 'mask2':apply_attn_mask2, 'mask3': apply_attn_mask3, 'mask4': apply_attn_mask4}[with_attention_transform]
        attn_transform = Lambda(func=apply_attn_func) 

        val_transforms = get_new_transforms(val_transforms, [8], [attn_transform])
        # new_val_transforms.insert(8, attn_transform)
        # new_val_transforms.insert(9, DeleteItemsd(keys='image_path'))

    elif ((isinstance(with_attention_transform, str) and with_attention_transform not in ['mask1', 'mask2', 'mask3', 'mask4'])) or with_attention_transform != None:
        raise ValueError('with_attention_transform needs to be in (None, mask1, mask2, mask3, mask4)')

    if with_binary and isinstance(with_attention_transform, str):
        # new_val_transforms.insert(7, Lambda(copy_binary_channel))
        # new_val_transforms.insert(10, Lambda(set_binary_channel))
        val_transforms = get_new_transforms(val_transforms, [7, 10], [Lambda(copy_binary_channel), Lambda(set_binary_channel)])
    
    elif with_binary and not isinstance(with_attention_transform, str):
        # new_val_transforms.insert(7, Lambda(copy_binary_channel))
        # new_val_transforms.insert(9, Lambda(set_binary_channel))
        val_transforms = get_new_transforms(val_transforms, [7, 9], [Lambda(copy_binary_channel), Lambda(set_binary_channel)])
        
    return val_transforms


def generate_detection_inference_transform(
    image_key,
    pred_box_key,
    pred_label_key,
    pred_score_key,
    gt_box_mode,
    intensity_transform,
    affine_lps_to_ras=False,
    amp=True,
    with_attention_transform: str | None = None
):
    """
    Generate validation transform for detection.

    Args:
        image_key: the key to represent images in the input json files
        pred_box_key: the key to represent predicted boxes
        pred_label_key: the key to represent predicted box labels
        pred_score_key: the key to represent predicted classification scores
        gt_box_mode: ground truth box mode in the input json files
        intensity_transform: transform to scale image intensities,
            usually ScaleIntensityRanged for CT images, and NormalizeIntensityd for MR images.
        affine_lps_to_ras: Usually False.
            Set True only when the original images were read by itkreader with affine_lps_to_ras=True
        amp: whether to use half precision

    Return:
        validation transform for detection
    """
    if amp:
        compute_dtype = torch.float16
    else:
        compute_dtype = torch.float32

    test_transforms = Compose(
        [
            LoadImaged(keys=[image_key], image_only=False, meta_key_postfix="meta_dict"),
            EnsureChannelFirstd(keys=[image_key]),
            EnsureTyped(keys=[image_key], dtype=torch.float32),
            Orientationd(keys=[image_key], axcodes="RAS"),
            intensity_transform,
            EnsureTyped(keys=[image_key], dtype=compute_dtype),
        ]
    )
    post_transforms = Compose(
        [
            ClipBoxToImaged(
                box_keys=[pred_box_key],
                label_keys=[pred_label_key, pred_score_key],
                box_ref_image_keys=image_key,
                remove_empty=True,
            ),
            AffineBoxToWorldCoordinated(
                box_keys=[pred_box_key],
                box_ref_image_keys=image_key,
                image_meta_key_postfix="meta_dict",
                affine_lps_to_ras=affine_lps_to_ras,
            ),
            ConvertBoxModed(box_keys=[pred_box_key], src_mode="xyzxyz", dst_mode=gt_box_mode),
            DeleteItemsd(keys=[image_key]),
        ]
    )
    
    if with_attention_transform in ['mask1', 'mask3']:
        new_test_transforms = list(test_transforms.transforms)
        apply_attn_func = {'mask1': apply_attn_mask, 'mask3': apply_attn_mask3}[with_attention_transform]
        attn_transform = Lambda(func=apply_attn_func) 
        
        new_test_transforms.insert(0, CopyItemsd(keys=image_key, names='image_path'))
        new_test_transforms.insert(2, attn_transform)
        new_test_transforms.insert(3, DeleteItemsd(keys='image_path'))
        test_transforms.transforms = tuple(new_test_transforms)
        
    elif with_attention_transform=='mask2':
        new_test_transforms = list(test_transforms.transforms)
        attn_transform = Lambda(func=apply_attn_mask2)
        
        new_test_transforms.insert(0, CopyItemsd(keys=image_key, names='image_path'))
        new_test_transforms.insert(8, attn_transform)
        new_test_transforms.insert(9, DeleteItemsd(keys='image_path'))
        test_transforms.transforms = tuple(new_test_transforms)

    elif with_attention_transform == 'mask4':
        new_test_transforms = list(test_transforms.transforms)
        attn_transform = Lambda(func=apply_attn_mask4)
        
        new_test_transforms.insert(0, CopyItemsd(keys=image_key, names='image_path'))
        new_test_transforms.insert(8, attn_transform)
        new_test_transforms.insert(9, DeleteItemsd(keys='image_path'))
        # new_train_transforms.insert(10, DeleteItemsd(keys='threshold'))
        test_transforms.transforms = tuple(new_test_transforms)

    return test_transforms, post_transforms

