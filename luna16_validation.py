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

import argparse
import gc
import json
import logging
import sys
import time
import os
import glob

import pandas as pd
import numpy as np
import torch
import json
from script_utils import (
    load_data,
    load_data2,
    prepare_json_files,
    prepare_json_files_luna,
    load_luna16,
    load_luna16_val
)
from generate_transforms import (
    generate_detection_train_transform,
    generate_detection_val_transform,
)

from warmup_scheduler import GradualWarmupScheduler

import monai
from monai.apps.detection.metrics.coco import COCOMetric
from monai.apps.detection.metrics.matching import matching_batch
from monai.apps.detection.networks.retinanet_detector import RetinaNetDetector
from monai.apps.detection.networks.retinanet_network import (
    RetinaNet,
    resnet_fpn_feature_extractor,
)
from monai.apps.detection.utils.anchor_utils import AnchorGeneratorWithAnchorShape
from monai.data import DataLoader, ThreadDataLoader, CacheDataset, Dataset, box_utils, load_decathlon_datalist
from monai.data.utils import no_collation
from monai.networks.nets import resnet
from monai.transforms import ScaleIntensityRanged, NormalizeIntensityd
from monai.utils import set_determinism

#using wandb instead of tensorboard
import wandb
print('everything is imported')
wandb.login(key='KEY')

def validate(
    val_ds,
    val_loader,
    pretrained_model, 
    name,
    group,
    model_folder,
    run_id=None,
    detections_per_img=150,
    set='test'
):


    set_determinism(seed=0)

    amp = True
    if amp:
        compute_dtype = torch.float16
    else:
        compute_dtype = torch.float32

    monai.config.print_config()
    torch.backends.cudnn.benchmark = True
    torch.set_num_threads(4)

    config_run=dict([])
    for k in config_dict.keys():
        config_run[k] = args.__dict__[k]

    # 1. build model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    # 1) build anchor generator
    # returned_layers: when target boxes are small, set it smaller
    # base_anchor_shapes: anchor shape for the most high-resolution output,
    #   when target boxes are small, set it smaller
    anchor_generator = AnchorGeneratorWithAnchorShape(
        feature_map_scales=[2**l for l in range(len(args.returned_layers) + 1)],
        base_anchor_shapes=args.base_anchor_shapes,
    )

    # 2) build network
    conv1_t_size = [max(7, 2 * s + 1) for s in args.conv1_t_stride]
    backbone = resnet.ResNet(
        block=resnet.ResNetBottleneck,
        layers=[3, 4, 6, 3],
        block_inplanes=resnet.get_inplanes(),
        n_input_channels=args.n_input_channels,
        conv1_t_stride=args.conv1_t_stride,
        conv1_t_size=conv1_t_size,
    )
    feature_extractor = resnet_fpn_feature_extractor(
        backbone=backbone,
        spatial_dims=args.spatial_dims,
        pretrained_backbone=False,
        trainable_backbone_layers=None,
        returned_layers=args.returned_layers,
    )
    num_anchors = anchor_generator.num_anchors_per_location()[0]
    size_divisible = [s * 2 * 2 ** max(args.returned_layers) for s in feature_extractor.body.conv1.stride]
    net = torch.jit.script(
        RetinaNet(
            spatial_dims=args.spatial_dims,
            num_classes=len(args.fg_labels),
            num_anchors=num_anchors,
            feature_extractor=feature_extractor,
            size_divisible=size_divisible,
        )
    )
    # 3) build detector
    detector = RetinaNetDetector(network=net, anchor_generator=anchor_generator, debug=args.verbose).to(device)

    # set training components
    detector.set_cls_loss(torch.nn.BCEWithLogitsLoss(reduction="mean"))
    detector.set_atss_matcher(num_candidates=4, center_in_gt=False)
    detector.set_hard_negative_sampler(
        batch_size_per_image=64,
        positive_fraction=args.balanced_sampler_pos_fraction,
        pool_size=20,
        min_neg=16,
    )
    detector.set_target_keys(box_key="box", label_key="label")

    # set validation components
    detector.set_box_selector_parameters(
        score_thresh=args.score_thresh,
        topk_candidates_per_level=1000,
        nms_thresh=args.nms_thresh,
        detections_per_img=detections_per_img, 
    )
    detector.set_sliding_window_inferer(
        roi_size=args.val_patch_size,
        overlap=0.25,
        sw_batch_size=1,
        mode="constant",
        device="cpu",
    )

    detector.network.load_state_dict(pretrained_model)

    # 2. train
    coco_metric = COCOMetric(classes=["lesion"], iou_list=[0.1, 0.5, 0.75, 0.9], max_detection=[detections_per_img]) 
    w_cls = 1.0          #config_dict.get("w_cls", 1.0)  # weight between classification loss and box regression loss, default 1.0

    
    #init wandb
    # start a new wandb run to track this script
    if run_id == None:
        run_id = wandb.util.generate_id()
        wandb.init(
            #set_group
            group=group,
            #set id in case you want to resume
            id = run_id,
            resume="allow",
            # set the wandb project where this run will be logged
            project="monai_detection-mbd-validations",
            
            # track hyperparameters and run metadata
            config={
                "run_id": run_id,
                "architecture": "RetinaNet (MONAI)",
                "dataset": "mbd",
                "detections_per_img": detections_per_img,
                "config": config_run,
                "train_val": train_val
            }
        )
        if isinstance(name, str):
            wandb.run.name=name
    else:
        wandb.init(id=run_id, resume=True, project="monai_detection-mbd")

    run_name = wandb.run.name
    run_id = wandb.run.id
    wandb.config.update({"run_name": run_name})

    # ------------- Validation for model selection -------------
    detector.eval()
    val_outputs_all = []
    val_targets_all = []
    start_time = time.time()
    with torch.no_grad():
        for val_data in val_loader:
            # if all val_data_i["image"] smaller than args.val_patch_size, no need to use inferer
            # otherwise, need inferer to handle large input images.
            use_inferer = not all(
                [val_data_i["image"][0, ...].numel() < np.prod(args.val_patch_size) for val_data_i in val_data]
            )
            val_inputs = [val_data_i.pop("image").to(device) for val_data_i in val_data]

            if amp:
                with torch.cuda.amp.autocast():
                    val_outputs = detector(val_inputs, use_inferer=use_inferer)
            else:
                val_outputs = detector(val_inputs, use_inferer=use_inferer)

            # save outputs for evaluation
            val_outputs_all += val_outputs
            val_targets_all += val_data

    end_time = time.time()
    print(f"Validation time: {end_time-start_time}s")

    # compute metrics
    del val_inputs
    torch.cuda.empty_cache()

    if args.parts:
        for i, val_data in enumerate(train_val[args.validation_key]):
            val_targets_all[i]['image'] = val_data['image']
            del val_targets_all[i]['image_meta_dict']
            for k, v in val_targets_all[i].items():
                if k=='image' or k=='image_path' or k=='id':
                    continue
                else:
                    val_targets_all[i][k] = list([x.tolist() for x in v]) 
                    
        # assemble parts for validation per patient
        preds, targets = assemble_parts(val_outputs_all, val_targets_all)
    
        
        results_metric = matching_batch(
            iou_fn=box_utils.box_iou,
            iou_thresholds=coco_metric.iou_thresholds,
            pred_boxes=[x['box'].numpy() for x in preds],
            pred_classes=[x['label'].numpy() for x in preds],
            pred_scores=[x['label_scores'].numpy() for x in preds],
            gt_boxes=[np.array(x['box']) for x in targets],
            gt_classes=[np.array(x['label']) for x in targets],
            max_detections=detections_per_img
        )
    
        results_metric_dict = {
            'results': results_metric,
            'preds': preds,
            'gt': targets,
            'iou_thresholds': coco_metric.iou_thresholds
        }
    else:
        results_metric = matching_batch(
            iou_fn=box_utils.box_iou,
            iou_thresholds=coco_metric.iou_thresholds,
            pred_boxes=[
                val_data_i[detector.target_box_key].cpu().detach().numpy() for val_data_i in val_outputs_all
            ],
            pred_classes=[
                val_data_i[detector.target_label_key].cpu().detach().numpy() for val_data_i in val_outputs_all
            ],
            pred_scores=[
                val_data_i[detector.pred_score_key].cpu().detach().numpy() for val_data_i in val_outputs_all
            ],
            gt_boxes=[val_data_i[detector.target_box_key].cpu().detach().numpy() for val_data_i in val_targets_all],
            gt_classes=[
                val_data_i[detector.target_label_key].cpu().detach().numpy() for val_data_i in val_targets_all
            ],
            max_detections=detections_per_img
        )

        for i, val_data in enumerate(train_val[args.validation_key]):
            val_targets_all[i]['image'] = val_data['image']
            del val_targets_all[i]['image_meta_dict']
            for k, v in val_targets_all[i].items():
                if k=='image' or k=='image_path' or k=='id':
                    continue
                else:
                    val_targets_all[i][k] = list([x.tolist() for x in v]) 

        results_metric_dict = {
            'results': results_metric,
            'preds': val_outputs_all,
            'gt': val_targets_all,
            'iou_thresholds': coco_metric.iou_thresholds
        }

    # log values
    val_keys = [
        f'mAP_IoU_0.10_0.50_0.05_MaxDet_{detections_per_img}',
        f'AP_IoU_0.10_MaxDet_{detections_per_img}',
        f'mAR_IoU_0.10_0.50_0.05_MaxDet_{detections_per_img}',
        f'AR_IoU_0.10_MaxDet_{detections_per_img}',
    ]
    
    val_epoch_metric_dict = coco_metric(results_metric)
    ap_iou_10 = val_epoch_metric_dict[0][f'AP_IoU_0.10_MaxDet_{detections_per_img}']
    ap_iou_50 = val_epoch_metric_dict[0][f'AP_IoU_0.50_MaxDet_{detections_per_img}']
    map_iou = val_epoch_metric_dict[0][f'mAP_IoU_0.10_0.50_0.05_MaxDet_{detections_per_img}']
    
    ar_iou_10 = val_epoch_metric_dict[0][f'AR_IoU_0.10_MaxDet_{detections_per_img}']
    ar_iou_50 = val_epoch_metric_dict[0][f'AR_IoU_0.50_MaxDet_{detections_per_img}']
    mar_iou = val_epoch_metric_dict[0][f'mAR_IoU_0.10_0.50_0.05_MaxDet_{detections_per_img}']
    print(val_epoch_metric_dict[0])

    val_epoch_metric = [val_epoch_metric_dict[0][key] for key in val_keys]
    val_epoch_metric = sum(val_epoch_metric) / len(val_epoch_metric)

    wandb.log({f"({set}) val_metric": val_epoch_metric,
               f'({set}) AP_IoU_0.10_MaxDet_{detections_per_img}': ap_iou_10,
               f'({set}) AP_IoU_0.50_MaxDet_{detections_per_img}': ap_iou_50,
               f'({set}) mAP_IoU_0.10_0.50_0.05_MaxDet_{detections_per_img}': map_iou,
               f'({set}) AR_IoU_0.10_MaxDet_{detections_per_img}': ar_iou_10,
               f'({set}) AR_IoU_0.50_MaxDet_{detections_per_img}': ar_iou_50,
               f'({set}) mAR_IoU_0.10_0.50_0.05_MaxDet_{detections_per_img}': mar_iou
              })
    
    with open(f"{model_folder}/{set}_coco_metric_{run_name}_{run_id}_last.json", "w") as f:
        json.dump(val_epoch_metric_dict, f)

    results_path = f"{model_folder}/{set}_results_metric_{run_name}_{run_id}_last.pkl"
    pd.to_pickle(results_metric_dict, results_path)


    torch.cuda.empty_cache()
    gc.collect()

    wandb.finish()
    return

def main():
    # Define global variables
    global args, env_dict, config_dict, train_val, train_loader, val_loader, epoch_len
    # Define args
    parser = argparse.ArgumentParser(description="PyTorch Object Detection Training")
    parser.add_argument(
        "-e",
        "--environment-file",
        default="./config/environment_luna16_fold_mbd.json",
        help="environment json file that stores environment path",
    )
    parser.add_argument(
        "-c",
        "--config-file",
        default="./config/config_train_luna16_16g.json",
        help="config json file that stores hyper-parameters",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        default=False,
        action="store_true",
        help="whether to print verbose detail during training, recommand True when you are not sure about hyper-parameters",
    )
    parser.add_argument(
        "-s",
        "--set",
        default="test",
        help="Which set to validate ons.",
    )
    parser.add_argument(
        "-f",
        "--fold",
        default='fold0',
        help="Which fold to train and validate.",
    )
    parser.add_argument(
        "--parts",
        default=False,
        action='store_true',
        help="Use ROI-based model.",
    )

    args = parser.parse_args()

    model_path=glob.glob(f"./cv_models/{args.fold}_LUNA16_*/*_last.pt")[0]
    validation_key=args.set
    setattr(args, 'validation_key', args.set)
        
    model_folder="/".join(model_path.split('/')[:-1])
    
    try:
        pretrained_model=torch.jit.load(model_path).state_dict()
        print('jit load worked')
    except Exception as e:
        print('jit load did not work')
        print(e)
        try:
            pretrained_model=torch.load(model_path).state_dict()
            print('normal load worked')
        except Exception as e:
            print('normal load did not work')
            print(e)
            
    detections_per_img = 150
    
    # Settings for luna training run
    cv_split = 5
    cache_rate = 1.0
    
    env_dict, config_dict, train_val_fold = prepare_json_files_luna(args, cv_split=cv_split)
    
    for k, v in env_dict.items():
        setattr(args, k, v)
    for k, v in config_dict.items():
        setattr(args, k, v)

    fold = args.fold

    train_val = json.load(open('./LUNA16_datasplit/dataset_fold_mbd.json', "r"))
    run_name = 'val'+'_'+'LUNA16_20'+'_'+fold

    with open('train_val_luna_fold.json', 'w') as f:
        json.dump(train_val_fold[fold], f)

    val_loader, val_ds = load_luna16_val(args, cache_rate=cache_rate)
    

    
    validate(
        val_ds=val_ds,
        val_loader=val_loader,
        pretrained_model=pretrained_model, 
        name=run_name,
        group=fold,
        model_folder=model_folder,
        set=args.set
    )
        

if __name__ == "__main__":
    logging.basicConfig(
        stream=sys.stdout,
        level=logging.INFO,
        format="[%(asctime)s.%(msecs)03d][%(levelname)5s](%(name)s) - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    main()