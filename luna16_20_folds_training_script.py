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

import pandas as pd
import numpy as np
import torch
import json
from script_utils import (
    load_data,
    load_data2,
    prepare_json_files,
    prepare_json_files_luna,
    load_luna16
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

def train(
    train_loader,
    val_loader,
    num_epochs=300, 
    batch_size=4, 
    patch_size=[192, 192, 80], 
    cls_loss=torch.nn.BCEWithLogitsLoss(reduction="mean"),
    optim='default',
    learning_rate=1e-2,
    optim_param=dict([]),
    scheduler=None, 
    pretrained_backbone=False,
    pretrained_model=None, 
    val_interval=2, 
    detections_per_img=150,
    run_id=None,
    name=None
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

    args.patch_size=patch_size
    args.batch_size=batch_size
    args.lr=learning_rate

    config_run=dict([])
    for k in config_dict.keys():
        config_run[k] = args.__dict__[k]

    
    # 3. build model
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
        pretrained_backbone=pretrained_backbone,
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
    detector.set_cls_loss(cls_loss)
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
        
    if pretrained_model != None:
        detector.network.load_state_dict(pretrained_model.state_dict())

    # 4. Initialize training
    # initlize optimizer
    max_epochs = num_epochs
    if optim=='default':
        optimizer = torch.optim.SGD(
            detector.network.parameters(),
            args.lr,
            momentum=0.9,
            weight_decay=3e-5,
            nesterov=True,
        )
    elif optim=='SGD':
        optimizer = torch.optim.SGD(
            detector.network.parameters(),
            args.lr,
            **optim_param
        )
    elif optim=='Adam':
        optimizer = torch.optim.Adam(
            detector.network.parameters(),
            args.lr,
            **optim_param
        )

    if scheduler=='cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)

    scaler = torch.cuda.amp.GradScaler() if amp else None
    optimizer.zero_grad()
    optimizer.step()

    # 5. train
    coco_metric = COCOMetric(classes=["lesion"], iou_list=[0.1, 0.5, 0.75, 0.9], max_detection=[detections_per_img]) 
    best_val_epoch_metric = 0.0
    best_val_epoch = -1  # the epoch that gives best validation metrics

    w_cls = 1.0          #config_dict.get("w_cls", 1.0)  # weight between classification loss and box regression loss, default 1.0

    
    #init wandb
    # start a new wandb run to track this script
    if run_id == None:
        run_id = wandb.util.generate_id()
        wandb.init(
            #set id in case you want to resume
            id = run_id,
            resume="allow",
            # set the wandb project where this run will be logged
            project="monai_detection-mbd",
            
            # track hyperparameters and run metadata
            config={
                "run_id": run_id,
                "pretrained_backbone": pretrained_backbone,
                "optimizer": optim,
                "optim_param": optim_param,
                "architecture": "RetinaNet (MONAI)",
                "dataset": "luna16",
                "epochs": max_epochs,
                "detections_per_img": detections_per_img,
                "config": config_run,
                "train_val": train_val
            }
        )
    else:
        wandb.init(id=run_id, resume=True, project="monai_detection-mbd")

    if isinstance(name, str):
        wandb.run.name = name

    save_dict = dict([])

    run_name = wandb.run.name
    wandb.config.update({"run_name": run_name})
    CHECKPOINT_PATH = f"./checkpoint_{run_name}.tar"
    os.makedirs(f'./trained_models/{run_name}/', exist_ok=True)
    model_path = args.model_path.split('/')
    model_path.insert(1, run_name)
    model_path = "/".join(model_path)

    if wandb.run.resumed:
        checkpoint = torch.load(CHECKPOINT_PATH)
        detector.network.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start = checkpoint["epoch"]
    else:
        start = 0
    
    for epoch in range(start, max_epochs):
        # ------------- Training -------------
        print("-" * 10)
        print(f"epoch {epoch + 1}/{max_epochs}")
        detector.train()
        epoch_loss = 0
        epoch_cls_loss = 0
        epoch_box_reg_loss = 0
        step = 0
        train_outputs_all = []
        train_targets_all = []
        start_time = time.time()
        
        # Training
        for batch_data in train_loader:
            step += 1
            inputs = [
                batch_data_ii["image"].to(device) for batch_data_i in batch_data for batch_data_ii in batch_data_i
            ]
            targets = [
                dict(
                    label=batch_data_ii["label"].to(device),
                    box=batch_data_ii["box"].to(device),
                )
                for batch_data_i in batch_data
                for batch_data_ii in batch_data_i
            ]

            for param in detector.network.parameters():
                param.grad = None

            if amp and (scaler is not None):
                with torch.cuda.amp.autocast():
                    outputs = detector(inputs, targets)
                    loss = w_cls * outputs[detector.cls_key] + outputs[detector.box_reg_key]
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = detector(inputs, targets)
                loss = w_cls * outputs[detector.cls_key] + outputs[detector.box_reg_key]
                loss.backward()
                optimizer.step()

            # save to wandb 
            epoch_loss += loss.detach().item()
            epoch_cls_loss += outputs[detector.cls_key].detach().item()
            epoch_box_reg_loss += outputs[detector.box_reg_key].detach().item()
            print(f"{step}/{epoch_len}, train_loss: {loss.item():.4f}")
            
            save_dict.update({"train_loss": loss.detach().item(), "epoch": epoch+1})
            wandb.log({"train_loss": loss.detach().item(),
                       "epoch": epoch+1
                      })

        
        if scheduler != None:
            scheduler.step()

        end_time = time.time()
        print(f"Training time: {end_time-start_time}s")
        del inputs, batch_data
        torch.cuda.empty_cache()
        gc.collect()

        # save to wandb
        epoch_loss /= step
        epoch_cls_loss /= step
        epoch_box_reg_loss /= step
        print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")
        
        save_dict.update({"avg_train_loss": epoch_loss, 
                          "avg_train_cls_loss": epoch_cls_loss,
                          "avg_train_box_reg_loss": epoch_box_reg_loss,
                          "train_lr": optimizer.param_groups[0]["lr"],
                         })
        wandb.log({"avg_train_loss": epoch_loss, 
                   "avg_train_cls_loss": epoch_cls_loss,
                   "avg_train_box_reg_loss": epoch_box_reg_loss,
                   "train_lr": optimizer.param_groups[0]["lr"],
                   "epoch": epoch+1
                  })

        # ------------- Validation for model selection -------------
        if (epoch + 1) % val_interval == 0:
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
                'iou_thresholds': coco_metric.iou_thresholds,
                'epoch': epoch+1
            }
            
            
            val_keys = [
                f'mAP_IoU_0.10_0.50_0.05_MaxDet_{detections_per_img}',
                f'AP_IoU_0.10_MaxDet_{detections_per_img}',
                f'mAR_IoU_0.10_0.50_0.05_MaxDet_{detections_per_img}',
                f'AR_IoU_0.10_MaxDet_{detections_per_img}',
            ]
            val_epoch_metric_dict = coco_metric(results_metric)
            val_epoch_metric_dict[0]['epoch'] = epoch+1
            ap_iou_10 = val_epoch_metric_dict[0][f'AP_IoU_0.10_MaxDet_{detections_per_img}']
            ap_iou_50 = val_epoch_metric_dict[0][f'AP_IoU_0.50_MaxDet_{detections_per_img}']
            map_iou = val_epoch_metric_dict[0][f'mAP_IoU_0.10_0.50_0.05_MaxDet_{detections_per_img}']
            
            ar_iou_10 = val_epoch_metric_dict[0][f'AR_IoU_0.10_MaxDet_{detections_per_img}']
            ar_iou_50 = val_epoch_metric_dict[0][f'AR_IoU_0.50_MaxDet_{detections_per_img}']
            mar_iou = val_epoch_metric_dict[0][f'mAR_IoU_0.10_0.50_0.05_MaxDet_{detections_per_img}']
            print(val_epoch_metric_dict[0])

            val_epoch_metric = [val_epoch_metric_dict[0][key] for key in val_keys]
            val_epoch_metric = sum(val_epoch_metric) / len(val_epoch_metric)

            wandb.log({"val_metric": val_epoch_metric,
                       f'AP_IoU_0.10_MaxDet_{detections_per_img}': ap_iou_10,
                       f'AP_IoU_0.50_MaxDet_{detections_per_img}': ap_iou_50,
                       f'mAP_IoU_0.10_0.50_0.05_MaxDet_{detections_per_img}': map_iou,
                       f'AR_IoU_0.10_MaxDet_{detections_per_img}': ar_iou_10,
                       f'AR_IoU_0.50_MaxDet_{detections_per_img}': ar_iou_50,
                       f'mAR_IoU_0.10_0.50_0.05_MaxDet_{detections_per_img}': mar_iou,
                       "epoch": epoch+1
                      })
            
            with open(f"{args.model_path.split('/')[0]}/{run_name}/coco_metric_{run_name}_last.json", "w") as f:
                json.dump(val_epoch_metric_dict, f)
            
            results_path = f"{args.model_path.split('/')[0]}/{run_name}/results_metric_{run_name}_last.pkl"
            pd.to_pickle(results_metric_dict, results_path)
            
            # save best trained model
            if val_epoch_metric > best_val_epoch_metric:
                best_val_epoch_metric = val_epoch_metric
                best_val_epoch = epoch + 1
                torch.jit.save(detector.network, f"{model_path.split('.')[0]}_{run_name}_val.pt")
                
                with open(f"{args.model_path.split('/')[0]}/{run_name}/coco_metric_{run_name}_val.json", "w") as f:
                    json.dump(val_epoch_metric_dict, f)
                    
                results_path = f"{args.model_path.split('/')[0]}/{run_name}/results_metric_{run_name}_val.pkl"
                pd.to_pickle(results_metric_dict, results_path)
                    
                    
                print("saved new best metric model")
    
            
            print(
                "current epoch: {} current metric: {:.4f} "
                "best metric: {:.4f} at epoch {}".format(
                    epoch + 1, val_epoch_metric, best_val_epoch_metric, best_val_epoch
                )
            )

        # save last trained model
        save_dict.update({"model_state_dict": detector.network.state_dict(),
                          "optimizer_state_dict": optimizer.state_dict()
                         })
        torch.jit.save(detector.network, f"{model_path.split('.')[0]}_{run_name}_last.pt")
        print("saved last model")

        # create checkpoint
        torch.save(
            save_dict, 
            CHECKPOINT_PATH,
        )
          
    
    print(f"train completed, best_metric: {best_val_epoch_metric:.4f} " f"at epoch: {best_val_epoch}")

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
        "-f",
        "--fold",
        default='fold0',
        help="Which fold to train and validate.",
    )

    args = parser.parse_args()
    
    # Settings for luna training run
    cv_split = 5
    cache_rate = 1.0
    
    env_dict, config_dict, train_val_fold = prepare_json_files_luna(args, cv_split=cv_split)
    
    for k, v in env_dict.items():
        setattr(args, k, v)
    for k, v in config_dict.items():
        setattr(args, k, v)
        
    setattr(args, 'validation_key', 'validation')
    setattr(args, 'training_key', 'training')

    fold = args.fold

    train_val = train_val_fold[fold]
    fold_name = fold+'_'+'LUNA16_20'

    with open('train_val_luna_fold.json', 'w') as f:
        json.dump(train_val, f)

    train_loader, val_loader, epoch_len = load_luna16(args, cache_rate=cache_rate)
    

    
    train(
        train_loader = train_loader,
        val_loader = val_loader,
        num_epochs = 1000,
        batch_size = 4,
        patch_size = [192, 192, 80],
        optim = 'default',
        scheduler='cosine',
        learning_rate = 1e-3,
        pretrained_backbone = False,
        pretrained_model = None, 
        val_interval = 20, 
        detections_per_img = 150,
        run_id = None,
        name = fold_name
    )
        

if __name__ == "__main__":
    logging.basicConfig(
        stream=sys.stdout,
        level=logging.INFO,
        format="[%(asctime)s.%(msecs)03d][%(levelname)5s](%(name)s) - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    main()