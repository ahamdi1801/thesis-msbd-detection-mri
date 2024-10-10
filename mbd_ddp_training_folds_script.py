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
import html
from script_utils import (
    load_data,
    load_train_data,
    load_data2,
    prepare_json_files,
    prepare_json_files_parts,
    assemble_parts
)
from script_utils_ddp import (
    load_data_ddp,
    ddp_setup,
    cleanup,
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

import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
#using wandb instead of tensorboard
import wandb

def train(
    rank,
    world_size,
    train_ds=None,
    val_ds=None,
    train_loader=None,
    val_loader=None,
    num_epochs=10, 
    batch_size=4, 
    patch_size=[80, 140, 40], 
    cls_loss=torch.nn.BCEWithLogitsLoss(reduction="mean"),
    optim='default',
    learning_rate=1e-2,
    optim_param=dict([]),
    scheduler=None, 
    pretrained_backbone=False,
    pretrained_model=None, 
    val_interval=20,
    detections_per_img=150,
    run_id=None,
    resume=False,
    checkpoint_path=None,
    name=None,
    group=None
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

    # 1. and 2. load data
    train_loader, val_loader, epoch_len, train_ds, val_ds = load_data_ddp(args, rank, world_size, training_key=args.training_key, validation_key=args.validation_key)
        
    epoch_len = len(train_ds) // train_loader.batch_size
        
    # 3. build model
        
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
    detector = RetinaNetDetector(network=net, anchor_generator=anchor_generator, debug=args.verbose).to(rank)

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

    detector = DDP(detector, device_ids=[rank], find_unused_parameters=True)

    # 4. Initialize training
    # initlize optimizer
    max_epochs = num_epochs
    if optim=='default':
        optimizer = torch.optim.SGD(
            detector.parameters(),
            args.lr,
            momentum=0.9,
            weight_decay=3e-5,
            nesterov=True,
        )
    elif optim=='SGD':
        optimizer = torch.optim.SGD(
            detector.parameters(),
            args.lr,
            **optim_param
        )
    elif optim=='Adam':
        optimizer = torch.optim.Adam(
            detector.parameters(),
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
    if run_id == None and rank == 0:
        run_id = wandb.util.generate_id()
        wandb.init(
            #set_group
            group=group,
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
                "dataset": "mbd",
                "epochs": max_epochs,
                "detections_per_img": detections_per_img,
                "config": config_run,
                "train_val": train_val,
                "test": args.test,
                "test_chunks": args.test_chunks
            }
        )
        if isinstance(name, str):
            wandb.run.name=name
    elif rank == 0:
        wandb.init(id=run_id, resume=True, project="monai_detection-mbd")

    save_dict = dict([])

    if rank == 0:
        run_name = wandb.run.name
        run_id = wandb.run.id
        wandb.config.update({"run_name": run_name})
        CHECKPOINT_PATH = f"./checkpoint_{run_name}_{run_id}.tar"
        os.makedirs(f'./trained_models/{run_name}_{run_id}/', exist_ok=True)
        model_path = args.model_path.split('/')
        model_path.insert(1, f'{run_name}_{run_id}')
        model_path = "/".join(model_path)

    if resume and checkpoint_path != None:
        checkpoint = torch.load(checkpoint_path)
        detector.network.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start = checkpoint["epoch"]
    else:
        start = 0
    
    for epoch in range(start, max_epochs):
        # train_loader.sampler.set_epoch(epoch)
        # val_loader.sampler.set_epoch(epoch)
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
        
        train_loss = []
        # scheduler_warmup.step()
        # Training
        for batch_data in train_loader:
            optimizer.zero_grad()
            step += 1
            inputs = [
                batch_data_ii["image"].to(rank) for batch_data_i in batch_data for batch_data_ii in batch_data_i
            ]
            targets = [
                dict(
                    label=batch_data_ii["label"].to(rank),
                    box=batch_data_ii["box"].to(rank),
                )
                for batch_data_i in batch_data
                for batch_data_ii in batch_data_i
            ]

            # for param in detector.network.parameters():
            #     param.grad = None

            if amp and (scaler is not None):
                with torch.cuda.amp.autocast():
                    outputs = detector(inputs, targets)
                    loss = w_cls * outputs[detector.module.cls_key] + outputs[detector.module.box_reg_key]
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = detector(inputs, targets)
                loss = w_cls * outputs[detector.module.cls_key] + outputs[detector.module.box_reg_key]
                loss.backward()
                optimizer.step()  

            # save to wandb 
            train_loss.append(loss.detach().item())
            epoch_loss += loss.detach().item()
            epoch_cls_loss += outputs[detector.module.cls_key].detach().item()
            epoch_box_reg_loss += outputs[detector.module.box_reg_key].detach().item()
            
            print(f"Rank {rank}, {step}/{epoch_len}, train_loss: {loss.item():.4f}")

        end_time = time.time()

        # Gather data from different gpu's to log in wandb
        train_loss = torch.Tensor(train_loss).to(rank)
        epoch_cls_loss = torch.Tensor([epoch_cls_loss]).to(rank)
        epoch_box_reg_loss = torch.Tensor([epoch_box_reg_loss]).to(rank)

        gathering_start_time = time.time()
        
        all_train_loss = [torch.zeros_like(train_loss).to(rank) for _ in range(dist.get_world_size())]
        all_epoch_cls_loss = [torch.zeros(1).to(rank) for _ in range(dist.get_world_size())]
        all_epoch_box_reg_loss = [torch.zeros(1).to(rank) for _ in range(dist.get_world_size())]

        dist.all_gather(all_train_loss, train_loss)
        dist.all_gather(all_epoch_cls_loss, epoch_cls_loss)
        dist.all_gather(all_epoch_box_reg_loss, epoch_box_reg_loss)

        all_train_loss = torch.concat(all_train_loss).cpu()
        all_epoch_cls_loss = torch.concat(all_epoch_cls_loss).cpu()
        all_epoch_box_reg_loss = torch.concat(all_epoch_box_reg_loss).cpu()
        
        train_loss = all_train_loss
        step = len(train_loss)
        epoch_loss = train_loss.sum().item()
        epoch_cls_loss = all_epoch_cls_loss.sum().item()
        epoch_box_reg_loss = all_epoch_box_reg_loss.sum().item()

        epoch_loss /= step
        epoch_cls_loss /= step
        epoch_box_reg_loss /= step

        # save to wandb
        if rank == 0:
            for loss in train_loss:
                wandb.log({"train_loss": loss.item(), "epoch": epoch+1})
                save_dict.update({"train_loss": loss.item(), "epoch": epoch+1})
            
            save_dict.update({"avg_train_loss": epoch_loss, 
                              "avg_train_cls_loss": epoch_cls_loss,
                              "avg_train_box_reg_loss": epoch_box_reg_loss,
                              "train_lr": optimizer.param_groups[0]["lr"],
                             })
            wandb.log(
                 {
                     "avg_train_loss": epoch_loss, 
                     "avg_train_cls_loss": epoch_cls_loss,
                     "avg_train_box_reg_loss": epoch_box_reg_loss,
                     "train_lr": optimizer.param_groups[0]["lr"],
                     "epoch": epoch+1
                 }
            )
            
            
            print(f"Gathering time: {time.time()-gathering_start_time}s")
            print(f"Training time: {end_time-start_time}s")
            print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")
            
        del inputs, batch_data
        torch.cuda.empty_cache()
        gc.collect()
        
            
        if scheduler!=None:
            scheduler.step()

        # ------------- Validation for model selection -------------
        if rank == 0 and ((epoch + 1) % val_interval == 0 or (epoch + 1) == max_epochs):
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
                    val_inputs = [val_data_i.pop("image").to(rank) for val_data_i in val_data]

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
                    val_data_i[detector.module.target_box_key].cpu().detach().numpy() for val_data_i in val_outputs_all
                ],
                pred_classes=[
                    val_data_i[detector.module.target_label_key].cpu().detach().numpy() for val_data_i in val_outputs_all
                ],
                pred_scores=[
                    val_data_i[detector.module.pred_score_key].cpu().detach().numpy() for val_data_i in val_outputs_all
                ],
                gt_boxes=[val_data_i[detector.module.target_box_key].cpu().detach().numpy() for val_data_i in val_targets_all],
                gt_classes=[
                    val_data_i[detector.module.target_label_key].cpu().detach().numpy() for val_data_i in val_targets_all
                ],
                max_detections=detections_per_img
            )

            # assemble parts for validation per patient
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

            # log values
            
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

            #save results
            with open(f"{args.model_path.split('/')[0]}/{run_name}_{run_id}/coco_metric_{run_name}_{run_id}_last.json", "w") as f:
                json.dump(val_epoch_metric_dict, f)

            results_path = f"{args.model_path.split('/')[0]}/{run_name}_{run_id}/results_metric_{run_name}_{run_id}_last.pkl"
            pd.to_pickle(results_metric_dict, results_path)
            
            # save best trained model
            best_val_epoch_metric = val_epoch_metric
            best_val_epoch = epoch + 1
            torch.save(detector.module.network.state_dict(), f"{model_path.split('.')[0]}_{run_name}_{run_id}_val.pt")
            
            with open(f"{args.model_path.split('/')[0]}/{run_name}_{run_id}/coco_metric_{run_name}_{run_id}_val.json", "w") as f:
                json.dump(val_epoch_metric_dict, f)
                
            results_path = f"{args.model_path.split('/')[0]}/{run_name}_{run_id}/results_metric_{run_name}_{run_id}_val.pkl"
            pd.to_pickle(results_metric_dict, results_path)
                
                
            print("saved new best metric model")
            
            print(
                "current epoch: {} current metric: {:.4f} "
                "best metric: {:.4f} at epoch {}".format(
                    epoch + 1, val_epoch_metric, best_val_epoch_metric, best_val_epoch
                )
            )

            del val_targets_all, val_outputs_all, targets 
            torch.cuda.empty_cache()

        if rank == 0:
            # save last trained model
            save_dict.update({"optimizer_state_dict": optimizer.state_dict()})
            torch.save(detector.module.network.state_dict(), f"{model_path.split('.')[0]}_{run_name}_{run_id}_last.pt")
            print("saved last model")
    
            # create checkpoint
            torch.save(
                save_dict, 
                CHECKPOINT_PATH,
            )
    
    torch.cuda.empty_cache()
    gc.collect()  
    
    if rank == 0:
        print(f"train completed, best_metric: {best_val_epoch_metric:.4f} " f"at epoch: {best_val_epoch}")
        wandb.finish()
    return

    

    
    

def main(rank: int, world_size: int):
    # Define global variables
    global args, env_dict, config_dict, train_val
    
    if rank == 0:
        wandb.login(key='KEY')
        print('everything is imported')
        
    # Define args
    parser = argparse.ArgumentParser(description="PyTorch Object Detection Training")
    parser.add_argument(
        "-e",
        "--environment-file",
        default="./config/environment_mbd_script.json",
        help="environment json file that stores environment path",
    )
    parser.add_argument(
        "-c",
        "--config-file",
        default="./config/config_train_mbd_script_16g.json",
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
        "-m",
        "--model",
        default="3ch",
        help="Choose which model to validate (3ch, skeleton, binary_mask, float_mask)."
    )
    parser.add_argument(
        "-f",
        "--fold",
        default="fold0",
        help="Which fold to train and validate",
    )

    ddp_setup(rank, world_size)

    args = parser.parse_args()

    # Choose settings of data
    if args.model == "3ch":    
        modalities=["T1", "ADC", "b1000"]
        name = "_".join(modalities)
        with_attention_transform=None
        with_binary=False
        
    elif args.model == "skeleton":    
        modalities=["T1", "ADC", "b1000", "skeleton"]
        name = modalities[-1]
        with_attention_transform=None
        with_binary=True
        
    elif args.model == "binary_mask":    
        modalities=["T1", "ADC", "b1000", "attn_mask2"]
        name = modalities[-1]
        with_attention_transform=None
        with_binary=True
        
    elif args.model == "float_mask":    
        modalities=["T1", "ADC", "b1000"]
        name = "attn_mask4"
        with_attention_transform="mask4"
        with_binary=False
    
    else:
        raise ValueError('Did not give a valid model.')
    
    # Choose settings of data
    training_key="training_chunks"
    validation_key="validation"
    sizes=['>', 7.5]
    cv_split=5
    base_anchor_shapes=None

    # Hyperparamters
    num_epochs = 1500
    patch_size = [192, 192, 80]
    optim = 'SGD'
    scheduler ='cosine'
    learning_rate = 1e-3
    pretrained_backbone = False
    pretrained_model = None 
    val_interval = 100 
    detections_per_img = 150
    run_id = None
    if len(modalities)>3:
        name = modalities[-1]
    else:
        name = "_".join(modalities)


    # Settings for train run
    env_dict, config_dict, train_val_fold = prepare_json_files(
        args=args,                                
        modalities=modalities,
        with_attention_transform=with_attention_transform,
        sizes=sizes,
        cv_split=cv_split,
        base_anchor_shapes=base_anchor_shapes,
        use_2=True
    )
    config_dict["chunks"] = training_key=="training_chunks"
    config_dict["validation_chunks"] = validation_key=="validation_chunks"
    config_dict["training_key"] = training_key
    config_dict['validation_key'] = validation_key
    
    for k, v in env_dict.items():
        setattr(args, k, v)
    for k, v in config_dict.items():
        setattr(args, k, v)
        
        
    setattr(args, 'with_binary', with_binary)
    setattr(args, 'test', train_val_fold['test'])
    setattr(args, 'test_chunks', train_val_fold['test_chunks'])

    
    fold = args.fold 
    train_val = train_val_fold[fold]
    

    fold_name = name + '_' + fold

    args.data_list_file_path = 'train_val_script_3.json'
    
    with open(args.data_list_file_path, 'w') as f:
        json.dump(train_val, f)  


    train(
        rank,
        world_size,
        num_epochs = num_epochs, 
        patch_size = patch_size,
        optim = optim,
        scheduler = scheduler,
        learning_rate = learning_rate,
        pretrained_backbone = pretrained_backbone,
        pretrained_model = pretrained_model, 
        val_interval = val_interval,
        detections_per_img = detections_per_img,
        run_id = run_id,
        name = fold_name,
        group = fold
    )

    cleanup()


if __name__ == "__main__":
    logging.basicConfig(
        stream=sys.stdout,
        level=logging.INFO,
        format="[%(asctime)s.%(msecs)03d][%(levelname)5s](%(name)s) - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    world_size = torch.cuda.device_count()
    print('World size:', world_size)
    mp.spawn(main, args=(world_size,), nprocs=world_size)