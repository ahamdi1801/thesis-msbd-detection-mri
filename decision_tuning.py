import argparse
import gc
import json
import logging
import sys
import time
import os
import pandas as pd
import tabulate
import SimpleITK as sitk
import glob
from copy import deepcopy

import numpy as np
import torch
import matplotlib.pyplot as plt
from copy import deepcopy
from results_toolkit import DetectionMetrics2, create_image_preds
from monai.apps.detection.metrics.matching import matching_batch
from script_utils import create_json_cv_dataset_parts, create_json_cv_dataset, prepare_json_files_luna, load_luna16
from joblib import Parallel, delayed
from functools import partial

def get_settings(model, fold):
    runs=glob.glob("./cv_models/*/test_results*last*")
    if model == 'LUNA16_20':
        dict_runs = {x.split('/')[2]: x for x in runs}
        path = dict_runs[f"{fold}_{model}"]
    else:
        dict_runs = {"_".join(x.split('/')[2].split('_')[:-1]): x for x in runs}
        path = dict_runs[f"{model}_{fold}"]
        
    fold_settings = {fold: {'nms_th': 'default', 'score_th': 'default'}}
    
    results = pd.read_pickle(path)
    
    detection_metrics = DetectionMetrics2(results)
    max = 0
    for nms_th in np.arange(0.01, .23, .01):
        detection_metrics.set(nms_th=nms_th)
        for score_th in np.arange(0.02, .4, .02):
            detection_metrics.set(score_th=score_th)
            df = detection_metrics.get_report(greedy=True).loc["0.1"]
            precision = df.loc[:, 'precision']
            sensitivity = df.loc[:, 'recall']
            avg_f2 = round((5*sensitivity*precision/(sensitivity+4*precision)).mean(), 2)
    
            if avg_f2 > max:
                max = avg_f2
                fold_settings[fold]['nms_th'] = round(nms_th, 2)
                fold_settings[fold]['score_th'] = round(score_th, 2)

    return fold_settings

def get_avg_f2(detection_metrics, nms_th_list, score_th_range=np.arange(0.02, .4, .02)):
    detection_metrics = deepcopy(detection_metrics)
    avg_f2_list = []
    for nms_th in nms_th_list:
        detection_metrics.set(score_th=nms_th)
        for score_th in score_th_range:
            detection_metrics.set(score_th=score_th)
            df = detection_metrics.get_report(greedy=True).loc["0.1"]
            precision = df.loc[:, 'precision']
            sensitivity = df.loc[:, 'recall']
            avg_f2 = round((5*sensitivity*precision/(sensitivity+4*precision)).mean(), 2)
            avg_f2_list.append(avg_f2)

    return avg_f2_list
    

def main():
    parser = argparse.ArgumentParser(description="Post-tuning decision threshold")
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
        default="skeleton_parts",
        help="Which fold to train and validate",
    )

    args = parser.parse_args()

    model = args.model
    folds = [f"fold{i}" for i in range(5)]

    settings = {
        f'fold{i}': {'nms_th': 'default', 'score_th': 'default'} for i in range(5)
    }
    
    parallel = Parallel(n_jobs=5, prefer='processes', max_nbytes=None)
    
    fold_settings = parallel(delayed(partial(get_settings, model=model))(fold=fold) for fold in folds)

    model_settings = {}
    for i in range(5):
        model_settings.update(fold_settings[i])

    pd.to_pickle(model_settings, f'./cv_models/{model}_test_settings.pkl')





if __name__ == "__main__":
    logging.basicConfig(
        stream=sys.stdout,
        level=logging.INFO,
        format="[%(asctime)s.%(msecs)03d][%(levelname)5s](%(name)s) - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    main()