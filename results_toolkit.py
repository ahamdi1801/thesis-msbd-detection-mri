import numpy as np
import torch
import SimpleITK as sitk
import pandas as pd
import matplotlib.pyplot as plt
import os
import json

from copy import deepcopy
from monai.apps.detection.metrics.coco import COCOMetric
from monai.apps.detection.metrics.matching import matching_batch
from greedy_matching import greedy_matching_batch
from monai.data import box_utils

class DetectionMetrics2():
    def __init__(self, results_dict, labels=[0], iou_thresholds=[.1, .5, .75, .9], max_detections=150, score_th=0, nms_th=0, size=['>', 0]):
        self._spacing = json.load(open('./config/config_train_mbd_script_16g.json'))['spacing']
        self._labels = labels
        self._max_detections = max_detections
        self._score_th = score_th
        self._nms_th = nms_th
        self._size = size
        self._preds = deepcopy(results_dict['preds'])
        self._new_preds = deepcopy(results_dict['preds'])
        self._gts = deepcopy(results_dict['gt'])
        self._new_gts = deepcopy(results_dict['gt'])
        self._patients = self._get_patients()
        self._iou_thresholds = list(results_dict['iou_thresholds'])
        self._dt_matches = None           # (#patients, (#iou_thresholds, #detections))
        self._dt_greedy_matches = None    # (#patients, (#iou_thresholds, #detections))
        self._gt_matches = None           # (#patients, (#iou_thresholds, #gt))
        self._dt_scores = None            # (#pred_scores)
        self._results = self._get_new_results()

    def set_max_detections(self, max_detections):
        self._max_detections = max_detections
        self._results = self._get_new_results()
        return

    def set_score_th(self, score_th):
        self._score_th = score_th
        self._results = self._get_new_results()
        return

    def set_nms_th(self, nms_th):
        self.nms_th = nms_th
        self._results = self._get_new_results()
        return

    def set_iou_thresholds(self, iou_thresholds):
        self._iou_thresholds = iou_thresholds
        self._results = self._get_new_results()
        return

    def set_size(self, size):
        self._size = size
        self._results = self._get_new_results()
        return

    def set(self, max_detections=None, score_th=None, nms_th=None, iou_thresholds=None, size=None):
        if max_detections != None:
            self._max_detections = max_detections
        if score_th != None:
            self._score_th = score_th
        if nms_th != None:
            self._nms_th = nms_th
        if iou_thresholds != None:
            self._iou_thresholds = iou_thresholds
        if size != None:
            self._size = size
        self._results = self._get_new_results()
        return

    

    def _get_patients(self):
        patients = []
        for x in self._gts:
            if isinstance(x['image'], list):
                patients.append(x['image'][0].split('/')[0])
            else:
                patients.append(x['image'].split('/')[0])
        return patients
            

    def _get_new_results(self):
        preds = deepcopy(self._preds)
        gts = deepcopy(self._gts)
        for i, pred in enumerate(preds):
            indices = list(range(min((pred['label_scores']>self._score_th).sum(), self._max_detections)))
            if self._nms_th > 0:
                to_keep = box_utils.non_max_suppression(pred['box'][indices], pred['label_scores'][indices], nms_thresh = self._nms_th)
                indices = list(set(to_keep.tolist()) & set(indices))
            if self._size[1] > 0:
                if self._size[0] == '>':
                    to_keep = np.where(self._get_size(pred['box'])>self._size[1])[0]
                elif self._size[0] == '<':
                    to_keep = np.where(self._get_size(pred['box'])<self._size[1])[0]
                else:
                    raise ValueError('Operator has to be "<" or ">"')
                indices = list(set(to_keep.tolist()) & set(indices))
                
            pred = {k: v[indices] for k, v in pred.items()}
            preds[i] = pred

        if self._size[1] > 0:
            for i, gt in enumerate(gts):
                if self._size[0] == '>':
                    indices = np.where(self._get_size(gt['box'])>self._size[1])[0]
                elif self._size[0] == '<':
                    indices = np.where(self._get_size(gt['box'])<self._size[1])[0]
                else:
                    raise ValueError('Operator has to be "<" or ">"')
                    
                gt['box'] = np.array(gt['box'])[indices]
                gt['label'] = np.array(gt['label'])[indices]
                gts[i] = gt
                
    
        results = greedy_matching_batch(
            iou_fn=box_utils.box_iou,
            iou_thresholds=self._iou_thresholds,
            pred_boxes=[x['box'].cpu().numpy() for x in preds],
            pred_classes=[x['label'].cpu().numpy() for x in preds],
            pred_scores=[x['label_scores'].cpu().numpy() for x in preds],
            gt_boxes=[np.array(x['box']) for x in gts],
            gt_classes=[np.array(x['label']) for x in gts],
            max_detections=self._max_detections
        )

        self._new_preds = preds
        self._new_gts = gts
        self._dt_matches = [x[label]['dtMatches'] for label in self._labels for x in results]               # (#patients, (#iou_thresholds, #detections))
        self._dt_greedy_matches = [x[label]['dtGreedyMatches'] for label in self._labels for x in results]  # (#patients, (#iou_thresholds, #detections))
        self._gt_matches = [x[label]['gtMatches'] for label in self._labels for x in results]               # (#patients, (#iou_thresholds, #gt))
        self._dt_scores = [x[label]['dtScores'] for label in self._labels for x in results]                 # (#pred_scores)
        
        return results

    def _get_vol_size(self, box_preds):
        if isinstance(box_preds, list):
            box_preds = np.array(box_preds)
        size = 1
        for i in range(3):
            size *= box_preds[:, i+3] - box_preds[:,i] 
        return size

    def _get_diameter_size(self, box_preds):
        if isinstance(box_preds, list):
            box_preds = np.array(box_preds)
        elif isinstance(box_preds, torch.Tensor):
            box_preds = box_preds.cpu().numpy()
        diameter_size = np.zeros((len(box_preds), 3))
        for i in range(3):
            diameter_size[:, i] = self._spacing[i]*(box_preds[:, i+3] - box_preds[:,i])
        diameter_size = diameter_size.max(axis=1)  
        return diameter_size

    def _get_size(self, box_preds):
        return self._get_diameter_size(box_preds)

    def get_coco_results(self):
        return self._coco_metric(self._results)


    def get_score_th(self):
        return self._score_th
    
    def get_max_detections(self):
        return self._max_detections

    def get_report(self, greedy=True):
        patients = self._patients.copy()
        iou_thresholds = self._iou_thresholds
        dt_matches = self._dt_greedy_matches if greedy else self._dt_matches
        gt_matches = self._gt_matches
        dt_scores = self._dt_scores
        report = pd.DataFrame(columns=['iou_th', 'patient','lesions', 'tp', 'fp' ,'fn' ,'precision' ,'recall'])
        # first_step=True
        for i, iou_th in enumerate(iou_thresholds):
            for patient, dt_match, gt_match, dt_score in zip(patients, dt_matches, gt_matches, dt_scores):
                lesions = len(gt_match[i])
                nr_preds = len(dt_score)
                tp = gt_match[i].sum()
                fp = nr_preds - dt_match[i].sum()
                fn = lesions - tp
                precision = tp/(tp + fp + np.spacing(0))
                recall = tp/(tp + fn + np.spacing(0))
                report.loc[len(report), :] = (str(round(iou_th, 2)), patient, lesions, tp, fp, fn, precision, recall)
        
        report = report.set_index(['iou_th', 'patient'])

        return report

    def pr_curve(self, iou_th, save_filename=None):
        iou_to_idx = {iou_key: i for i, iou_key in enumerate(self._iou_thresholds)}
        idx = iou_to_idx[iou_th]

        precision = self._coco_metric._compute_statistics(self._results)['precision'][idx].flatten()
        recall = self._coco_metric.recall_thresholds
        
        plt.figure()
        plt.title(f'Precision-Recall for IoU = {round(iou_th, 2)}')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.plot(recall, precision)
        if save_filename != None:
            plt.savefig(save_filename, format='png')
        plt.show()
        return
                 
    def get_statistics(self):
        return self._coco_metric._compute_statistics(self._results)
        
    def get_preds(self):
        return self._preds

    def get_gts(self):
        return self._gts
        
    def get_iou_thresholds(self):
        return self._iou_thresholds

    def get_dt_matches(self):
        return self._dt_matches
        
    def get_gt_matches(self):
        return self._gt_matches
    
    def get_dt_scores(self):
        return self._dt_scores

    def get_patients(self):
        return self._patients
        
    def get_results(self):
        return self._results

    def get_new_results_dict(self):
        results_dict = {
            'results': self._results,
            'preds': self._new_preds,
            'gt': self._new_gts,
            'iou_thresholds': self._iou_thresholds
        }
        return results_dict


def create_image_preds(run_name, results_metric=None, iou_th_idx=0, params=None, folder="./cv_models", mode='last', finetuned=False, save=False):
    if results_metric==None:
        results_metric = pd.read_pickle(f'{folder}/{run_name}/results_metric_{run_name}_{mode}.pkl')
    os.makedirs(f'./image_preds/{run_name}', exist_ok=True)
    
    
    writer = sitk.ImageFileWriter()
    if params==None:
        all_box_preds = [x['box'].cpu().numpy() for x in results_metric['preds']]
        all_box_gts = [np.array(x['image']) for x in results_metric['gt']]
        image_paths = [x['image'][0] for x in results_metric['gt']]
        all_dtmatches = [np.array(x[0]['dtMatches'][iou_th_idx], dtype=bool) for x in det_met._dt_matches]
        all_box_matches = [box_preds[idx] for box_preds, idx in zip(all_box_preds, all_dtmatches)]
    else:
        det_met = DetectionMetrics2(results_metric)
        det_met.set(**params)
        results = det_met.get_results()
        
        all_box_preds = [x['box'].cpu().numpy() for x in det_met._new_preds]
        all_box_gts = [np.array(x['image']) for x in det_met._new_gts]
        image_paths = [x['image'][0] for x in det_met._new_gts]
        all_dtmatches = [np.array(x[0]['dtMatches'][iou_th_idx], dtype=bool) for x in results]
        all_box_matches = [box_preds[idx] for box_preds, idx in zip(all_box_preds, all_dtmatches)]
    
    iou_th = results_metric['iou_thresholds'][iou_th_idx]
    bbox_draw_dict = {'img': [], 'path': []}
    
    for box_preds, dt_matches, image_path in zip(all_box_preds, all_dtmatches, image_paths):
        path = image_path.split('/')
        path[-1] = 'GT_resampled.nii.gz'
        path= "./data/"+"/".join(path)
        img = sitk.ReadImage(path)
        img_arr =  sitk.GetArrayFromImage(img)
        img_arr[:, :, :] = 0
    
        for box_pred, dt_match in zip(box_preds, dt_matches):
            x_indices = np.arange(round(box_pred[0]), round(box_pred[3])+1, dtype=int)
            y_idx = (round(box_pred[1]), round(box_pred[4])+1)
            z_idx = (round(box_pred[2]), round(box_pred[5])+1)
            for x_idx in x_indices:
                img_arr[z_idx[0]:z_idx[1], y_idx[0]:y_idx[1], x_idx] += 1
    
        new_img = sitk.GetImageFromArray(img_arr)
        new_img.SetOrigin(img.GetOrigin())
        new_img.SetSpacing(img.GetSpacing())
        new_img.SetDirection(img.GetDirection())

        if finetuned==False:
            new_image_path = f"./image_preds/{run_name}/{image_path.split('/')[0]}_preds_{mode}.nii.gz"
        else:
            new_image_path = f"./image_preds/{run_name}/{image_path.split('/')[0]}_finetuned_preds_{mode}.nii.gz"
            
        if save:
            writer.SetFileName(new_image_path)
            writer.Execute(new_img)
    
        bbox_draw_dict['img'].append(new_img)
        bbox_draw_dict['path'].append(new_image_path)

    return bbox_draw_dict


# class DetectionMetrics(): 
#     def __init__(self, results_dict, labels=[0], iou_thresholds=[.1, .5, .75, .9], max_detections=150, score_th = None):
#         self._coco_metric = COCOMetric(classes=["lesion"], iou_list=iou_thresholds, max_detection=[max_detections])
#         self._labels = labels
#         self._max_detections = max_detections
#         self._score_th = score_th
#         self.nms_th = nms_th
#         self._original_results = deepcopy(results_dict['results'])
#         self._preds = deepcopy(results_dict['preds'])
#         self._gt = deepcopy(results_dict['gt'])
#         self._patients = self._get_patients()
#         self._iou_thresholds = list(results_dict['iou_thresholds'])
#         self._dt_matches = [x[label]['dtMatches'][:, :max_detections] for label in labels for x in self._original_results] # (#patients, (#iou_thresholds, #detections))
#         self._gt_matches = [x[label]['gtMatches'] for label in labels for x in self._original_results]                     # (#patients, (#iou_thresholds, #gt))
#         self._dt_scores = [x[label]['dtScores'][:max_detections] for label in labels for x in self._original_results]      # (#pred_scores)
        
#         if self._score_th != None:
#             indices = [(x>=self._score_th).sum() for x in self._dt_scores]
#             self._dt_matches = [x[label]['dtMatches'][:, :idx] for label in labels for x, idx in zip(self._original_results, indices)] # (#patients, (#iou_thresholds, #detections))
#             self._dt_scores = [x[label]['dtScores'][:idx] for label in labels for x, idx in zip(self._original_results, indices)]      # (#pred_scores)

#         if self.nms_th != None:
#             indices = [
#                 box_utils.non_max_suppression(preds, scores, nms_thresh=self.nms_th) for preds, scores in zip(self._preds, self._dt_scores)
#             ]
#             self._dt_matches = [x[:, idx] for x, idx in zip(self._dt_matches, indices)] # (#patients, (#iou_thresholds, #detections))
#             self._dt_scores = [x[:, idx] for x, idx in zip(self._dt_scores, indices)]      # (#pred_scores)
            
            

#         self._results = self._get_new_results()
            

#     def _get_patients(self):
#         patients = []
#         for x in self._gt:
#             if isinstance(x['image'], list):
#                 patients.append(x['image'][0].split('/')[0])
#             else:
#                 patients.append(x['image'].split('/')[0])
#         return patients

#     def set_max_detections(self, max_detections):
#         self._max_detections = max_detections
#         self._dt_matches = [x[label]['dtMatches'][:, :max_detections] for label in labels for x in self._results] # (#patients, (#iou_thresholds, #detections))
#         self._dt_scores = [x[label]['dtScores'][:max_detections] for label in labels for x in self._results]      # (#pred_scores)
#         self._results = _get_new_results()
#         return

#     def set_score_th(self, score_th):
#         self._score_th = score_th
#         indices = [(x>=self._score_th).sum() for x in self._dt_scores]
#         self._dt_matches = [x[label]['dtMatches'][:, :idx] for label in labels for x, idx in zip(self._results, indices)] # (#patients, (#iou_thresholds, #detections))
#         self._dt_scores = [x[label]['dtScores'][:idx] for label in labels for x, idx in zip(self._results, indices)]      # (#pred_scores)
#         self._results = _get_new_results()
#         return

#     def set_nms_th(self, nms_th):
#         self.nms_th = nms_th
#         indices = [
#             box_utils.non_max_suppression(preds, scores, nms_thresh=self.nms_th) for preds, scores in zip(self._preds, self._dt_scores)
#         ]
#         self._dt_matches = [x[:, idx] for x, idx in zip(self._dt_matches, indices)] # (#patients, (#iou_thresholds, #detections))
#         self._dt_scores = [x[:, idx] for x, idx in zip(self._dt_scores, indices)]      # (#pred_scores)

#     def _get_new_results(self):
#         results = []
#         if self._score_th != None:
#             indices = [(x>=self._score_th).sum() for x in self._dt_scores]
#         else:
#             indices = [self._max_detections for _ in range(len(self._dt_scores))]

#         if self._nms_th != None:
#             preds = self._preds
#         for x, idx in zip(self._original_results, indices):
#             entry = {}
#             for label in self._labels:
#                 label_entry = {}
#                 for k, v in x[label].items():
#                     label_entry[k] = v[..., :min(idx, self._max_detections)] if k != 'gtMatches' else v
#                 entry[label] = label_entry
#             results.append(entry)
            
#         return results

#     def _get_new_results(self):
#         results = []
#         if self._score_th != None:
#             indices = [(x>=self._score_th).sum() for x in self._dt_scores]
#         else:
#             indices = [self._max_detections for _ in range(len(self._dt_scores))]
#         for x, idx in zip(self._original_results, indices):
#             entry = {}
#             for label in self._labels:
#                 label_entry = {}
#                 for k, v in x[label].items():
#                     label_entry[k] = v[..., :min(idx, self._max_detections)] if k != 'gtMatches' else v
#                 entry[label] = label_entry
#             results.append(entry)

#         return results

#     def get_mr_results(self):
#         return

#     def get_coco_results(self):
#         return self._coco_metric(self._results)


#     def get_score_th(self):
#         return self._score_th
    
#     def get_max_detections(self):
#         return self._max_detections

#     def get_report(self):
#         patients = self._patients.copy()
#         iou_thresholds = self._iou_thresholds.copy()
#         dt_matches = self._dt_matches.copy()
#         gt_matches = self._gt_matches.copy()
#         dt_scores = self._dt_scores.copy()
#         report = pd.DataFrame(columns=['iou_th', 'patient','lesions', 'tp', 'fp' ,'fn' ,'precision' ,'recall'])
#         # first_step=True
#         for i, iou_th in enumerate(iou_thresholds):
#             for patient, dt_match, gt_match, dt_score in zip(patients, dt_matches, gt_matches, dt_scores):
#                 lesions = len(gt_match[i])
#                 nr_preds = len(dt_score)
#                 tp = dt_match[i].sum()
#                 fp = nr_preds - dt_match[i].sum()
#                 fn = lesions - tp
#                 precision = tp/(tp + fp + np.spacing(0))
#                 recall = tp/(tp + fn + np.spacing(0))
#                 report.loc[len(report), :] = (str(round(iou_th, 2)), patient, lesions, tp, fp, fn, precision, recall)
        
#         report = report.set_index(['iou_th', 'patient'])

#         return report

#     def pr_curve(self, iou_th, save_filename=None):
#         iou_to_idx = {iou_key: i for i, iou_key in enumerate(self._iou_thresholds)}
#         idx = iou_to_idx[iou_th]

#         precision = self._coco_metric._compute_statistics(self._results)['precision'][idx].flatten()
#         recall = self._coco_metric.recall_thresholds
        
#         plt.figure()
#         plt.title(f'Precision-Recall for IoU = {round(iou_th, 2)}')
#         plt.xlabel('Recall')
#         plt.ylabel('Precision')
#         plt.plot(recall, precision)
#         if save_filename != None:
#             plt.savefig(save_filename, format='png')
#         plt.show()
#         return
                 
#     def get_statistics(self):
#         return self._coco_metric._compute_statistics(self._results)
        
#     def get_preds(self):
#         return self._preds

#     def get_gt(self):
#         return self._gt
        
#     def get_iou_thresholds(self):
#         return self._iou_thresholds

#     def get_dt_matches(self):
#         return self._dt_matches
        
#     def get_gt_matches(self):
#         return self._gt_matches
    
#     def get_dt_scores(self):
#         return self._dt_scores

#     def get_patients(self):
#         return self._patients



