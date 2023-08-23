import torch
from torch import nn
import numpy as np


def iou_metric(heatmap, gt_mask, epsilon=1e-8):
    detection_mask = (heatmap >= 0.5).astype(heatmap.dtype)
    intersect = np.sum(detection_mask * gt_mask)
    detection_sum = np.sum(detection_mask)
    gt_sum = np.sum(gt_mask)
    union = detection_sum + gt_sum
    iou = intersect / (union - intersect + epsilon)
    return iou
