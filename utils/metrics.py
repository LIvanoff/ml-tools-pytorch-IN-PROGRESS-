import torch
import numpy as np
from utils.torch_utils import RMSELoss
from torchmetrics.classification import BinaryAccuracy, Accuracy, MulticlassAccuracy, MulticlassF1Score, BinaryROC, ROC
from torchmetrics.detection import IntersectionOverUnion, MeanAveragePrecision
from torchmetrics.regression import R2Score


def select_metric(metric: str = None, task: str = None, num_classes: int = None):
    if metric is not None:
        metric = metric.lower()
        if metric == 'iou':
            return IntersectionOverUnion()
        elif metric == 'accuracy':
            return Accuracy(task="multiclass", num_classes=num_classes)
        elif metric == 'rmse':
            return RMSELoss()
        elif metric == 'map':
            return MeanAveragePrecision()
        elif metric == 'f1score':
            return MulticlassF1Score()
    else:
        task = task.lower()
        if task == 'regression':
            return RMSELoss()
        elif task == 'classification':
            return  # accuracy_score()
        elif task == 'segmentation':
            return IntersectionOverUnion()
        elif task == 'detection':
            return MeanAveragePrecision()
