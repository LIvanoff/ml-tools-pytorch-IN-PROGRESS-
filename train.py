import argparse
import os
import logging
from pathlib import Path
import random
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import lr_scheduler
from tqdm import tqdm

from utils.torch_utils import select_optimizer, select_model, select_loss
from utils.data_preprocessing import check_dataset

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))

LOCAL_RANK = int(os.getenv('LOCAL_RANK', -1))
RANK = int(os.getenv('RANK', -1))
WORLD_SIZE = int(os.getenv('WORLD_SIZE', 1))


def train(
        epochs: int,
        dataset_path: str,
        loss_function: object = None,
        dataset: tuple = None,
        model: object = None,
        test_size: float = None,
        lr: float = None,
        model_name: str = '',
        repo_or_dir='',
        weights: str = '',
        optimizer: object = None,
        optimizer_name: str = 'Adam',
        loss_name: str = '',
        device: str = 'cpu',
        batch_size: str = 64,
        sep: int = None,
        permutate: bool = True,
        plot_loss: bool = True,
        workers: int = 8,
        augment: bool = False,
        save_plot: bool = True,
        freeze: int = None,
        project_path: str = ROOT / 'runs/train',
        name: str = 'exp',
        image_size: int = 224,
        seed: int = 0
    ):
    device = str(device).strip().lower().replace('cuda:', '').replace('gpu', '0').replace('none', 'cpu')

    if device == '':
        device = 'cpu'

    cpu = device == 'cpu'
    if not cpu:
        assert torch.cuda.is_available() and torch.cuda.device_count() >= len(device.replace(',', '')), \
            f"Invalid CUDA '--device {device}' requested, use device='cpu' or pass valid CUDA device(s)"

    msg = ''
    if not cpu and torch.cuda.is_available():
        devices = range(torch.cuda.device_count())
        n = len(devices)
        if n > 1 and batch_size > 0:
            assert batch_size % n == 0, f'batch-size {batch_size} not multiple of GPU count {n}'
        space = ' ' * (len(msg) + 1)
        for i, d in enumerate(devices):
            p = torch.cuda.get_device_properties(i)
            msg += f"{'' if i == 0 else space}CUDA:{d} ({p.name}, {p.total_memory / (1 << 20):.0f}MiB)\n"  # bytes to MB
        device = 'cuda:0'
        logging.info(msg)

    assert (repo_or_dir != '' and model_name != '') or (model_name != '' and weights) or model is not None, \
        "There is no model and weights, pass an instance of the model or " \
        "specify model_name, weights and optionally repo_or_dir"

    if model_name != '' and model is None:
        model = select_model(repo_or_dir, model_name, weights)
    else:
        assert model_name == '' and model is not None, \
            "Pass only model_name or an instance of the model"

    # assert optimizer_name != '' or optimizer is not None, "Pass only optimizer_name or an instance of the optimizer"

    if optimizer is None:
        optimizer = select_optimizer(optimizer_name, model, lr)
    if freeze is not None:
        for i, child in enumerate(model.named_parameters()):
            if i <= freeze:
                for param in child.parameters():
                    param.requires_grad = False

    assert loss_function is not None or loss_name != '', "The name of the loss function and object is missing," \
                                                         "define the name or pass instance of " \
                                                         "the loss function"
    if loss_function is None:
        criterion = select_loss(loss_name)

    if device != 'cpu' and RANK == -1 and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    labels = None
    if dataset is None:
        dataset_path, filetype = check_dataset(dataset_path, ROOT, sep)
        if filetype == '.xlsx':
            data = pd.read_excel(dataset_path[0])
        elif filetype == '.csv' or '.txt':
            data = pd.read_csv(dataset_path[0])
        data = data.values
        X, y = data[0:, :sep].values, data[0:, sep:]
    else:
        X, y = dataset[0], dataset[1]

    '''
    Часть кода для присваваивания значений X,y в случае с CV данными
    
    
    
    
    '''

    if labels is None:
        X_train, X_val, y_train, y_val  = train_test_split(X, y, test_size=test_size, random_state=seed)
    else:
        X_train, X_val = train_test_split(X, test_size=test_size, random_state=seed, stratify=labels)

    # if RANK == {-1, 0}:

    torch.cuda.empty_cache()
    return model


if __name__ == '__main__':
    train(
        epochs=10,
        lr=0.01,
        model_name='resnet18',
        weights='ResNet18_Weights.IMAGENET1K_V1.pt',
        optimizer_name='SGD',
        loss_name='mse',
        sep=4,
        dataset_path='dataset.xlsx'
        )
