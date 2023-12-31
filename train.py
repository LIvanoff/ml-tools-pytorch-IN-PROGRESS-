import os
import logging
from pathlib import Path
import random
import sys
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
from torch.optim import lr_scheduler
from tqdm import tqdm

from utils.torch_utils import select_optimizer, select_model, select_loss
from utils.data_preprocessing import check_dataset, create_dataset
from utils.plot import plot_loss, save_figure
from utils.metrics import *
from utils.utils import *

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
        batch_size: int = None,
        task: str = None,
        metric: str = None,
        sep: int = None,
        permutate: bool = True,
        plot: bool = False,
        workers: int = 8,
        augment: bool = False,
        save_plot: bool = False,
        freeze: int = None,
        scheduler: object = None,
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

    if scheduler is not None and optimizer is None:
        msg = "Missing an instance of the optimizer"
        raise NotImplementedError(msg)

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
        criterion, loss_name = select_loss(loss_name)
    else:
        loss_name = 'Loss'

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
        X, y = data[0:, :sep], data[0:, sep:]
    else:
        X, y = dataset[0], dataset[1]

    '''
    Часть кода для присваваивания значений X,y в случае с CV данными
    
    
    
    
    '''

    if labels is None:
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=test_size, random_state=seed,stratify=y)
    else:
        X_train, X_val = train_test_split(X, test_size=test_size, random_state=seed, stratify=labels)

    if labels is None:
        num_classes = len(np.unique(y_train))
    else:
        num_classes = len(np.unique(labels))

    if metric is not None:
        metric = select_metric(metric=metric, num_classes=num_classes)
    elif task is not None:
        metric = select_metric(task=task, num_classes=num_classes)
        save_plot = True

    if batch_size is None:
        warnings.warn("batch_size not specified, this can lead to a problem with network training, specify the size "
                      "of the batch")

    Xy_train = create_dataset(X_train,
                             permutate,
                             workers,
                             batch_size,
                             augment,
                             filetype,
                             image_size,
                             y_data=y_train,
                             mode='train')
    if test_size is not None:
        Xy_val = create_dataset(X_val,
                               permutate,
                               workers,
                               batch_size,
                               augment,
                               filetype,
                               image_size,
                               y_data=y_val,
                               mode='val')

    model.to(device)
    history = {'train_loss': [],
               'val_loss': [],
               'train_met': [],
               'val_met': []
               }
    # train_history = []
    # val_history = []
    # metric_train_history = []
    # metric_val_history = []
    log_template = "\nEpoch {ep:03d} train_loss: {t_loss:0.4f} " \
                   "train_metric {t_met:0.4f}   val_loss: {v_loss:0.4f} " \
                   "val_metric {v_met:0.4f}"

    best_acc = -1
    plt.ion()
    with tqdm(desc="epoch", total=epochs) as pbar_outer:
        for epoch in range(epochs):
            model.train()
            loss_value = 0.0
            metric_value = 0.0
            processed_data = 0
            processed_size = 0
            for X_batch, Y_batch in Xy_train:
                optimizer.zero_grad()
                X_batch, Y_batch = to_device(device, X_batch, Y_batch)

                preds = model(X_batch)
                loss = criterion(preds, Y_batch)
                loss.backward()
                optimizer.step()

                loss_value += loss.item() * X_batch.size(0)
                metric_value += metric(preds, Y_batch).item()  # metric_value += torch.sum(preds == Y_batch)
                processed_data += X_batch.size(0)
            if scheduler is not None:
                scheduler.step()

            history['train_loss'].append(loss_value / processed_data)  # train_loss = loss_value / processed_data
            history['train_met'].append(metric_value / len(Xy_train))  # metric_train = metric_value / len(Xy_train)
            # train_loss = loss_value / processed_data
            # metric_train = metric_value / len(Xy_train)

            # train_history.append(train_loss)
            # metric_train_history.append(metric_value)

            loss_value = 0.0
            metric_value = 0.0
            if test_size is not None:
                with torch.no_grad():
                    model.eval()
                    for X_val, Y_val in Xy_val:
                        X_val, Y_val = to_device(device, X_val, Y_val)
                        preds = model(X_val)
                        loss = criterion(preds, Y_val)
                        metric_value += metric(preds, Y_val).item()

                    loss_value += loss.item() * X_val.size(0)
                    # metric_val = metric_value / len(Xy_val)
                    history['val_met'].append(metric_value / len(Xy_val))
                    processed_size += X_val.size(0)

                history['val_loss'].append(loss_value / processed_size)  # val_loss = loss_value / processed_size

                # val_history.append(val_loss)
                # metric_val_history.append(metric_value)

            if plot and epochs % 10 == 0:
                # plot_loss(train_history, val_history, loss_name)
                plot_loss(history['train_loss'], history['val_loss'], loss_name)

            pbar_outer.update(1)
            tqdm.write(log_template.format(ep=epoch + 1,
                                           t_loss=history['train_loss'][epoch],
                                           t_met=history['train_met'][epoch],
                                           v_loss=history['val_loss'][epoch],
                                           v_met=history['val_met'][epoch]))

            # current_acc = test(test_dataloader, model, device, task)
            #
            #
            # # Save Model Checkpoint Regularly
            # if (epoch + 1) % checkpoint_every == 0:
            #     print("checkpoint saved at epoch {}".format(epoch))
            #     save_checkpoint(epoch, model, checkpoint_dir, best=False)
            #
            # # Save Best Model Checkpoint
            # if (current_acc >= best_acc):
            #     best_acc = current_acc
            #     print("best model saved at epoch {}".format(epoch))
            #     save_checkpoint(epoch, model, checkpoint_dir, best=True)

    plt.ioff()
    plt.show()

    if save_plot is True:
        save_figure()

    torch.cuda.empty_cache()
    return model, history


if __name__ == '__main__':
    from SimpleNN import Net
    model = Net()
    model, metric = train(
        epochs=50,
        lr=0.001,
        model=model,
        # model_name='resnet18',
        # weights='ResNet18_Weights.IMAGENET1K_V1.pt',
        optimizer_name='Adam',
        loss_name='crossentropy',
        sep=13,
        device='cpu',
        metric='accuracy',
        dataset_path='dataset.xlsx',
        plot=True,
        batch_size=10000,
        test_size=0.2
    )
