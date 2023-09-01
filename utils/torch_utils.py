import re
import logging
import sys

from utils.loss_function import RMSELoss

import torch
import torch.nn as nn
from torchvision import models


def select_loss(name):
    name = name.lower()
    if name == 'mae' or 'l1loss' or 'maeloss':
        return nn.L1Loss()
    elif name == 'mse' or 'l2loss' or 'mseloss':
        return nn.MSELoss()
    elif name == 'crossentropy' or 'logloss' or 'crossentropyloss':
        return nn.CrossEntropyLoss()
    elif name == 'bce' or 'bceloss' or 'binarycrossentropy':
        return nn.BCELoss()
    elif name == 'rmse' or 'rmseloss':
        return RMSELoss()
    else:
        raise NotImplementedError(f'Loss {name} not implemented')


def select_optimizer(name, model, lr):
    params = model.parameters()
    match name:
        case 'Adadelta':
            if lr in None:
                lr = 1.0
            return torch.optim.Adadelta(params=params, lr=lr)
        case 'Adagrad':
            if lr in None:
                lr = 0.01
            return torch.optim.Adagrad(params=params, lr=lr)
        case 'Adam':
            if lr in None:
                lr = 1e-3
            return torch.optim.Adam(params=params, lr=lr)
        case 'AdamW':
            if lr in None:
                lr = 1e-3
            return torch.optim.AdamW(params=params, lr=lr)
        case 'SparseAdam':
            if lr in None:
                lr = 1e-3
            return torch.optim.SparseAdam(params=params, lr=lr)
        case 'Adamax':
            if lr in None:
                lr = 2e-3
            return torch.optim.Adamax(params=params, lr=lr)
        case 'ASGD':
            if lr in None:
                lr = 0.01
            return torch.optim.ASGD(params=params, lr=lr)
        case 'LBFGS':
            if lr in None:
                lr = 1
            return torch.optim.LBFGS(params=params, lr=lr)
        case 'NAdam':
            if lr in None:
                lr = 2e-3
            return torch.optim.NAdam(params=params, lr=lr)
        case 'RAdam':
            if lr in None:
                lr = 1e-3
            return torch.optim.RAdam(params=params, lr=lr)
        case 'RMSProp':
            if lr in None:
                lr = 0.01
            return torch.optim.RMSprop(params=params, lr=lr)
        case 'Rprop':
            if lr in None:
                lr = 0.01
            return torch.optim.Rprop(params=params, lr=lr)
        case 'SGD':
            assert lr is not None, "Missing learning rate, define lr value"
            return torch.optim.SGD(params=params, lr=lr)
        case _:
            raise NotImplementedError(f'Optimizer {name} not implemented.\n'
                                      f'You can use follow optimizers: Adadelta, Adagrad, Adam, AdamW, SparseAdam,\n'
                                      f'Adamax, ASGD, LBFGS, NAdam, RAdam, RMSProp, Rprop, SGD')


def select_model(repo_or_dir, model_name, weights):
    if weights.endswith('.pt'):
        weights = weights.replace('.pt', '')
    if repo_or_dir == '':
        res = re.findall(r'(\w+?)(\d+)', model_name)[0]
        model_name, model_num = res[0], res[1]
        match model_name:
            case 'resnet':
                match model_num:
                    case '18':
                        return models.resnet18(weights=weights)
                    case '34':
                        return models.resnet34(weights=weights)
                    case '50':
                        return models.resnet50(weights=weights)
                    case '101':
                        return models.resnet101(weights=weights)
                    case '152':
                        return models.resnet152(weights=weights)
            case 'alexnet':
                return models.alexnet(weights=weights)
            case 'vgg':
                match model_num:
                    case '11':
                        return models.vgg11(weights=weights)
                    case '13':
                        return models.vgg13(weights=weights)
                    case '16':
                        return models.vgg16(weights=weights)
                    case '19':
                        return models.vgg19(weights=weights)
            case 'googlenet':
                return models.googlenet(weights=weights)
            case 'densenet':
                match model_num:
                    case '121':
                        return models.densenet121(weights=weights)
                    case '161':
                        return models.densenet161(weights=weights)
                    case '169':
                        return models.densenet169(weights=weights)
                    case '169':
                        return models.densenet201(weights=weights)
            case 'efficientnet_b':
                match model_num:
                    case '0':
                        return models.efficientnet_b0(weights=weights)
                    case '1':
                        return models.efficientnet_b1(weights=weights)
                    case '2':
                        return models.efficientnet_b2(weights=weights)
                    case '3':
                        return models.efficientnet_b3(weights=weights)
                    case '4':
                        return models.efficientnet_b4(weights=weights)
                    case '5':
                        return models.efficientnet_b5(weights=weights)
                    case '6':
                        return models.efficientnet_b6(weights=weights)
                    case '7':
                        return models.efficientnet_b7(weights=weights)
            case 'mobilenet_v':
                return models.mobilenet_v2(weights=weights)
            case 'inception_v':
                return models.inception_v3(weights=weights)
            case _:
                msg = f"Model {model_name + model_num} not implemented, " \
                      "to download any model from pytorch use repo_or_dir, model_name and  weights\n" \
                      "read about the available models and weights on https://pytorch.org/vision/stable/models.html\n" \
                      "you must specify a repository or directory such as 'pytorch/vision'"
                raise NotImplementedError(msg)
    else:
        if weights != '':
            return torch.hub.load(repo_or_dir=repo_or_dir, model=model_name, weights=weights)
        else:
            return torch.hub.load(repo_or_dir=repo_or_dir, model=model_name)
