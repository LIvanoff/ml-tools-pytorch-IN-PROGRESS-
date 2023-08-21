import re
import logging
import sys

import torch
import torch.nn as nn
from torchvision import models


def __select_loss():
    pass


def __select_optimizer():
    pass


def __select_model(repo_or_dir, model_name, weights):
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
                    msg = f"NameError: {model_name + model_num} is unknown model, " \
                          "to download any model from pytorch, use repo_or_dir, model_name and  weights"
                    logging.warning(msg)
                    sys.exit(1)
    else:
        if weights != '':
            if weights.endswith('.pt'):
                weights = weights.lower().replace('.pt', '')
            return torch.hub.load(repo_or_dir=repo_or_dir, model=model_name, weights=weights)
        else:
            return torch.hub.load(repo_or_dir=repo_or_dir, model=model_name)
