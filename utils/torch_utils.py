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


def __select_model(repo_or_dir, model_name, weights, pretrained):
    if repo_or_dir == '':
        res = re.findall(r'(\w+?)(\d+)',model_name)[0]
        model_name, model_num = res[0], res[1]
        match model_name:
            case 'resnet':
                match model_num:
                    case '18':
                        return models.resnet18(pretrained=pretrained)
                    case '34':
                        return models.resnet34(pretrained=pretrained)
                    case '50':
                        return models.resnet50(pretrained=pretrained)
                    case '101':
                        return models.resnet101(pretrained=pretrained)
                    case '152':
                        return models.resnet152(pretrained=pretrained)
            case 'alexnet':
                return models.alexnet(pretrained=pretrained)
            case 'vgg':
                match model_num:
                    case '11':
                        return models.vgg11(pretrained=pretrained)
                    case '13':
                        return models.vgg13(pretrained=pretrained)
                    case '16':
                        return models.vgg16(pretrained=pretrained)
                    case '19':
                        return models.vgg19(pretrained=pretrained)
            case 'googlenet':
                return models.googlenet(pretrained=pretrained)
            case 'densenet':
                match model_num:
                    case '121':
                        return models.densenet121(pretrained=pretrained)
                    case '161':
                        return models.densenet161(pretrained=pretrained)
                    case '169':
                        return models.densenet169(pretrained=pretrained)
                    case '169':
                        return models.densenet201(pretrained=pretrained)
            case 'efficientnet_b':
                match model_num:
                    case '0':
                        return models.efficientnet_b0(pretrained=pretrained)
                    case '1':
                        return models.efficientnet_b1(pretrained=pretrained)
                    case '2':
                        return models.efficientnet_b2(pretrained=pretrained)
                    case '3':
                        return models.efficientnet_b3(pretrained=pretrained)
                    case '4':
                        return models.efficientnet_b4(pretrained=pretrained)
                    case '5':
                        return models.efficientnet_b5(pretrained=pretrained)
                    case '6':
                        return models.efficientnet_b6(pretrained=pretrained)
                    case '7':
                        return models.efficientnet_b7(pretrained=pretrained)
            case 'mobilenet_v':
                return models.mobilenet_v2(pretrained=pretrained)
            case 'inception_v':
                return models.inception_v3(pretrained=pretrained)
            case _:
                    msg = f"NameError: {model_name + model_num} is unknown model, " \
                          "to download any model from pytorch, use repo_or_dir, model_name and  weights"
                    logging.warning(msg)
                    sys.exit(1)
    else:
        if weights != '':
            if weights.endswith('.pt'):
                weights = weights.lower().replace('.pt', '')
            return torch.hub.load(repo_or_dir=repo_or_dir, model=model_name, pretrained=pretrained, weights=weights)
        else:
            return torch.hub.load(repo_or_dir=repo_or_dir, model=model_name, pretrained=pretrained)



