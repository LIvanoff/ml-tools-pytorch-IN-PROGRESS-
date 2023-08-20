import argparse
import os
from pathlib import Path
import random
import sys

import numpy as np
import torch
from torch.optim import lr_scheduler
from tqdm import tqdm

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))


def train(hyp, opt, device, callbacks):  # hyp is path/to/hyp.yaml or hyp dictionary
    save_dir, epochs, batch_size, weights, single_cls, evolve, data, cfg, resume, noval, nosave, workers, freeze = \
        Path(opt.save_dir), opt.epochs, opt.batch_size, opt.weights, opt.single_cls, opt.evolve, opt.data, opt.cfg, \
            opt.resume, opt.noval, opt.nosave, opt.workers, opt.freeze

    torch.cuda.empty_cache()
    return


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('model', type=str,  help='model path')
    parser.add_argument('data-train', type=str, help='train dataset path')
    parser.add_argument('--data-val', type=str, help='val dataset path')
    parser.add_argument('--test-size', type=float, default=0.2, help='percentage of the test dataset from the '
                                                                     'training sample')
    parser.add_argument('--plot-loss', action='store_true', help='output of the loss plot')
    parser.add_argument('--weights', type=str, help='weight path')
    parser.add_argument('--batch-size', type=int, default=64, help='batch size')
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=224, help='inference size (pixels)')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--workers', type=int, default=8, help='max dataloader workers (per RANK in DDP mode)')
    parser.add_argument('--augment', action='store_false', help='augmented inference')
    parser.add_argument('--save-xlsx', action='store_true', help='save results to *.xlsx')
    parser.add_argument('--save-csv', action='store_false', help='save results to *.csv')
    parser.add_argument('--freeze', type=int, help='number of frozen layers')
    parser.add_argument('--project', default=ROOT / 'runs/train', help='save to project/name')
    parser.add_argument('--name', default='exp', help='save to project/name')
    opt = parser.parse_args()
    return opt


def main(opt):
    pass


if __name__ == '__main__':
    opt = parse_opt()
    main(opt)
