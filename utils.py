import os
import re
import glob
import random
from pathlib import Path
import shutil

import numpy as np

import torch

from transformers.trainer_utils import set_seed


class LossObject:
    """
    loss 값을 관리하는 Object Class
    """
    def __init__(self):
        self.loss = 0
        self.example_cnt = 0
    
    def update(self, new_loss, new_example_cnt):
        self.loss += new_loss
        self.example_cnt += new_example_cnt

    def get_avg_loss(self):
        return self.loss / self.example_cnt

    def reset(self):
        self.loss = 0
        self.example_cnt = 0

class SaveLimitObject:
    """저장되는 모델을 관리하는 Object Class"""
    def __init__(self, save_total_limit):
        self.save_total_limit = save_total_limit
        self.paths = []

    def update(self, new_path):
        if len(self.paths) >= self.save_total_limit:
            shutil.rmtree(self.paths.pop(0))            
        self.paths.append(new_path)


def increment_path(path, overwrite=False) -> str:
    """ Automatically increment path, i.e. runs/exp --> runs/exp0, runs/exp1 etc.
    Args:
        path (str or pathlib.Path): f"{model_dir}/{args.name}".
        overwrite (bool): whether to overwrite or increment path (increment if False).

    Returns:
        path: new path
    """
    path = Path(path)

    if os.path.isabs(path):
        path = Path(os.path.relpath(path))

    if (path.exists() and overwrite) or (not path.exists()):
        if not os.path.exists(str(path).split('/')[0]):
            os.mkdir(str(path).split('/')[0])
        if not path.exists():
            os.mkdir(path)
        return str(path)

    else:
        dirs = glob.glob(f"{path}*")
        matches = [re.search(rf"%s(\d+)" % path.stem, d) for d in dirs]
        i = [int(m.groups()[0]) for m in matches if m]
        n = max(i) + 1 if i else 2
        path = f"{path}{n}"
        if not os.path.exists(path):
            os.mkdir(path)
        return path

def set_seed_all(seed):
    """Fix the seed number for all"""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    set_seed(seed)