import os
import re
import glob
from pathlib import Path

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