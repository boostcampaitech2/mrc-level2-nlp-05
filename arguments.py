from dataclasses import asdict, dataclass, field
from typing import Any, Union, Dict, List, Optional
from enum import Enum
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import optimizer

class Loss(Enum):
    CrossEntropyLoss = nn.CrossEntropyLoss()
    MSELoss = nn.MSELoss()
    BCELoss = nn.BCELoss()
    FocalLoss = None

class Optimizer(Enum):
    Adam  = optim.Adam
    AdamW = optim.AdamW
    SGD   = optim.SGD

@dataclass
class ModelArguments:
    """
    Model Arguements
    """
    model_name: Union[str, os.PathLike] = field(
        default="klue/bert-base",
        metadata={
            "help": "Path to pretrained model or model identifier from huggingface.co/models"
        },
    )

    num_labels: int = field(
        default=1,
        metadata={
            "help": "number of labels. if 1, BCELoss / if > 2, CELoss"
        }
    )

    loss_fn: Loss = field(
        default=Loss.CrossEntropyLoss,
        metadata={
            "help": "Loss function used internally in the model"
        }
    )

@dataclass
class DatasetArguments:
    """
    Dataset Arguments
    """
    dataset_path: Union[str, os.PathLike] = field(
        default="/opt/ml/data/train_dataset",
        metadata={
            "help": "Path to dataset"
        },
    )

@dataclass
class TrainerArguments:
    """
    Trainer Arguments
    """
    lr: float = field(
        default=3e-5,
        metadata={
            "help": "learning rate"
        }
    )

    epochs: int = field(
        default=1,
        metadata={
            "help": "num epochs in float"
        }
    )

    optimizer: Optimizer = field(
        default=Optimizer.Adam,
        metadata={
            "help": "optimizer type"
        }
    )

    print_every: int = field(
        default=500, 
        metadata={
            "help": "print every steps"
        }
    )