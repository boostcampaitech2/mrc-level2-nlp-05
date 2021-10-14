from dataclasses import asdict, dataclass, field
from typing import Any, Union, Dict, List, Optional
from enum import Enum
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import optimizer

from simple_parsing.helpers import Serializable


class Loss(Enum):
    CrossEntropyLoss = nn.CrossEntropyLoss()
    MSELoss = nn.MSELoss()
    BCELoss = nn.BCELoss()
    FocalLoss = None


class Optimizer(Enum):
    Adam  = optim.Adam
    AdamW = optim.AdamW
    SGD   = optim.SGD


class LRScheduler(Enum):
    constant = "constant"
    linear   = "linear"


@dataclass
class ModelArguments(Serializable):
    """Model Arguements"""

    model: Union[str, os.PathLike] = "klue/bert-base"
    """path to pretrained model or model identifier from huggingface.co/models"""

    config: Optional[Union[str, os.PathLike]] = None
    """pretrained config name or path if not the same as model_name"""

    tokenizer: Optional[Union[str, os.PathLike]] = None
    """pretrained tokenizer name or path if not the same as model_name"""

    head: Optional[Union[str]] = None
    """output head"""

    loss_fn: Loss = Loss.CrossEntropyLoss
    """loss function used internally in the model"""

@dataclass
class DatasetArguments(Serializable):
    """Dataset/DataLoader Arguments"""

    dataset_path: os.PathLike = "/opt/ml/data/train_dataset"
    """path for the dataset"""

    max_seq_len: int = 384
    """maximum total input sequence length after tokenization"""

    stride_len: int = 128
    """stride value when splitting up a long document into chunks"""

    max_ans_len: int = 30
    """maximum length of an answer that can be generated"""

    use_max_padding: bool = False
    """Whether to pad all samples to `max_seq_length`. 
    Run slow if set to False on TPUs"""

    use_bucketing: bool = False
    """Whether to use bucketing"""

    num_workers: int = 1
    """num workers for preprocessing"""


class RetrieverArguments(Serializable):
    """Retriever Arguments"""

    retriver: str = "tf-idf"
    """name of retriever"""

    top_k_retrieval: int = 1
    """numb top-k passages to retrieve"""

    use_eval_retrieval: bool = True
    """whether to run retrieval on eval"""

    use_faiss: bool = False
    """whether to build with faiss"""

    num_clusters: int = 64
    """num clusters to use for faiss"""