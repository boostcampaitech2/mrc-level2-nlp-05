from dataclasses import asdict, dataclass, field
from typing import Any, Union, Dict, List, Optional
from enum import Enum
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import optimizer

from transformers import TrainingArguments
from simple_parsing.helpers import Serializable


class Loss(Enum):
    CE  = "CrossEntropyLoss"
    BCE = "BCELoss"
    MSE = "MSELoss"
    L1  = "L1Loss"



@dataclass
class BaseArguments(Serializable):
    def __str__(self):
        self_as_dict = asdict(self)
        self_as_dict = {k: f"<{k.upper()}>" for k, v in self_as_dict.items()}

        attrs_as_str = [f"{k}={v},\n" for k, v in sorted(self_as_dict.items())]
        return f"{self.__class__.__name__}(\n{''.join(attrs_as_str)})"


@dataclass
class DatasetArguments(BaseArguments):
    """Dataset/DataLoader Arguments"""

    dataset_path: str = "/opt/ml/data/train_dataset/"
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

    overwrite_cache: bool = True
    """overwrite cache file if True"""

@dataclass
class ModelArguments(BaseArguments):
    """Model Arguements"""

    model: str = "klue/bert-base"
    """path to pretrained model or model identifier from huggingface.co/models"""

    config: str = None
    """pretrained config name or path if not the same as model_name"""

    tokenizer: str = None
    """pretrained tokenizer name or path if not the same as model_name"""

    head: str = None
    """output head"""

    loss_fn: Loss = Loss.CE
    """loss function used internally in the model"""


@dataclass
class RetrieverArguments(BaseArguments):
    """Retriever Arguments"""

    retriever_type: str = "SparseRetrieval"
    """name of retriever"""

    top_k_retrieval: int = 1
    """numb top-k passages to retrieve"""

    num_negs: int = 2
    """numb of negative samples for in-batch negatives"""

    dpr_model: str =  "bert-base-multilingual-cased"
    """path to pretrained model or model identifier from huggingface.co/models"""    

    # use_eval_retrieval: bool = True
    # """whether to run retrieval on eval"""

    # use_faiss: bool = False
    # """whether to build with faiss"""

    # num_clusters: int = 64
    # """num clusters to use for faiss"""
    dpr_learning_rate: float = 3e-5
    """learning rate for DPR fine-tuning"""

    dpr_train_batch: int = 2
    """train batch size for DPR fine-tuning"""

    dpr_eval_batch: int = 2
    """eval batch size for DPR fine-tuning"""

    dpr_epochs: int = 1
    """numb of epochs for DPR fine-tuning"""

    dpr_weight_decay: float = 0.01
    """weight decay for DPR fine-tuning"""