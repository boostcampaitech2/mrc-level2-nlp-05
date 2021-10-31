import os
import sys
import logging
import random
import numpy as np
from importlib import import_module

from arguments import (
    DefaultArguments,
    DatasetArguments,
    ModelArguments,
    RetrieverArguments,
)

import torch

from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForQuestionAnswering,
    TrainingArguments,
    HfArgumentParser
)
from transformers.trainer_utils import set_seed, EvalPrediction

from datasets import load_from_disk, load_metric
from utils import increment_path
from processor import QAProcessor
from trainer_qa import QATrainer

logger = logging.getLogger(__name__)

metric = load_metric("squad")

def compute_metrics(pred: EvalPrediction):
    return metric.compute(predictions=pred.predictions, references=pred.label_ids)

def get_args():
    """argument 객체 생성 함수"""
    parser = HfArgumentParser(
        (DefaultArguments, DatasetArguments, ModelArguments, RetrieverArguments, TrainingArguments)
    )
    default_args, dataset_args, model_args, retriever_args, training_args = parser.parse_args_into_dataclasses()
    return default_args, dataset_args, model_args, retriever_args, training_args

def set_logging(default_args, dataset_args, model_args, retriever_args, training_args):
    """logging setting 함수"""
    logging_level_dict = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL
    }
    logging.basicConfig(
        format="%(asctime)s - %(module)s - %(levelname)s  %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
        level=logging_level_dict[default_args.log_level]
    )
    logger.debug("Default arguments %s", default_args)
    logger.debug("Dataset arguments %s", dataset_args)
    logger.debug("Model arguments %s", model_args)
    logger.debug("Retriever arguments %s", retriever_args)
    logger.debug("Training argumenets %s", training_args)

def update_save_path(training_args):
    """모델 및 로그 저장 경로 생성 함수"""
    training_args.output_dir = os.path.join(training_args.output_dir, training_args.run_name)
    training_args.output_dir = increment_path(
        training_args.output_dir, training_args.overwrite_output_dir
    )
    training_args.logging_dir = increment_path(
        training_args.logging_dir, training_args.overwrite_output_dir
    )
    print(f"output_dir : {training_args.output_dir}")

def set_seed_everything(seed):
    """seed 고정 함수"""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    set_seed(seed)

def get_model_and_tokenizer(model_args):
    """모델 및 토크나이저 생성 함수"""
    config = AutoConfig.from_pretrained(
        model_args.config if model_args.config is not None else model_args.model
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer if model_args.tokenizer is not None else model_args.model
    )
    if model_args.custom_model is None:
        model = AutoModelForQuestionAnswering.from_pretrained(
            model_args.model, from_tf=bool(".ckpt" in model_args.model), config=config
        )
    else:
        config.dropout_ratio = model_args.head_dropout_ratio
        custom_model_module = getattr(import_module('model.custom_models'), model_args.custom_model)
        model = custom_model_module(config).from_pretrained(
            model_args.model, from_tf=bool(".ckpt" in model_args.model), config=config
        )
    return config, tokenizer, model

def get_processor(dataset_args, tokenizer):
    return QAProcessor(
        dataset_args=dataset_args,
        tokenizer=tokenizer,
        concat=dataset_args.concat_eval
    )

def set_wandb(default_args):
    os.environ['WANDB_ENTITY'] = default_args.wandb_entity
    os.environ['WANDB_PROJECT'] = default_args.wandb_project

def train_mrc(model, processor, training_args):
    trainer = QATrainer(
        model=model,
        args=training_args,
        train_dataset=processor.get_train_features(),
        eval_dataset=processor.get_eval_features(),
        eval_examples=processor.get_eval_examples(),
        post_process_function=processor.post_processing_function,
        compute_metrics=compute_metrics
    )
    trainer.train()

def main():
    default_args, dataset_args, model_args, retriever_args, training_args = get_args()
    set_logging(default_args, dataset_args, model_args, retriever_args, training_args)
    update_save_path(training_args)
    set_seed_everything(training_args.seed)

    config, tokenizer, model = get_model_and_tokenizer(model_args)
    processor = get_processor(dataset_args, tokenizer)

    set_wandb(default_args)

    train_mrc(model, processor, training_args)

if __name__ == "__main__":
    main()