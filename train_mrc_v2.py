import os
import random
import logging
from importlib import import_module

import numpy as np
import pandas as pd

import torch
import torch.nn as nn

import transformers
from transformers import AutoConfig, AutoTokenizer, AutoModelForQuestionAnswering, training_args
from transformers import TrainingArguments, HfArgumentParser
from transformers.trainer_utils import set_seed, EvalPrediction

import datasets
from datasets import load_metric

import wandb

from arguments import DefaultArguments, DatasetArguments, ModelArguments, RetrieverArguments
from processor import QAProcessor
from trainer_qa import QATrainer, FreezeEmbeddingCallback, FreezeBackboneCallback
from utils import increment_path


logger = transformers.logging.get_logger(__name__)


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


def get_model(model_args: ModelArguments, config=None):
    """Gets a custom model if `model_args.custom_model` is provided.
    Otherwise, returns the HuggingFace's implmenetation of the downstream model."""
    if model_args.custom_model is not None:
        print("Loading custom model class")
        custom_model_module = getattr(import_module("model.custom_models"), model_args.custom_model)
        model = custom_model_module.from_pretrained(model_args.model, config=config)
        return model
    else:
        print("Loadding HuggingFace's Implementation")
        model = AutoModelForQuestionAnswering.from_pretrained(model_args.model, config=config)
        return model


def main():
    parser = HfArgumentParser(
        (DefaultArguments, DatasetArguments, ModelArguments, RetrieverArguments, TrainingArguments)
    )
    default_args, dataset_args, model_args, retriever_args, training_args = parser.parse_args_into_dataclasses()

    # Increment paths to prevent overwriting
    training_args.output_dir = os.path.join(training_args.output_dir, training_args.run_name)
    training_args.output_dir = increment_path(training_args.output_dir, training_args.overwrite_output_dir)
    print(f"output_dir:  {training_args.output_dir}")

    training_args.logging_dir = os.path.join(training_args.logging_dir, training_args.run_name)
    training_args.logging_dir = increment_path(training_args.logging_dir, training_args.overwrite_output_dir)
    print(f"logging_dir: {training_args.logging_dir}")

    # Set seed before the model is initialized
    set_seed_all(training_args.seed)
    print(f"seed number: {training_args.seed}")

    config = AutoConfig.from_pretrained(model_args.model)
    tokenizer = AutoTokenizer.from_pretrained(model_args.model)
    model = get_model(model_args, config)

    print(f"config: {config}")
    print(f"tokenizer: {type(tokenizer)}")
    print(f"model: {type(model)}")

    # Fixing max_seq_len
    if dataset_args.max_seq_len != tokenizer.model_max_length:
        print(f"dataset_args.max_seq_len ({dataset_args.max_seq_len}) is different from tokenizer.model_max_length ({tokenizer.model_max_length}).")
        dataset_args.max_seq_len = min(dataset_args.max_seq_len, tokenizer.model_max_length)

    logger.debug("Default arguments %s", default_args)
    logger.debug("Dataset arguments %s", dataset_args)
    logger.debug("Model arguments %s", model_args)
    logger.debug("Retriever arguments %s", retriever_args)
    logger.debug("Training argumenets %s", training_args)

    qa_processor = QAProcessor(
        dataset_args=dataset_args, 
        tokenizer=tokenizer, 
        concat=(dataset_args.concat_eval)
    )

    train_examples = qa_processor.get_train_examples()
    eval_examples  = qa_processor.get_eval_examples()

    train_features = qa_processor.get_train_features()
    eval_features  = qa_processor.get_eval_features()

    # Define metric for squad dataset
    metric = load_metric("squad")
    def compute_metrics(pred: EvalPrediction):
        return metric.compute(predictions=pred.predictions, references=pred.label_ids)

    # Initialize wandb
    wandb.init(
        project=default_args.wandb_project, 
        entity=default_args.wandb_entity, 
        name=training_args.run_name,
        notes=default_args.description,
    )

    # Initialize callbacks
    # Comment at least one of the lines below out if you don't need to freeze the embedding layer
    # freeze_callback = FreezeEmbeddingCallback(model, freeze_epochs=1.0)
    freeze_callback = FreezeBackboneCallback(model, "roberta", freeze_epochs=1.0)

    # Build a custom trainer
    trainer = QATrainer(
        model=model,
        args=training_args,
        train_dataset=train_features,
        eval_dataset=eval_features,
        eval_examples=eval_examples,
        post_process_function=qa_processor.post_processing_function,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    # Comment this out if you don't need to freeze the layers
    trainer.add_callback(freeze_callback)

    # Train!
    trainer.train()

if __name__ == "__main__":
    main()