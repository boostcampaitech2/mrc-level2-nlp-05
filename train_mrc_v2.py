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
from transformers.trainer_utils import set_seed, get_last_checkpoint, EvalPrediction

import datasets
from datasets import load_metric

import wandb

from arguments import DefaultArguments, DatasetArguments, ModelArguments, RetrieverArguments
from processor import QAProcessor
from trainer_qa import QATrainer, FreezeEmbeddingCallback, FreezeBackboneCallback
from utils import increment_path, set_seed_all


logger = transformers.logging.get_logger(__name__)


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

    # Setup logging
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Detect checkpoint
    last_checkpoint = None

    # Train staring from the last checkpoint if resume_from_checkpoint is provided
    # otherwise, create a new path if the path given exists.
    training_args.output_dir  = os.path.join(training_args.output_dir, training_args.run_name)
    training_args.logging_dir = os.path.join(training_args.logging_dir, training_args.run_name)

    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            # checkpoint는 발견하지 못했는데 output_dir 내 폴더가 존재할 경우, 새로운 폴더를 생성합니다.
            training_args.output_dir  = increment_path(training_args.output_dir, training_args.overwrite_output_dir)
            training_args.logging_dir = increment_path(training_args.logging_dir, training_args.overwrite_output_dir)

        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            # 만약 주어진 output_dir 내에 checkpoint가 존재하는데, training arguments에 resume_from_checkpoint가 주어지지 않은 경우
            # 발견한 last_checkpoint에서 훈련을 진행합니다.
            # 만약, training arguments로 resume_from_checkpoint를 준다면 해당 checkpoint에서 진행됩니다.
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )
    print(f"output_dir:  {training_args.output_dir}")
    print(f"logging_dir: {training_args.logging_dir}")

    # Set seed before the model is initialized
    set_seed_all(training_args.seed)
    print(f"seed number: {training_args.seed}")

    config = AutoConfig.from_pretrained(model_args.config if model_args.config is not None else model_args.model)
    tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer if model_args.tokenizer is not None else model_args.model, use_fast=True)
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

    if model_args.freeze_type == "embeddings":
        freeze_callback = FreezeEmbeddingCallback(model, freeze_epochs=model_args.freeze_epoch)
    elif model_args.freeze_type is not None:
        freeze_callback = FreezeBackboneCallback(model, model_args.freeze_type, freeze_epochs=model_args.freeze_epoch)

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
    if model_args.freeze_type is not None:
        trainer.add_callback(freeze_callback)

    # Train!
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()
        metrics = train_result.metrics

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Eval!
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        metrics = trainer.evaluate()
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

if __name__ == "__main__":
    main()