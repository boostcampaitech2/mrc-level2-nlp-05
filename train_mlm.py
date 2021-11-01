#!/usr/bin/env python
# coding=utf-8
# Copyright 2020 The HuggingFace Team All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for masked language modeling (BERT, ALBERT, RoBERTa...) on a text file or a dataset.
Here is the full list of checkpoints on the hub that can be fine-tuned by this script:
https://huggingface.co/models?filter=masked-lm
"""

import os
import math
import random
import logging
from importlib import import_module

import numpy as np
import pandas as pd

import torch
import torch.nn as nn

import transformers
from transformers import AutoConfig, AutoTokenizer, AutoModelForMaskedLM
from transformers import TrainingArguments, HfArgumentParser, DataCollatorForLanguageModeling
from transformers.trainer import Trainer
from transformers.trainer_utils import set_seed, get_last_checkpoint, EvalPrediction

import datasets
from datasets import load_metric

import wandb

from arguments import DefaultArguments, DatasetArguments, ModelArguments, RetrieverArguments
from build_wiki import get_wiki
from utils import increment_path, set_seed_all


logger = transformers.logging.get_logger(__name__)


def main():
    parser = HfArgumentParser(
        (DatasetArguments, ModelArguments, TrainingArguments)
    )
    dataset_args, model_args, training_args = parser.parse_args_into_dataclasses()

    # Setup logging
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

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

    config = AutoConfig.from_pretrained(model_args.model)
    tokenizer = AutoTokenizer.from_pretrained(model_args.model, use_fast=True)
    model = AutoModelForMaskedLM.from_pretrained("klue/roberta-large")
    print(type(config), type(tokenizer), type(model))

    # Fixing max_seq_len
    if dataset_args.max_seq_len != tokenizer.model_max_length:
        print(f"dataset_args.max_seq_len ({dataset_args.max_seq_len}) is different from tokenizer.model_max_length ({tokenizer.model_max_length}).")
        dataset_args.max_seq_len = min(dataset_args.max_seq_len, tokenizer.model_max_length)

    logger.debug("Dataset arguments %s", dataset_args)
    logger.debug("Model arguments %s", model_args)
    logger.debug("Training argumenets %s", training_args)

    wiki_dataset = get_wiki("/opt/ml/data/wikipedia_documents.json", "/opt/ml/data/wiki_data")
    print(wiki_dataset.features)
    # ['text', 'corpus_source', 'url', 'domain', 'title', 'author', 'html', 'document_id', '__index_level_0__']
    column_names = wiki_dataset.column_names
    
    raw_datasets = wiki_dataset.train_test_split(test_size=0.1, shuffle=True)
    text_column_name = "text"

    def tokenize_function(examples):
        return tokenizer(examples[text_column_name], return_special_tokens_mask=True)

    with training_args.main_process_first(desc="dataset map tokenization"):
        tokenized_datasets = raw_datasets.map(
            tokenize_function,
            batched=True,
            num_proc=4,
            remove_columns=column_names,
            desc="Running tokenizer on every text in dataset",
        )

    # Main data processing function that will concatenate all texts from our dataset 
    # and generate chunks of max_seq_length.
    def group_texts(examples):
        # Concatenate all texts.
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, we could add padding if the model supported it instead of this drop, 
        # you can customize this part to your needs.
        if total_length >= dataset_args.max_seq_len:
            total_length = (total_length // dataset_args.max_seq_len) * dataset_args.max_seq_len
        # Split by chunks of max_len.
        result = {
            k: [t[i : i + dataset_args.max_seq_len] for i in range(0, total_length, dataset_args.max_seq_len)]
            for k, t in concatenated_examples.items()
        }
        return result

    # Note that with `batched=True`, this map processes 1,000 texts together, 
    # so group_texts throws away a remainder for each of those groups of 1,000 texts. 
    # You can adjust that batch_size here but a higher value might be slower to preprocess.
    #
    # To speed up this part, we use multiprocessing. 
    # See the documentation of the map method for more information:
    # https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.map

    with training_args.main_process_first(desc="grouping texts together"):
        tokenized_datasets = tokenized_datasets.map(
            group_texts,
            batched=True,
            num_proc=4,
            desc=f"Grouping texts in chunks of {dataset_args.max_seq_len}",
        )

    if training_args.do_train:
        train_dataset = tokenized_datasets["train"]

    if training_args.do_eval:
        eval_dataset = tokenized_datasets["test"]

    MLM_PROB = 0.15
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm_probability=MLM_PROB,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    # Training
    if training_args.do_train:
        train_result = trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
        trainer.save_model()  # Saves the tokenizer too for easy upload
        metrics = train_result.metrics

        metrics["train_samples"] = len(train_dataset)

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        metrics = trainer.evaluate()

        metrics["eval_samples"] = len(eval_dataset)
        try:
            perplexity = math.exp(metrics["eval_loss"])
        except OverflowError:
            perplexity = float("inf")
        metrics["perplexity"] = perplexity

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

if __name__ == "__main__":
    main()