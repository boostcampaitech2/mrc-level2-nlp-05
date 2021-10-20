import os
import sys
import logging

from importlib import import_module

from transformers.trainer_utils import EvalPrediction
from transformers.utils.dummy_pt_objects import DataCollatorWithPadding

from arguments import (
    DatasetArguments,
    ModelArguments,
    RetrieverArguments,
)

from transformers import TrainingArguments, HfArgumentParser
from simple_parsing import ArgumentParser

from model.models import BaseModel
from utils import increment_path

import torch

from transformers import AutoConfig, AutoTokenizer, AutoModelForQuestionAnswering
from datasets import load_from_disk, load_metric, Dataset, DatasetDict

from utils_qa import check_no_error
from trainer_qa import QuestionAnsweringTrainer
from retrieval import SparseRetrieval

from preprocessor import BaselinePreprocessor, Preprocessor
from postprocessor import post_processing_function

logger = logging.getLogger(__name__)

metric = load_metric("squad")

def compute_metrics(pred: EvalPrediction):
    return metric.compute(predictions=pred.predictions, references=pred.label_ids)


def main():

    #########################
    ### Argument Parsing
    #########################

    parser = HfArgumentParser(
        (DatasetArguments, ModelArguments, RetrieverArguments, TrainingArguments)
    )
    (
        dataset_args,
        model_args,
        retriever_args,
        training_args,
    ) = parser.parse_args_into_dataclasses()

    # dataset_args = DatasetArguments(dataset_args)
    # model_args = ModelArguments(model_args)
    # retriever_args = RetrieverArguments(retriever_args)
    # training_args = TrainingArguments(training_args)

    training_args.output_dir = increment_path(
        training_args.output_dir, training_args.overwrite_output_dir
    )

    training_args.logging_dir = increment_path(
        training_args.logging_dir, training_args.overwrite_output_dir
    )

    print(f"output_dir : {training_args.output_dir}")
    print(f"logging_dir: {training_args.logging_dir}")

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -    %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    logger.debug("Dataset arguments %s", dataset_args)
    logger.debug("Model arguments %s", model_args)
    logger.debug("Retriever arguments %s", retriever_args)
    logger.info("Training argumenets %s", training_args)

    #########################
    ### Load Config, Tokenizer, Model
    #########################

    config = AutoConfig.from_pretrained(
        model_args.config if model_args.config is not None else model_args.model
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer if model_args.tokenizer is not None else model_args.model
    )

    # TODO: load custom model here
    model = AutoModelForQuestionAnswering.from_pretrained(
        model_args.model, from_tf=bool(".ckpt" in model_args.model), config=config
    )

    print(type(training_args), type(model_args), type(tokenizer), type(model), end="\n")

    #########################
    ### Load & Preprocess Dataset
    #########################

    datasets = load_from_disk(dataset_args.dataset_path)

    print(type(datasets))

    if training_args.do_train:
        column_names = datasets["train"].column_names
    else:
        column_names = datasets["validation"].column_names

    # 오류가 있는지 확인합니다.
    last_checkpoint, max_seq_length = check_no_error(
        dataset_args, training_args, datasets, tokenizer
    )

    # TODO: Argparse
    preprocessor = BaselinePreprocessor(
        dataset_args=dataset_args, tokenizer=tokenizer, column_names=column_names
    )

    if training_args.do_train:
        if "train" not in datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = datasets["train"]
        train_dataset = train_dataset.map(
            preprocessor.prepare_train_features,
            batched=True,
            num_proc=dataset_args.num_workers,
            remove_columns=column_names,
            load_from_cache_file=not dataset_args.overwrite_cache,
        )

    if training_args.do_eval:
        if "validation" not in datasets:
            # If validation set does not exist,
            # then evaluate the model with train set
            eval_dataset = datasets["train"]
        else:
            eval_dataset = datasets["validation"]

        eval_dataset = eval_dataset.map(
            preprocessor.prepare_eval_features,
            batched=True,
            num_proc=dataset_args.num_workers,
            remove_columns=column_names,
            load_from_cache_file=not dataset_args.overwrite_cache,
        )

    data_collator = DataCollatorWithPadding(tokenizer)

    trainer = QuestionAnsweringTrainer(
        model=model,
        args=training_args,
        dataset_args=dataset_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        eval_examples=datasets["validation"] if training_args.do_eval else None,
        tokenizer=tokenizer,
        datasets=datasets,
        # data_collator=data_collator,
        post_process_function=post_processing_function,
        compute_metrics=compute_metrics,
    )

    trainer.train()

if __name__ == "__main__":
    main()
