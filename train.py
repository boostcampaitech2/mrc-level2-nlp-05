import os
import sys
import logging
from typing import Optional

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


logger = logging.getLogger(__name__)


def main():

    #########################
    ### Argument Parsing
    #########################
    """You must set some of the huggnigface training_args...
    - output_dir (you can maintain the same output_dir... the output numbering will be automatically increased)
    - 
    """

    parser = HfArgumentParser(
        (DatasetArguments, ModelArguments, RetrieverArguments, TrainingArguments)
    )
    dataset_args, model_args, retriever_args, training_args = parser.parse_args_into_dataclasses()

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
        model_args.model,
        from_tf=bool(".ckpt" in model_args.model),
        config=config
    )

    print(
        type(training_args),
        type(model_args),
        type(tokenizer),
        type(model),
        end="\n"
    )

    #########################
    ### Load Dataset
    #########################

    datasets = load_from_disk(dataset_args.dataset_path)

    print(type(datasets))


if __name__ == "__main__":
    main()
