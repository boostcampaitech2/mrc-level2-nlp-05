import os
import sys
import logging

from arguments import (
    DatasetArguments,
    ModelArguments, 
    RetrieverArguments,
)
from transformers import TrainingArguments
from simple_parsing import ArgumentParser

from model.models import BaseModel
from utils import increment_path
from transformers import set_seed

import torch

from transformers import AutoConfig, AutoTokenizer, AutoModelForQuestionAnswering
from datasets import load_from_disk, load_metric, Dataset, DatasetDict



def parse_args():

    parser = ArgumentParser()
    
    parser.add_arguments(DatasetArguments, dest="dataset")
    parser.add_arguments(ModelArguments, dest="model")
    parser.add_arguments(RetrieverArguments, dest="retriever")
    parser.add_arguments(TrainingArguments, dest="training")

    args = parser.parse_args()
    
    return args


logger = logging.getLogger(__name__)


def main():
    
    #########################
    ### Argument Parsing
    #########################

    args = parse_args()

    dataset_args: DatasetArguments   = args.dataset
    model_args: ModelArguments       = args.model
    rt_args: RetrieverArguments      = args.retriever
    training_args: TrainingArguments = args.training

    training_args.output_dir  = increment_path(
        training_args.output_dir, 
        training_args.overwrite_output_dir
    )
    training_args.logging_dir = increment_path(
        training_args.logging_dir, 
        training_args.overwrite_output_dir
    )

    print(f"output_dir : {training_args.output_dir}")
    print(f"logging_dir: {training_args.logging_dir}")

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -    %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    logger.info("Training/evaluation parameters %s", training_args)

    set_seed(training_args.seed)

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
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
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

    datasets = load_from_disk(dataset_args.dataset_name)

    print(type(datasets))

    

    
if __name__ == '__main__':
    main()