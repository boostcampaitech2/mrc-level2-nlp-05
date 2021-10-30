import os
import sys
import logging

from importlib import import_module

from transformers.trainer_utils import EvalPrediction, set_seed

from arguments import (
    DefaultArguments,
    DatasetArguments,
    ModelArguments,
    RetrieverArguments,
)

from transformers import TrainingArguments, HfArgumentParser
from simple_parsing import ArgumentParser

from model.models import BaseModel
from utils import increment_path

import torch

import wandb

from transformers import AutoConfig, AutoTokenizer, AutoModelForQuestionAnswering
from datasets import load_from_disk, load_metric, Dataset, DatasetDict

from utils_qa import check_no_error
from trainer_qa import QATrainer
from retrieval import SparseRetrieval

from processor import QAProcessor
# from preprocessor import BaselinePreprocessor, Preprocessor
# from postprocessor import post_processing_function

logger = logging.getLogger(__name__)

metric = load_metric("squad")

def compute_metrics(pred: EvalPrediction):
    return metric.compute(predictions=pred.predictions, references=pred.label_ids)


def update_save_path(training_args: TrainingArguments):
    """모델 및 로그 저장 경로 생성 함수"""
    training_args.output_dir = os.path.join(training_args.output_dir, training_args.run_name)
    training_args.output_dir = increment_path(
        training_args.output_dir, training_args.overwrite_output_dir
    )
    training_args.logging_dir = increment_path(
        training_args.logging_dir, training_args.overwrite_output_dir
    )
    print(f"output_dir : {training_args.output_dir}")


def set_logging(default_args, dataset_args, model_args, retriever_args, training_args):
    """logging setting 함수"""
    logging_level_dict = {
        "DEBUG": logging.DEBUG,         # 문제를 해결할 때 필요한 자세한 정보
        "INFO": logging.INFO,           # 작업이 정상적으로 작동하고 있다는 확인 메시지
        "WARNING": logging.WARNING,     # 예상하지 못한 일이 발생 or 발생 가능한 문제점을 명시
        "ERROR": logging.ERROR,         # 프로그램이 함수를 실행하지 못 할 정도의 심각한 문제
        "CRITICAL": logging.CRITICAL    # 프로그램이 동작할 수 없을 정도의 심각한 문제
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


def main():

    #########################
    ### Argument Parsing
    #########################

    parser = HfArgumentParser(
        (DefaultArguments, DatasetArguments, ModelArguments, RetrieverArguments, TrainingArguments)
    )
    (
        default_args,
        dataset_args,
        model_args,
        retriever_args,
        training_args,
    ) = parser.parse_args_into_dataclasses()

    set_seed(training_args.seed)
    # update_save_path(training_args)

    print(f"output_dir : {training_args.output_dir}")
    print(f"logging_dir: {training_args.logging_dir}")

    #########################
    ### Set wandb
    #########################

    wandb.login()
    wandb.init(
        project=default_args.wandb_project,
        entity=default_args.wandb_entity,
        name=training_args.run_name
    )
    
    wandb.config.update(default_args)
    wandb.config.update(dataset_args)
    wandb.config.update(model_args)
    wandb.config.update(retriever_args)
    wandb.config.update(training_args)  

    #########################
    ### Load Config, Tokenizer, Model
    #########################

    config = AutoConfig.from_pretrained(
        model_args.config if model_args.config is not None else model_args.model
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer if model_args.tokenizer is not None else model_args.model
    )
    
    # TODO: Load custom model here...
    model = AutoModelForQuestionAnswering.from_pretrained(
        model_args.model, from_tf=bool(".ckpt" in model_args.model), config=config
    )

    print(type(training_args), type(model_args), type(tokenizer), type(model), end="\n")

    #########################
    ### Load & Preprocess Dataset
    #########################

    qa_processor = QAProcessor(dataset_args, tokenizer, concat=False)

    datasets = qa_processor.get_datasets()
    print(type(datasets))

    train_dataset = qa_processor.get_train_features()
    eval_dataset  = qa_processor.get_eval_features()
    eval_examples = qa_processor.get_eval_examples()

    trainer = QATrainer(
        model=model,
        args=training_args,
        # dataset_args=dataset_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        eval_examples=eval_examples if training_args.do_eval else None,
        tokenizer=tokenizer,
        # datasets=datasets,
        post_process_function=qa_processor.post_processing_function,
        compute_metrics=compute_metrics,
    )

    trainer.train()

if __name__ == "__main__":
    main()
