import logging
import os
import sys
from typing import Callable, List, Dict, NoReturn, Tuple
from importlib import import_module

import numpy as np

import torch
import torch.nn as np

import transformers
from transformers import AutoConfig, AutoModelForQuestionAnswering, AutoTokenizer
from transformers import EvalPrediction, HfArgumentParser, TrainingArguments, set_seed

import datasets
from datasets import load_metric, load_from_disk, Sequence, Value, Features, Dataset, DatasetDict

from retrieval import SparseRetrieval
from retrieval_TFIDF import SparseRetrieval_TFIDF

from arguments import ModelArguments, DatasetArguments, RetrieverArguments
from processor import QAProcessor
from trainer_qa import QATrainer

import pandas as pd

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


def run_sparse_retrieval(
    tokenize_fn: Callable[[str], List[str]],
    examples: Dataset,
    training_args: TrainingArguments,
    dataset_args: DatasetArguments,
    retriever_args: RetrieverArguments,
    data_path: str = "/opt/ml/data/",
    context_path: str = "wikipedia_documents.json",
) -> Dataset:
    
    if retriever_args.retriever_type == "SparseRetrieval":
        retriever = SparseRetrieval(tokenize_fn, data_path, context_path)
        retriever.get_sparse_embedding_bm25()
        df = retriever.retrieve(examples, topk=retriever_args.top_k_retrieval)

    elif retriever_args.retriever_type  == 'TFIDF':
        retriever = SparseRetrieval_TFIDF(tokenize_fn = tokenize_fn, data_path=data_path, context_path=context_path)
        retriever.get_sparse_embedding()
        df = retriever.retrieve(datasets["validation"], topk=retriever_args.top_k_retrieval)

    
    if training_args.do_predict:
        f = Features(
            {
                "context": Value(dtype="string", id=None),
                "id": Value(dtype="string", id=None),
                "question": Value(dtype="string", id=None),
            }
        )

    elif training_args.do_eval:
        f = Features(
            {
                "answers": Sequence(
                    feature={
                        "text": Value(dtype="string", id=None),
                        "answer_start": Value(dtype="int32", id=None),
                    },
                    length=-1,
                    id=None,
                ),
                "context": Value(dtype="string", id=None),
                "id": Value(dtype="string", id=None),
                "question": Value(dtype="string", id=None),
            }
        )
    if retriever_args.retriever_type == "SparseRetrieval" or retriever_args.retriever_type == "TFIDF":
        dataset = Dataset.from_pandas(df, features=f)

    elif retriever_args.retriever_type == "ElasticSearch":
        df = pd.read_csv('ES_contest_main.csv')
        dataset = Dataset.from_pandas(df, features=f)

    return dataset


def main():
    parser = HfArgumentParser(
        (DatasetArguments, ModelArguments, RetrieverArguments, TrainingArguments)
    )
    dataset_args, model_args, retriever_args, training_args = parser.parse_args_into_dataclasses()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    set_seed(training_args.seed)

    config = AutoConfig.from_pretrained(model_args.model)
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer if model_args.tokenizer is not None else model_args.model,
        use_fast=True
    )
    model = get_model(model_args, config=config)

    # Fixing max_seq_len
    if dataset_args.max_seq_len != tokenizer.model_max_length:
        print(f"dataset_args.max_seq_len ({dataset_args.max_seq_len}) is different from tokenizer.model_max_length ({tokenizer.model_max_length}).")
        dataset_args.max_seq_len = min(dataset_args.max_seq_len, tokenizer.model_max_length)

    logger.debug("Dataset arguments %s", dataset_args)
    logger.debug("Model arguments %s", model_args)
    logger.debug("Retriever arguments %s", retriever_args)
    logger.debug("Training argumenets %s", training_args)

    processor = QAProcessor(dataset_args, tokenizer)

    test_examples = processor.get_test_examples()

    if retriever_args.use_eval_retrieval:
        test_examples = run_sparse_retrieval(
            tokenizer.tokenize,
            test_examples,
            training_args,
            dataset_args,
            retriever_args
        )
    
    column_names = test_examples.column_names
    question_column = "question" if "question" in column_names else column_names[0]
    context_column  = "context"  if "context"  in column_names else column_names[1]
    answer_column   = "answers"  if "answers"  in column_names else column_names[2]
    processor.set_column_names([question_column, context_column, answer_column])

    test_features = processor.get_test_features(dataset=test_examples)

    logger.info(f"test examples: {type(test_examples)} of length{len(test_examples)}")
    logger.info(f"test features: {type(test_features)} of length{len(test_features)}")

    metric = load_metric("squad")
    def compute_metrics(p: EvalPrediction) -> Dict:
        return metric.compute(predictions=p.predictions, references=p.label_ids)

    print("Init trainer...")
    trainer = QATrainer(
        model=model,
        args=training_args,
        train_dataset=None,
        eval_dataset=test_features,
        eval_examples=test_examples,
        tokenizer=tokenizer,
        post_process_function=processor.post_processing_function,
        compute_metrics=compute_metrics,
    )

    logger.info("*** Evaluate ***")
    
    if training_args.do_predict:
        predictions = trainer.predict(test_dataset=test_features, test_examples=test_examples)
        print("No metric can be presented because there is no correct answer given. Job done!")

    if training_args.do_eval:
        metrics = trainer.evaluate()
        metrics["eval_samples"] = len(test_examples)

        trainer.log_metrics("test", metrics)
        trainer.save_metrics("test", metrics)


if __name__ == "__main__":
    main()
