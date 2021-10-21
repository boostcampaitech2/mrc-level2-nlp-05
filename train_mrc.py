import os
import sys
import logging
from tqdm import tqdm
import wandb

from importlib import import_module

from transformers.trainer_utils import EvalPrediction
from transformers import DataCollatorWithPadding

from arguments import (
    DatasetArguments,
    ModelArguments,
    RetrieverArguments,
)

from transformers import TrainingArguments, HfArgumentParser


from model.models import BaseModel
from utils import increment_path

import torch
from torch.utils.data import DataLoader

from transformers import AutoConfig, AutoTokenizer, AutoModelForQuestionAnswering, AdamW
from datasets import load_from_disk, load_metric, Dataset, DatasetDict

from retrieval import SparseRetrieval

from preprocessor import BaselinePreprocessor, Preprocessor
from postprocessor import post_processing_function

logger = logging.getLogger(__name__)

metric = load_metric("squad")

def get_args():
    """argument 반환 함수"""
    parser = HfArgumentParser(
        (DatasetArguments, ModelArguments, RetrieverArguments, TrainingArguments)
    )
    (
        dataset_args,
        model_args,
        retriever_args,
        training_args,
    ) = parser.parse_args_into_dataclasses()
    return dataset_args, model_args, retriever_args, training_args

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
    print(f"logging_dir: {training_args.logging_dir}")

def set_logging(dataset_args, model_args, retriever_args, training_args):
    """logging 정보 setting 함수"""
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -    %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    logger.debug("Dataset arguments %s", dataset_args)
    logger.debug("Model arguments %s", model_args)
    logger.debug("Retriever arguments %s", retriever_args)
    logger.info("Training argumenets %s", training_args)

def get_model(model_args, training_args):
    """model, tokenizer, optimizer 반환 함수"""
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

    optimizer = AdamW(model.parameters(), lr=training_args.learning_rate)
    return model, tokenizer, optimizer

def get_data_loaders(dataset_args, training_args, tokenizer):
    datasets = load_from_disk(dataset_args.dataset_path)
    train_dataset = datasets['train']
    eval_dataset = datasets['validation']
    column_names = train_dataset.column_names
    #eval_column_names = eval_dataset.column_names

    # 오류가 있는지 확인합니다.
    # last_checkpoint, max_seq_length = check_no_error(
    #     dataset_args, training_args, datasets, tokenizer
    # )

    # TODO: Argparse
    preprocessor = BaselinePreprocessor(
        dataset_args=dataset_args, tokenizer=tokenizer, column_names=column_names
    )

    train_dataset = train_dataset.map(
        preprocessor.prepare_train_features,
        batched=True,
        num_proc=dataset_args.num_workers,
        remove_columns=column_names,
        load_from_cache_file=not dataset_args.overwrite_cache,
    )
    eval_dataset = eval_dataset.map(
        preprocessor.prepare_eval_features,
        batched=True,
        num_proc=dataset_args.num_workers,
        remove_columns=column_names,
        load_from_cache_file=not dataset_args.overwrite_cache,
    )

    data_collator = DataCollatorWithPadding(
        tokenizer,
        pad_to_multiple_of=8 if training_args.fp16 else None
    )
    train_dataloader = DataLoader(
        train_dataset,
        collate_fn = data_collator,
        batch_size=training_args.per_device_train_batch_size
    )
    eval_dataloader = DataLoader(
        eval_dataset,
        collate_fn = data_collator,
        batch_size=training_args.per_device_eval_batch_size
    )

    return train_dataloader, eval_dataloader

def compute_metrics(pred: EvalPrediction):
    return metric.compute(predictions=pred.predictions, references=pred.label_ids)

def train_mrc(
    dataset_args, model_args, retriever_args, training_args,
    device,
    model, optimizer, tokenizer,
    train_dataloader, eval_dataloader
):
    """train 함수"""
    best_em = 0
    best_f1 = 0
    global_steps = 0
    train_loss = 0
    acc_batches = 0
    for epoch in range(int(training_args.num_train_epochs)):
        pbar = tqdm(
            enumerate(train_dataloader),
            total=len(train_dataloader),
            position=0,
            leave=True
        )
        for step, batch in pbar:
            # train
            model.train()

            batch = batch.to(device)
            outputs = model(**batch)

            optimizer.zero_grad()
            loss = outputs.loss
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            global_steps += 1
            acc_batches += len(batch['input_ids'])
            description = f"epoch: {epoch+1:03d} | step: {global_steps:05d} | loss: {train_loss/acc_batches:.4f}"
            pbar.set_description(description)

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    dataset_args, model_args, retriever_args, training_args = get_args()

    update_save_path(training_args)
    set_logging(dataset_args, model_args, retriever_args, training_args)

    model, tokenizer, optimizer = get_model(model_args, training_args)
    model.to(device)

    train_dataloader, eval_dataloader = get_data_loaders(dataset_args, training_args, tokenizer)

    # set wandb
    wandb_entity = 'zgotter'
    wandb_project = 'mrc'

    wandb.login()
    wandb.init(
        project=wandb_project,
        entity=wandb_entity,
        name=training_args.run_name
    )

    train_mrc(
        dataset_args, model_args, retriever_args, training_args,
        device,
        model, optimizer, tokenizer,
        train_dataloader, eval_dataloader
    )

if __name__ == "__main__":




    main()