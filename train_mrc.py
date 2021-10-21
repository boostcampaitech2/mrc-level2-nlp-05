import os
import sys
import logging
from tqdm import tqdm
import wandb
import numpy as np

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
from utils_qa import check_no_error
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

def get_data(dataset_args, training_args, tokenizer):
    datasets = load_from_disk(dataset_args.dataset_path)
    train_dataset = datasets['train']
    eval_dataset = datasets['validation']
    eval_dataset_for_post = datasets['validation']
    column_names = train_dataset.column_names

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
        preprocessor.prepare_train_features,
        batched=True,
        num_proc=dataset_args.num_workers,
        remove_columns=column_names,
        load_from_cache_file=not dataset_args.overwrite_cache,
    )
    eval_dataset_for_post = eval_dataset_for_post.map(
        preprocessor.prepare_eval_features,
        batched=True,
        num_proc=dataset_args.num_workers,
        remove_columns=column_names,
        load_from_cache_file=not dataset_args.overwrite_cache,
    )    

    data_collator = DataCollatorWithPadding(tokenizer)

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

    return datasets, train_dataset, eval_dataset, eval_dataset_for_post, train_dataloader, eval_dataloader

def train_step(model, optimizer, batch, device):
    model.train()
    batch = batch.to(device)
    outputs = model(**batch)

    optimizer.zero_grad()
    loss = outputs.loss
    loss.backward()
    optimizer.step()

    return loss.item()

def concat_context_logits(logits, dataset, max_len):
    """Model의 Logit을 context 단위로 연결하는 함수"""
    step = 0
    logits_concat = np.full((len(dataset), max_len), -100, dtype=np.float64)

    for i, output_logit in enumerate(logits):
        batch_size = output_logit.shape[0]
        cols = output_logit.shape[1]
        if step + batch_size < len(dataset):
            logits_concat[step : step + batch_size, :cols] = output_logit
        else:
            logits_concat[step:, :cols] = output_logit[: len(dataset) - step]
        step += batch_size

    return logits_concat


def compute_metrics(pred: EvalPrediction):
    return metric.compute(predictions=pred.predictions, references=pred.label_ids)

def evaluation_step(model, datasets, eval_dataset, eval_dataloader, dataset_args, training_args, device):
    model.eval()

    start_logits_list = []
    end_logits_list = []

    loss = 0
    eval_num = 0
    for batch in eval_dataloader:
        batch = batch.to(device)
        outputs = model(**batch)

        loss += outputs.loss
        eval_num += len(batch['input_ids'])

        start_logits_list.append(outputs['start_logits'].detach().cpu().numpy())
        end_logits_list.append(outputs['end_logits'].detach().cpu().numpy())

    max_len = max(x.shape[1] for x in start_logits_list)

    start_logits_concat = concat_context_logits(start_logits_list, eval_dataset, max_len)
    end_logits_concat = concat_context_logits(end_logits_list, eval_dataset, max_len)

    eval_dataset.set_format(type=None, columns=list(eval_dataset.features.keys()))
    predictions = (start_logits_concat, end_logits_concat)
    eval_preds = post_processing_function(datasets['validation'], eval_dataset, datasets, predictions, training_args, dataset_args)
    eval_metric = compute_metrics(eval_preds)
    
    return eval_metric, loss, eval_num

def train_mrc(
    dataset_args, model_args, retriever_args, training_args,
    model, optimizer, tokenizer,
    datasets, train_dataset, eval_dataset, eval_dataset_for_post,
    train_dataloader, eval_dataloader,
    device
):
    """train 함수"""
    prev_eval_loss = float('inf')
    prev_em = 0
    prev_f1 = 0
    global_steps = 0
    train_acc_loss = 0
    eval_acc_loss = 0
    train_acc_num = 0
    eval_acc_num = 0
    for epoch in range(int(training_args.num_train_epochs)):
        pbar = tqdm(
            enumerate(train_dataloader),
            total=len(train_dataloader),
            position=0,
            leave=True
        )
        for step, batch in pbar:
            loss = train_step(model, optimizer, batch, device)

            train_acc_loss += loss
            global_steps += 1
            train_acc_num += len(batch['input_ids'])

            description = f"epoch: {epoch+1:03d} | step: {global_steps:05d} | loss: {train_acc_loss/train_acc_num:.4f}"
            pbar.set_description(description)

            if global_steps % training_args.eval_steps == 0:
                with torch.no_grad():
                    eval_metric, eval_loss, eval_num = evaluation_step(model, datasets, eval_dataset_for_post, eval_dataloader, dataset_args, training_args, device)
                eval_acc_loss += eval_loss
                eval_acc_num += eval_num
                if eval_acc_loss/eval_acc_num < prev_eval_loss:
                    #torch.save(model.state_dict(), os.path.join(training_args.output_dir, f"checkpoint-{global_steps:05d}.pt"))
                    prev_eval_loss = eval_acc_loss/eval_acc_num
                    prev_em = eval_metric['exact_match']
                    prev_f1 = eval_metric['f1']
                wandb.log({
                    'train/loss': train_acc_loss/train_acc_num,
                    'train/learning_rate': training_args.learning_rate,
                    'eval/loss': eval_acc_loss/eval_acc_num,
                    'eval/exact_match' : eval_metric['exact_match'],
                    'eval/f1_score' : eval_metric['f1'],
                    'global_steps': global_steps                    
                })
                train_acc_loss = 0
                eval_acc_loss = 0
                train_acc_num = 0
                eval_acc_num = 0                
            else:
                wandb.log({'global_steps':global_steps})   
                    # 네...마자요 불러오는 부분에서 수정이 좀 있어야 할거 가타여 ㅋㅋㅋㅋㅋ
                    # 내일은 뭐라도 기여를 해보겠습니다 오늘은 영 정신이 맑지가 않네요ㅠㅠ
                    # 잘 분리돼서 나오면 좋겠네요..< who?
                
def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    dataset_args, model_args, retriever_args, training_args = get_args()
    update_save_path(training_args)
    set_logging(dataset_args, model_args, retriever_args, training_args)

    model, tokenizer, optimizer = get_model(model_args, training_args)
    model.to(device)

    datasets, train_dataset, eval_dataset, eval_dataset_for_post, train_dataloader, eval_dataloader = get_data(dataset_args, training_args, tokenizer)

    # set wandb
    wandb_entity = 'this-is-real'
    wandb_project = 'mrc-xlm-roberta-large'
    
    wandb.login()
    wandb.init(
        project=wandb_project,
        entity=wandb_entity,
        name=training_args.run_name
    )

    train_mrc(
        dataset_args, model_args, retriever_args, training_args,
        model, optimizer, tokenizer,
        datasets, train_dataset, eval_dataset, eval_dataset_for_post,
        train_dataloader, eval_dataloader,
        device
    )

if __name__ == "__main__":
    main()