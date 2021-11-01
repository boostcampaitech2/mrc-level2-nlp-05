import os
import sys
import wandb
import shutil
import logging
import random
import numpy as np
from tqdm import tqdm
from importlib import import_module

from arguments import (
    DefaultArguments,
    DatasetArguments,
    ModelArguments,
    RetrieverArguments,
)

import torch
from torch.utils.data import DataLoader

from transformers import (
    set_seed,
    AutoConfig,
    AutoTokenizer,
    AutoModelForQuestionAnswering,
    DataCollatorWithPadding,
    AdamW,
    TrainingArguments,
    HfArgumentParser,
    get_cosine_with_hard_restarts_schedule_with_warmup
)

from datasets import load_from_disk, load_metric
from processor import QAProcessor
from retrieval import SparseRetrieval
from utils import increment_path, LossObject, SaveLimitObject

logger = logging.getLogger(__name__)

def get_args():
    """argument 객체 생성 함수"""
    parser = HfArgumentParser(
        (DefaultArguments, DatasetArguments, ModelArguments, RetrieverArguments, TrainingArguments)
    )
    default_args, dataset_args, model_args, retriever_args, training_args = parser.parse_args_into_dataclasses()
    return default_args, dataset_args, model_args, retriever_args, training_args

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

def set_logging(default_args, dataset_args, model_args, retriever_args, training_args):
    """logging setting 함수"""
    logging.basicConfig(
        format="%(asctime)s - %(module)s - %(levelname)s  %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
        level=training_args.get_process_log_level()
    )
    logger.debug("Default arguments %s", default_args)
    logger.debug("Dataset arguments %s", dataset_args)
    logger.debug("Model arguments %s", model_args)
    logger.debug("Retriever arguments %s", retriever_args)
    logger.debug("Training argumenets %s", training_args)

def set_seed_everything(seed):
    '''seed 고정 함수'''
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    set_seed(seed)

def get_grouped_parameters(model, training_args):
    """weight decay가 적용된 모델 파라미터 생성 함수"""
    no_decay = ["bias", "LayerNorm.weight"]
    grouped_parameters = [
        {
            "params": [param for name, param in model.named_parameters() if not any(nd in name for nd in no_decay)],
            "weight_decay": training_args.weight_decay
        },
        {
            "params": [param for name, param in model.named_parameters() if any(nd in name for nd in no_decay)],
            "weight_decay": 0.0
        }
    ]
    return grouped_parameters

def get_model(model_args, training_args):
    """model, tokenizer, optimizer 객체 생성 함수"""
    # model config
    config = AutoConfig.from_pretrained(
        model_args.config if model_args.config is not None else model_args.model
    )

    # tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer if model_args.tokenizer is not None else model_args.model
    )

    if model_args.custom_model is None:
        model = AutoModelForQuestionAnswering.from_pretrained(
            model_args.model, from_tf=bool(".ckpt" in model_args.model), config=config
        )
    else:
        config.dropout_ratio = model_args.head_dropout_ratio
        custom_model_module = getattr(import_module('model.custom_models'), model_args.custom_model)
        model = custom_model_module(config).from_pretrained(
            model_args.model, from_tf=bool(".ckpt" in model_args.model), config=config
        )

    # optimizer
    optimizer = AdamW(
        params=get_grouped_parameters(model, training_args),
        lr=training_args.learning_rate,
        eps=training_args.adam_epsilon
    )

    return config, model, tokenizer, optimizer

def get_dataloader(qa_processor, dataset_args, training_args, tokenizer):
    """dataLoader 객체 생성 함수"""
    data_collator = DataCollatorWithPadding(tokenizer)

    train_features = qa_processor.get_train_features()
    train_features = train_features.remove_columns(['example_id', 'offset_mapping', 'overflow_to_sample_mapping'])

    eval_features = qa_processor.get_eval_features()
    eval_features = eval_features.remove_columns(['example_id', 'offset_mapping', 'overflow_to_sample_mapping'])

    train_dataloader = DataLoader(
        train_features,
        collate_fn = data_collator,
        batch_size=training_args.per_device_train_batch_size
    )
    eval_dataloader = DataLoader(
        eval_features,
        collate_fn = data_collator,
        batch_size=training_args.per_device_eval_batch_size
    )

    return train_dataloader, eval_dataloader

def get_scheduler(optimizer, train_dataloader, training_args, model_args):
    num_training_steps = len(train_dataloader) // training_args.gradient_accumulation_steps * training_args.num_train_epochs
    scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(
        optimizer,
        num_warmup_steps=training_args.warmup_steps,
        num_training_steps=num_training_steps,
        num_cycles=model_args.warmup_cycles
    )
    return scheduler

    
def need_weight_freeze(model_args, epoch, max_epoch):

    freeze = False

    if model_args.freeze_pretrained_weight == 'all':
        freeze = True
    elif model_args.freeze_pretrained_weight == 'first':
        assert model_args.freeze_pretrained_weight_epoch < max_epoch, '`freeze_pretrained_weight_epoch` cannot be larger than `num_train_epochs`'
        if epoch <= model_args.freeze_pretrained_weight_epoch:
            freeze = True
    elif model_args.freeze_pretrained_weight == 'last':
        assert model_args.freeze_pretrained_weight_epoch < max_epoch, '`freeze_pretrained_weight_epoch` cannot be larger than `num_train_epochs`'
        if max_epoch - epoch < model_args.freeze_pretrained_weight_epoch:
            freeze = True
            
    return freeze


def control_pretained_weight(model, model_args, freeze=False): 
    """pretrained weight freeze options - none, all, first, last"""
    
    for name, param in model.named_parameters():
        if 'qa_outputs' not in name:
            param.requires_grad = not freeze
        if 'embeddings' in name :
            param.requires_grad = not model_args.freeze_embedding_layer_weight

    if freeze :
        logger.info("Current epoch's freeze status: freeze")
    else :
        logger.info("Current epoch's freeze status: unfreeze")
    
    if model_args.freeze_embedding_layer_weight :
        logger.info("Current epoch's embedding layer status: freeze")
    else :
        logger.info("Current epoch's embedding layer status: unfreeze")

    return model


def train_step(model, optimizer, scheduler, batch, device):
    """각 training batch에 대한 모델 학습 및 train loss 계산 함수"""
    model.train()
    batch = batch.to(device)
    outputs = model(**batch)
    
    optimizer.zero_grad()
    loss = outputs.loss
    loss.backward()
    optimizer.step()
    scheduler.step()

    return loss.item()

def evaluation_step(model, qa_processor, eval_features_for_predict, eval_examples, eval_dataloader, dataset_args, training_args, checkpoint_folder, device):
    """모든 evaluation dataset에 대한 loss 및 metric 계산 함수"""
    metric = load_metric("squad")
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

        start_logits = outputs['start_logits'] # (batch_size, token_num)
        end_logits = outputs['end_logits'] # (batch_size, token_num)
        start_logits_list.extend(start_logits.detach().cpu().numpy())
        end_logits_list.extend(end_logits.detach().cpu().numpy())

    eval_features_for_predict.set_format(type=None, columns=list(eval_features_for_predict.features.keys()))
    predictions = (start_logits_list, end_logits_list)

    output_dir_origin = training_args.output_dir

    checkpoint_dir = os.path.join(output_dir_origin, checkpoint_folder)
    training_args.output_dir = checkpoint_dir
    os.makedirs(checkpoint_dir)
    
    eval_preds = qa_processor.post_processing_function(
        eval_examples, eval_features_for_predict, predictions, training_args
    )
    eval_metric = metric.compute(predictions=eval_preds.predictions, references=eval_preds.label_ids) # compute_metrics

    training_args.output_dir = output_dir_origin
    
    return eval_metric, loss, eval_num

def train_mrc(
    default_args, dataset_args, model_args, retriever_args, training_args,
    model, optimizer, scheduler, tokenizer,
    qa_processor, eval_features_for_predict, eval_examples,
    train_dataloader, eval_dataloader,
    device
):
    """MRC 모델 학습 및 평가 함수"""
    prev_eval_loss = float('inf')
    prev_eval_em = 0
    best_checkpoint = ""
    train_loss_obj = LossObject()
    eval_loss_obj = LossObject()
    max_epoch = int(training_args.num_train_epochs)
    save_limit_obj = SaveLimitObject(training_args.save_total_limit) if training_args.save_total_limit is not None else None
    global_steps = 0
    max_steps = max_epoch * len(train_dataloader)
    for epoch in range(max_epoch):
        pbar = tqdm(
            enumerate(train_dataloader),
            total=len(train_dataloader),
            position=0,
            leave=True
        )
        control_pretained_weight(model, model_args, freeze=need_weight_freeze(model_args, epoch+1, max_epoch))

        for step, batch in pbar:
            loss = train_step(model, optimizer, scheduler, batch, device)

            global_steps += 1
            train_loss_obj.update(loss, len(batch['input_ids']))

            description = f"epoch: {epoch+1:03d} | step: {global_steps:05d} | train loss: {train_loss_obj.get_avg_loss():.4f}"
            pbar.set_description(description)

            lr = scheduler.get_last_lr()[0]

            if global_steps % training_args.eval_steps == 0 or global_steps == max_steps:
                checkpoint_folder = f"checkpoint-{global_steps:05d}"

                with torch.no_grad():
                    eval_metric, eval_loss, eval_num = evaluation_step(
                        model, qa_processor,
                        eval_features_for_predict, eval_examples, eval_dataloader,
                        dataset_args, training_args,
                        checkpoint_folder, device
                    )

                eval_loss_obj.update(eval_loss, eval_num)

                save_path = os.path.join(training_args.output_dir, checkpoint_folder)
                if eval_loss_obj.get_avg_loss() <= prev_eval_loss:
                    if save_limit_obj is not None:
                        save_limit_obj.update(save_path)    
                    model.save_pretrained(save_path)
                    prev_eval_loss = eval_loss_obj.get_avg_loss()
                    best_checkpoint = checkpoint_folder
                elif eval_metric['exact_match'] >= prev_eval_em:
                    if save_limit_obj is not None:
                        save_limit_obj.update(save_path)
                    model.save_pretrained(save_path)
                    prev_eval_em = eval_metric['exact_match']
                else:
                    shutil.rmtree(save_path)

                wandb.log({
                    'global_steps': global_steps,
                    'learning_rate': lr,
                    'train/loss': train_loss_obj.get_avg_loss(),
                    'eval/loss': eval_loss_obj.get_avg_loss(),
                    'eval/exact_match' : eval_metric['exact_match'],
                    'eval/f1_score' : eval_metric['f1']
                })
                train_loss_obj.reset()
                eval_loss_obj.reset()            
                
            else:
                wandb.log({
                    'global_steps':global_steps,
                    'learning_rate': lr
                })

    logger.info(f"Best model in {best_checkpoint}")

def main():
    default_args, dataset_args, model_args, retriever_args, training_args = get_args()
    set_logging(default_args, dataset_args, model_args, retriever_args, training_args)
    update_save_path(training_args)
    set_seed_everything(training_args.seed)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    config, model, tokenizer, optimizer = get_model(model_args, training_args)
    model.to(device)

    qa_processor = QAProcessor(
        dataset_args=dataset_args,
        tokenizer=tokenizer,
        concat=dataset_args.concat_eval
    )
    eval_features_for_predict = qa_processor.get_eval_features()
    eval_examples = qa_processor.get_eval_examples()
    train_dataloader, eval_dataloader = get_dataloader(qa_processor, dataset_args, training_args, tokenizer)
    scheduler = get_scheduler(optimizer, train_dataloader, training_args, model_args)

    # set wandb
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

    train_mrc(
        default_args, dataset_args, model_args, retriever_args, training_args,
        model, optimizer, scheduler, tokenizer,
        qa_processor, eval_features_for_predict, eval_examples,
        train_dataloader, eval_dataloader,
        device
    )

if __name__ == "__main__":
    main()