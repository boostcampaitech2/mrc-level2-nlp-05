import os
import sys
import wandb
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

from preprocessor import BaselinePreprocessor
from postprocessor import post_processing_function
from model.models import BaseModel
from retrieval import SparseRetrieval
from utils import increment_path, LossObject

logger = logging.getLogger(__name__)

def get_args():# arguments들 가져오기
    """argument 객체 생성 함수"""
    parser = HfArgumentParser(
        (DefaultArguments, DatasetArguments, ModelArguments, RetrieverArguments, TrainingArguments)
    )
    default_args, dataset_args, model_args, retriever_args, training_args = parser.parse_args_into_dataclasses()
    return default_args, dataset_args, model_args, retriever_args, training_args

def update_save_path(training_args): # 모델 저장 overwrite하는 것도 포함.
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
    # 로깅인데 지금 나한테 중요한건 아님.
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

def set_seed_everything(seed):
    # seed 적용 자동으로 안될것 같은 부분들 다 seed 적용해줌.
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
    # 파라미터들에 weight decay를 적용해줬네? 이부분 나한테 필요할듯
    """weight decay가 적용된 모델 파라미터 생성 함수"""
    # no_decay라는 array에 문자열 두개 가져옴.
    no_decay = ["bias", "LayerNorm.weight"]
    grouped_parameters = [
        {
            "params": [param for name, param in model.named_parameters() if not any(nd in name for nd in no_decay)],
            # model.named_parameters()를 가져옴 (name, param)이렇게 옴. -> 만약 bias가 name에 포함되어 있지 않고, LayerNorm.weight도 name에 없을때만 param 저장 
            "weight_decay": training_args.weight_decay# weight_decay가 적용된 args
        },
        {
            "params": [param for name, param in model.named_parameters() if any(nd in name for nd in no_decay)],
            # model.named
            # 만약 bias가 name에 포함되어 있거나, LayerNorm.weight라는 말이 name에 있을때만 param을 저장함.
            "weight_decay": 0.0 # wiehgt_decay는 0
        }
    ]
    return grouped_parameters

def need_weight_freeze(default_args, epoch, max_epoch):
    freeze = False

    if default_args.freeze_pretrained_weight == 'all':
        freeze = True

    if default_args.freeze_pretrained_weight == 'first':
        if epoch <= default_args.freeze_pretrained_weight_epoch:
            freeze = True
        else :
            freeze = False

    if default_args.freeze_pretrained_weight == 'last':
        if max_epoch - epoch <= default_args.freeze_pretrained_weight_epoch:
            freeze = True
        else :
            freeze = False
            
    return freeze

def control_pretained_weight(model, freeze=False): #default_args, 
    """pretrained weight freeze options - none, all, first, last, random"""
    requires_grad = not freeze
    for name, param in model.named_parameters():
        if 'qa_outputs' not in name:
            param.requires_grad = requires_grad
    if freeze :
        print('freeze')
    else :
        print('melt')
    return model

    




def get_model(model_args, training_args):
    # config, tokeinzer, model, optimizer 가져오는 부분
    """model, tokenizer, optimizer 객체 생성 함수"""
    # model config
    config = AutoConfig.from_pretrained(
        model_args.config if model_args.config is not None else model_args.model
    )

    # tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer if model_args.tokenizer is not None else model_args.model
    )

    # model
    model = AutoModelForQuestionAnswering.from_pretrained(
        model_args.model, from_tf=bool(".ckpt" in model_args.model), config=config
    )
    # TODO: load custom model here
    #model.qa_outputs = CustomModel(config)

    # optimizer
    optimizer = AdamW(
        params=get_grouped_parameters(model, training_args),
        lr=training_args.learning_rate,
        eps=training_args.adam_epsilon
    )

    return config, model, tokenizer, optimizer

def get_data(dataset_args, training_args, tokenizer):
    # 데이터 생성, preprocess 적용, dataloader 만듬.
    """데이터셋 생성, preprocess 적용, dataLoader 객체 생성 함수"""
    datasets = load_from_disk(dataset_args.dataset_path)
    train_dataset = datasets['train']
    eval_dataset = datasets['validation'] 
    eval_dataset_for_predict = datasets['validation']
    column_names = train_dataset.column_names

    preprocessor = BaselinePreprocessor(
        dataset_args=dataset_args, tokenizer=tokenizer, column_names=column_names
    )
    # 모델 학습 및 training loss 계산을 위한 dataset
    train_dataset = train_dataset.map(
        preprocessor.prepare_train_features,
        batched=True,
        num_proc=dataset_args.num_workers,
        remove_columns=column_names,
        load_from_cache_file=not dataset_args.overwrite_cache,
    )
    # 모델 평가 및 eval loss 계산을 위한 dataset
    eval_dataset = eval_dataset.map(
        preprocessor.prepare_train_features,
        batched=True,
        num_proc=dataset_args.num_workers,
        remove_columns=column_names,
        load_from_cache_file=not dataset_args.overwrite_cache,
    )
    # evaluation(validation) 데이터셋에 대한 예측값 생성 및 평가지표 계산을 위한 dataset
    eval_dataset_for_predict = eval_dataset_for_predict.map(
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

    return datasets, eval_dataset_for_predict, train_dataloader, eval_dataloader

def get_scheduler(optimizer, train_dataloader, training_args):
    # warmup 적용하는 부분
    num_training_steps = len(train_dataloader) // training_args.gradient_accumulation_steps * training_args.num_train_epochs
    scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(
        optimizer,
        num_warmup_steps=training_args.warmup_steps,
        num_training_steps=num_training_steps
    )
    return scheduler

def concat_context_logits(logits, dataset, max_len):
    # logit들을 하나로 합침( concat)
    """각 batch의 logits을 하나로 합쳐주는 함수"""
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

def train_step(model, optimizer, scheduler, batch, device):
    # train_step 마다 loss 연산
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

def evaluation_step(model, datasets, eval_dataset_for_predict, eval_dataloader, dataset_args, training_args, device):
    # metric 계산
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
        start_logits_list.append(start_logits.detach().cpu().numpy())
        end_logits_list.append(end_logits.detach().cpu().numpy())

    max_len = max(x.shape[1] for x in start_logits_list)

    start_logits_concat = concat_context_logits(start_logits_list, eval_dataset_for_predict, max_len)
    end_logits_concat = concat_context_logits(end_logits_list, eval_dataset_for_predict, max_len)

    eval_dataset_for_predict.set_format(type=None, columns=list(eval_dataset_for_predict.features.keys()))
    predictions = (start_logits_concat, end_logits_concat)
    eval_preds = post_processing_function(datasets['validation'], eval_dataset_for_predict, datasets, predictions, training_args, dataset_args)
    eval_metric = metric.compute(predictions=eval_preds.predictions, references=eval_preds.label_ids) # compute_metrics
    
    return eval_metric, loss, eval_num

def train_mrc(
    default_args, dataset_args, model_args, retriever_args, training_args,
    model, optimizer, scheduler, tokenizer,
    datasets, eval_dataset_for_predict,
    train_dataloader, eval_dataloader,
    device
):
    """MRC 모델 학습 및 평가 함수"""
    prev_eval_loss = float('inf')
    global_steps = 0
    train_loss_obj = LossObject()
    eval_loss_obj = LossObject()
    max_epoch = int(training_args.num_train_epochs)
    for epoch in range(max_epoch):
        pbar = tqdm(
            enumerate(train_dataloader),
            total=len(train_dataloader),
            position=0,
            leave=True
        )
        control_pretained_weight(model,freeze=need_weight_freeze(default_args, epoch+1, max_epoch))
        
        for step, batch in pbar:
            loss = train_step(model, optimizer, scheduler, batch, device)

            global_steps += 1
            train_loss_obj.update(loss, len(batch['input_ids']))

            description = f"epoch: {epoch+1:03d} | step: {global_steps:05d} | train loss: {train_loss_obj.get_avg_loss():.4f}"
            pbar.set_description(description)

            if global_steps % training_args.eval_steps == 0:
                with torch.no_grad():
                    eval_metric, eval_loss, eval_num = evaluation_step(model, datasets, eval_dataset_for_predict, eval_dataloader, dataset_args, training_args, device)

                eval_loss_obj.update(eval_loss, eval_num)

                if eval_loss_obj.get_avg_loss() < prev_eval_loss:
                    # TODO: 5개 저장됐을 때 삭제하는 로직 개발 필요 -> huggingface format 모델 저장 필요
                    model.save_pretrained(os.path.join(training_args.output_dir, f"checkpoint-{global_steps:05d}"))
                    prev_eval_loss = eval_loss_obj.get_avg_loss()
                # TODO: 하이퍼파라미터(arguments) 정보 wandb에 기록하는 로직 필요
                wandb.log({
                    'global_steps': global_steps,
                    'train/loss': train_loss_obj.get_avg_loss(),
                    'train/learning_rate': training_args.learning_rate,
                    'eval/loss': eval_loss_obj.get_avg_loss(),
                    'eval/exact_match' : eval_metric['exact_match'],
                    'eval/f1_score' : eval_metric['f1']
                })
                train_loss_obj.reset()
                eval_loss_obj.reset()            
                
            else:
                wandb.log({'global_steps':global_steps})

def main():
    default_args, dataset_args, model_args, retriever_args, training_args = get_args()
    set_logging(default_args, dataset_args, model_args, retriever_args, training_args)
    update_save_path(training_args)
    set_seed_everything(training_args.seed)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    config, model, tokenizer, optimizer = get_model(model_args, training_args)
    model.to(device)

    datasets, eval_dataset_for_predict, train_dataloader, eval_dataloader = get_data(dataset_args, training_args, tokenizer)

    scheduler = get_scheduler(optimizer, train_dataloader, training_args)

    # set wandb
    wandb.login()
    wandb.init(
        project=default_args.wandb_project,
        entity=default_args.wandb_entity,
        name=training_args.run_name
    )

    train_mrc(
        default_args, dataset_args, model_args, retriever_args, training_args,
        model, optimizer, scheduler, tokenizer,
        datasets, eval_dataset_for_predict,
        train_dataloader, eval_dataloader,
        device
    )

if __name__ == "__main__":
    main()