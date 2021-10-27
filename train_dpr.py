import json
import random
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from pprint import pprint
import os
import wandb
from typing import Any, Union, Dict, List, Optional

import torch
from torch.utils.data import DataLoader, TensorDataset, RandomSampler
import torch.nn.functional as F

from datasets import (
    load_metric,
    load_from_disk,
    Sequence,
    Value,
    Features,
    Dataset,
    DatasetDict,
)

from transformers import (
    AutoTokenizer, AutoModel, AutoConfig,
    BertModel, BertPreTrainedModel,
    AdamW, get_linear_schedule_with_warmup,
    TrainingArguments,HfArgumentParser
)


from arguments import DatasetArguments, RetrieverArguments

parser = HfArgumentParser((DatasetArguments, RetrieverArguments))
data_args, retriever_args= parser.parse_args_into_dataclasses()

# Inference 에서의 TrainingArguments 충돌방지 위해 따로 parsing
def get_dense_args():
    args = TrainingArguments(
            output_dir="dense_retrieval",
            evaluation_strategy="epoch",
            learning_rate=retriever_args.dpr_learning_rate,
            per_device_train_batch_size=retriever_args.dpr_train_batch,
            per_device_eval_batch_size=retriever_args.dpr_eval_batch,
            num_train_epochs=retriever_args.dpr_epochs,
            weight_decay=retriever_args.dpr_weight_decay,
            )

    retriever_dir = './models/retriever'
    p,q = 'p_encoder','q_encoder'

    if (os.path.isdir(os.path.join(retriever_dir,p)) and os.path.isdir(os.path.join(retriever_dir,q))):
        tokenizer = AutoTokenizer.from_pretrained(retriever_args.dpr_model)
        p_encoder  = BertEncoder.from_pretrained(os.path.join(retriever_dir, 'p_encoder'))
        q_encoder = BertEncoder.from_pretrained(os.path.join(retriever_dir, 'q_encoder'))
        return args, tokenizer, p_encoder, q_encoder
        
    else:
        print('No fine-tuned DPR exists ... Training Dense Passage Retrieval...')
        return args




class Dense:
    def __init__(self, args, dataset,
        tokenizer, p_encoder, q_encoder,  data_path: Optional[str] = "../data/",
        context_path: Optional[str] = "wikipedia_documents.json"
    ):
        self.args = args
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.p_encoder = p_encoder
        self.q_encoder = q_encoder

        self.data_path = data_path
        with open(os.path.join(data_path, context_path), "r", encoding="utf-8") as f:
            wiki = json.load(f)

        self.contexts = list(
            dict.fromkeys([v["text"] for v in wiki.values()])
        )
        self.ids = list(range(len(self.contexts)))


class BertEncoder(BertPreTrainedModel):
    def __init__(self, config):
        super(BertEncoder, self).__init__(config)

        self.bert = BertModel(config)
        self.init_weights()
      
    def forward(
            self,
            input_ids, 
            attention_mask=None,
            token_type_ids=None
        ): 

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )

        pooled_output = outputs[1]
        return pooled_output


class DenseTrain(Dense):
    def __init__(self, num_neg,**kwargs):
        super(DenseTrain, self).__init__(**kwargs)
        self.num_neg = num_neg
        self.prepare_in_batch_negative(num_neg = num_neg)
    
    def prepare_in_batch_negative(
        self,
        dataset = None,
        num_neg = None,
        tokenizer = None
    ):
        if dataset is None:
            dataset = self.dataset
        if tokenizer is None:
            tokenizer = self.tokenizer
        if num_neg is None:
            num_neg = self.num_neg
        
        corpus = np.array(list(set([example for example in dataset["context"]])))
        p_with_neg = []

        for c in dataset['context']:
            while True:
                neg_idxs = np.random.randint(len(corpus), size = num_neg)

                if not c in corpus[neg_idxs]:
                    p_neg = corpus[neg_idxs]

                    p_with_neg.append(c)
                    p_with_neg.extend(p_neg)
                    break

        q_seqs = tokenizer(
            dataset['question'],
            padding = "max_length",
            truncation = True,
            return_tensors = 'pt'
        )
        p_seqs = tokenizer(
            p_with_neg,
            padding = "max_length",
            truncation = True,
            return_tensors = 'pt'
        )

        max_len = p_seqs['input_ids'].size(-1)
        p_seqs['input_ids'] = p_seqs['input_ids'].view(-1, num_neg+1, max_len)
        p_seqs['attention_mask'] = p_seqs['attention_mask'].view(-1, num_neg+1, max_len)
        p_seqs['token_type_ids'] = p_seqs['token_type_ids'].view(-1, num_neg+1, max_len)

        train_dataset = TensorDataset(
            p_seqs['input_ids'], p_seqs['attention_mask'], p_seqs['token_type_ids'],
            q_seqs['input_ids'], q_seqs['attention_mask'], q_seqs['token_type_ids']
        )

        train_sampler = RandomSampler(train_dataset)
        self.train_dataloader = DataLoader(
            train_dataset,
            sampler = train_sampler,
            batch_size = self.args.per_device_train_batch_size
        )

        valid_seqs = tokenizer(
            self.contexts,
            padding = "max_length",
            truncation = True,
            return_tensors = 'pt'
        )

        passage_dataset = TensorDataset(
            valid_seqs['input_ids'],
            valid_seqs['attention_mask'],
            valid_seqs['token_type_ids']
        )

        self.passage_dataloader = DataLoader(
            passage_dataset,
            batch_size = self.args.per_device_train_batch_size
        )

    def train(self, args = None):
        if args is None:
            args = self.args
        batch_size = args.per_device_train_batch_size

        # Optim : AdamW
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {"params": [p for n, p in self.p_encoder.named_parameters() if not any(nd in n for nd in no_decay)], "weight_decay": args.weight_decay},
            {"params": [p for n, p in self.p_encoder.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
            {"params": [p for n, p in self.q_encoder.named_parameters() if not any(nd in n for nd in no_decay)], "weight_decay": args.weight_decay},
            {"params": [p for n, p in self.q_encoder.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0}
        ]
        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=args.learning_rate,
            eps=args.adam_epsilon
        )
        t_total = len(self.train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=args.warmup_steps,
            num_training_steps=t_total
        )
        self.p_encoder.train()
        self.q_encoder.train()

        # Train Start
        global_step = 0

        self.p_encoder.zero_grad()
        self.q_encoder.zero_grad()
        torch.cuda.empty_cache()

        # train_iterator = tqdm(range(int(args.num_train_epochs)), desc="Epoch")

        for _ in tqdm(range(int(args.num_train_epochs)), desc="Epoch"):
            # train_loss = 0.0

            with tqdm(self.train_dataloader, unit="batch") as tepoch:
                for batch in tepoch:
            
                    targets = torch.zeros(batch_size).long() # positive example은 전부 첫 번째에 위치하므로
                    targets = targets.to(args.device)

                    p_inputs = {
                        "input_ids": batch[0].view(batch_size * (self.num_neg + 1), -1).to(args.device),
                        "attention_mask": batch[1].view(batch_size * (self.num_neg + 1), -1).to(args.device),
                        "token_type_ids": batch[2].view(batch_size * (self.num_neg + 1), -1).to(args.device)
                    }
            
                    q_inputs = {
                        "input_ids": batch[3].to(args.device),
                        "attention_mask": batch[4].to(args.device),
                        "token_type_ids": batch[5].to(args.device)
                    }

                   
                    p_outputs = self.p_encoder(**p_inputs)  # (batch_size*(num_neg+1), emb_dim)
                    q_outputs = self.q_encoder(**q_inputs) # (batch_size*, emb_dim)

                    # Calculate similarity score & loss
                    p_outputs = torch.transpose(p_outputs.view(batch_size, self.num_neg+1,-1), 1, 2)
                    q_outputs = q_outputs.view(batch_size, 1, -1)

                    sim_scores = torch.bmm(q_outputs, p_outputs).squeeze()  #(batch_size, num_neg + 1)
                    sim_scores = sim_scores.view(batch_size, -1)
                    sim_scores = F.log_softmax(sim_scores, dim=1)

                    loss = F.nll_loss(sim_scores, targets)
                    # loss = loss / args.gradient_accumulation_steps
                    # print(loss)
                    tepoch.set_postfix(loss=f"{str(loss.item())}")
                    # train_loss += loss

                    loss.backward()
                    optimizer.step()
                    scheduler.step()

                    self.p_encoder.zero_grad()
                    self.q_encoder.zero_grad()

                    global_step += 1

                    torch.cuda.empty_cache()

                    del p_inputs, q_inputs
        
            # print(f'Train Loss:{train_loss/len(self.train_dataloader)} in Epoch : {epoch}')
        return self.p_encoder, self.q_encoder




def main():
    # wandb settings
    # wandb.login()
    # wandb.init(
    #     project='dpr', 
    #     # entity=default_args.wandb_entity,
    #     name='exp/dpr'
    # )

    datasets = load_from_disk(data_args.dataset_path)
    # train_dataset = datasets["train"]
    args = get_dense_args()

    tokenizer = AutoTokenizer.from_pretrained(retriever_args.dpr_model)
    p_encoder = BertEncoder.from_pretrained(retriever_args.dpr_model).to(args.device)
    q_encoder = BertEncoder.from_pretrained(retriever_args.dpr_model).to(args.device)


    # Retriever는 아래와 같이 사용할 수 있도록 코드를 짜봅시다.
    retriever = DenseTrain(
        args=args,
        dataset=datasets["train"],
        num_neg=retriever_args.num_negs,
        tokenizer=tokenizer,
        p_encoder=p_encoder,
        q_encoder=q_encoder
    )
    p_encoder, q_encoder = retriever.train()

    model_dir = './models/retriever'

    if not os.path.isdir(model_dir):
        os.makedirs(model_dir)

    p_encoder.save_pretrained(os.path.join(model_dir,'p_encoder'))
    q_encoder.save_pretrained(os.path.join(model_dir,'q_encoder'))

    print(f'passage & question encoders successfully saved at {model_dir}')


if __name__ == '__main__':
    main()