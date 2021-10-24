#!/bin/bash

python train_mrc.py \
  --run_name klue_bert_base \
  --output_dir ./models \
  --model klue/bert-base \
  --num_train_epochs 5 \
  --warmup_steps 500 \
  --use_max_padding \
  --do_train \
  --do_eval \
  --evaluation_strategy steps \
  --eval_steps 500 \
  --wandb_project huggingface