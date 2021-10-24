#!/bin/bash


python train_mrc.py \
  --run_name exp003_xlm_roberta_large_deepset-test \
  --output_dir ./models \
  --model deepset/xlm-roberta-large-squad2 \
  --num_train_epochs 5 \
  --warmup_steps 500 \
  --use_max_padding \
  --do_train \
  --do_eval \
  --evaluation_strategy steps \
  --eval_steps 10 \
  --wandb_project mrc-xlm-roberta-large