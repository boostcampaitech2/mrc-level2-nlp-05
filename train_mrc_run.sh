#!/bin/bash

#rm -rf ./models/*
python train_mrc.py \
  --run_name xlm_roberta_large_deepset3 \
  --output_dir ./models \
  --model deepset/xlm-roberta-large-squad2 \
  --num_train_epochs 1 \
  --use_max_padding \
  --do_train \
  --do_eval \
  --evaluation_strategy steps \
  --eval_steps 100 \
  --wandb_project mrc-xlm-roberta-large