#!/bin/bash

#rm -rf ./models/*
python train_mrc.py \
  --run_name exp003_xlm_roberta_large_aware \
  --output_dir ./models \
  --model a-ware/xlmroberta-squadv2 \
  --num_train_epochs 10 \
  --use_max_padding \
  --do_train \
  --do_eval \
  --evaluation_strategy steps \
  --eval_steps 500 \
  --wandb_project mrc-xlm-roberta-large