#!/bin/bash

python train_mrc.py \
  --run_name exp_freeze_xlm_roberta_large \
  --output_dir ./models \
  --model hanmaroo/xlm_roberta_large_korquad_v1 \
  --num_train_epochs 4 \
  --warmup_steps 500 \
  --use_max_padding \
  --do_train \
  --do_eval \
  --evaluation_strategy steps \
  --eval_steps 500 \
  --wandb_project mrc-freeze_test \
  --freeze_pretrained_weight first \
  --freeze_pretrained_weight_epoch 2