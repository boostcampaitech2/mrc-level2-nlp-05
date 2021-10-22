#!/bin/bash

#rm -rf ./models/*

python train_mrc.py \
--run_name exp004_klue_bert_base \
--output_dir ./models \
--model klue/bert-base \
--num_train_epochs 5 \
--use_max_padding \
--eval_steps 500 \
--warmup_steps 500 \
--wandb_project mrc-klue-based-models
