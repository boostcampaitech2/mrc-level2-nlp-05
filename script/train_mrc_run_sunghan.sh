#!/bin/bash

python train_mrc.py \
    --run_name exp008_custom_head_rnn_more_dropout_freeze_1_2 \
    --description "custom head rnn with dropout 0.5, freeze in 1 epoch, modify save method" \
    --output_dir ./models \
    --model klue/roberta-large \
    --num_train_epochs 5 \
    --use_max_padding \
    --do_eval \
    --warmup_steps 500 \
    --eval_steps 200 \
    --wandb_project mrc-head \
    --freeze_pretrained_weight first \
    --freeze_pretrained_weight_epoch 1 \
	--head CustomHeadRNN \
    --head_dropout_ratio 0.5 \
    --save_total_limit 3
