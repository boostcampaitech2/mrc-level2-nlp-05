#!/bin/bash

python train_mrc.py \
    --run_name exp008_custom_head_attention_shkim \
    --description "custom multihead attention , freeze first 2 (shkim)" \
    --output_dir ./models \
    --model klue/roberta-large \
    --num_train_epochs 5 \
    --use_max_padding \
    --do_eval \
    --warmup_steps 500 \
    --eval_steps 500 \
    --wandb_project mrc-head-test \
    --freeze_pretrained_weight first \
    --freeze_pretrained_weight_epoch 2 \
    --save_total_limit 5 \
    --custom_model CustomRobertaForQuestionAnsweringWithAttentionHead_V2 \

#    --head_dropout_ratio 0.5