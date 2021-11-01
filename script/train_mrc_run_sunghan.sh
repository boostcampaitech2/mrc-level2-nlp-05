#!/bin/bash

python train_mrc.py \
    --run_name exp008_custom_head_sh_8 \
    --description "Change how to create objects in a model using custom head" \
    --output_dir ./models \
    --model klue/roberta-large \
    --num_train_epochs 5 \
    --use_max_padding \
    --do_eval \
    --warmup_steps 500 \
    --eval_steps 100 \
    --wandb_project mrc-head \
    --freeze_pretrained_weight first \
    --freeze_pretrained_weight_epoch 2 \
    --save_total_limit 3 \
    --custom_model CustomRobertaForQuestionAnsweringWithLSTMCNNHead \
    --head_dropout_ratio 0.5