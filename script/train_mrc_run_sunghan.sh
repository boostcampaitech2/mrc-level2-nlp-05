#!/bin/bash

python train_mrc.py \
    --run_name save_model_with_em_test \
    --description "save model when eval loss increase but eval em increase" \
    --output_dir ./models \
    --model klue/roberta-large \
    --num_train_epochs 5 \
    --use_max_padding \
    --do_eval \
    --warmup_steps 500 \
    --eval_steps 500 \
    --wandb_project mrc-head \
    --freeze_pretrained_weight first \
    --freeze_pretrained_weight_epoch 2 \
    --save_total_limit 5 \

    # --custom_model CustomRobertaForQuestionAnsweringWithLSTMCNNHead \
    # --head_dropout_ratio 0.5