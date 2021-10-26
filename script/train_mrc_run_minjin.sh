#!/bin/bash

#rm -rf ./models/*

# CustomHeadRNN testing
python train_mrc.py \
    --run_name exp008_custom_head_rnn_more_dropout_cycle3 \
    --description "custom head RNN with dropout 0.5, freeze first 2, warmup_cycles 3" \
    --output_dir ./models \
    --model klue/roberta-large \
    --num_train_epochs 5 \
    --use_max_padding \
    --do_eval \
    --warmup_steps 500 \
    --eval_steps 200 \
    --wandb_project mrc-head \
    --freeze_pretrained_weight first \
    --freeze_pretrained_weight_epoch 2 \
    --warmup_cycles 3 \
		--head CustomHeadRNN \
    --head_dropout_ratio 0.5



# python train_mrc.py \
#   --run_name exp005_xlm_roberta_large_aware \
#   --output_dir ./models \
#   --model a-ware/xlmroberta-squadv2 \
#   --num_train_epochs 5 \
#   --do_eval \
#   --use_max_padding \
#   --eval_steps 500 \
#   --warmup_steps 500 \
#   --wandb_project mrc-model

# python train_mrc.py \
# --run_name exp004_klue_bert_base_test_concat \
# --output_dir ./models \
# --model klue/bert-base \
# --num_train_epochs 2 \
# --use_max_padding \
# --eval_steps 500 \
# --do_eval \
# --warmup_steps 500 \
# --wandb_project mrc-klue-based-models
