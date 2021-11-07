#!/bin/bash

python train_mrc.py \
--run_name merge_test \
--description "merge_test" \
--output_dir ./models \
--model klue/roberta-large \
--num_train_epochs 5 \
--use_max_padding \
--do_eval \
--warmup_steps 200 \
--eval_steps 200 \
--wandb_project mrc-test \
--freeze_pretrained_weight first \
--freeze_pretrained_weight_epoch 2 \
--save_total_limit 5 \
--custom_model CustomRobertaForQuestionAnsweringWithAttentionHead_V2 \