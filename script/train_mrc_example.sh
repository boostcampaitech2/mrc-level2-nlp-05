#!/bin/bash

python train_mrc.py \
--run_name exp011_klue_roberta_large_lstm_lm_concat_y \
--description "klue_roberta_large, lstm with layernorm custom head, concat_y" \
--output_dir ./saved \
--model klue/roberta-large \
--num_train_epochs 5 \
--use_max_padding \
--do_eval \
--warmup_steps 100 \
--eval_steps 100 \
--save_total_limit 7 \
--wandb_project mrc-ensemble \
--freeze_pretrained_weight first \
--freeze_pretrained_weight_epoch 2 \
--head CustomRobertaForQuestionAnsweringWithLSTMLNHead \
--head_dropout_ratio 0.7 \
--test_eval_dataset_path test_validation_dataset --concat_eval True \