#!/bin/bash

python train_mrc_trainer.py \
--run_name train_mrc_trainer_test \
--output_dir ./models \
--do_train \
--do_eval \
--per_device_train_batch_size 8 \
--per_device_eval_batch_size 16 \
--learning_rate 3e-5 \
--weight_decay 0.01 \
--label_smoothing_factor 0.1 \
--num_train_epochs 5 \
--lr_scheduler_type constant \
--evaluation_strategy steps \
--eval_steps 100 \
--logging_steps 100 \
--save_steps 100 \
--save_total_limit 5 \
--metric_for_best_model eval_exact_match \
--greater_is_better True \
--model klue/roberta-large \
--custom_model CustomRobertaForQuestionAnsweringWithLSTMCNNHead \
--head_dropout_ratio 0.5 \
--max_seq_len 512 \
--concat_eval False \
--report_to wandb \
--wandb_project mrc-trainer