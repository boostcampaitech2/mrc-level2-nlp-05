#!/bin/bash


python train_mrc_v2.py \
--do_train --do_eval \
--output_dir ./saved --logging_dir ./logs --seed 42 \
--model klue/bert-base \
--num_train_epochs 5 --learning_rate 3.4e-5 --weight_decay 0.015 \
--max_seq_len 512 \
--evaluation_strategy steps --eval_steps 100 --logging_steps 100 --save_steps 100 \
--save_total_limit 5 \
--label_smoothing_factor 0.02 \
--run_name exp011_klue_bert_base_concat_n \
--wandb_project mrc-ensemble --wandb_entity this-is-real \
--description klue_bert_base \
--load_best_model_at_end True --metric_for_best_model eval_exact_match --greater_is_better True \
--freeze_type roberta --freeze_epoch 2.0 \
--test_eval_dataset_path test_validation_dataset --concat_eval True \