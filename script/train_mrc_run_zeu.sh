#!/bin/bash

# python train_mrc.py \
#   --run_name exp_freeze_xlm_roberta_large \
#   --output_dir ./models \
#   --model hanmaroo/xlm_roberta_large_korquad_v1 \
#   --num_train_epochs 4 \
#   --warmup_steps 500 \
#   --use_max_padding \
#   --do_train \
#   --do_eval \
#   --evaluation_strategy steps \
#   --eval_steps 500 \
#   --wandb_project mrc-freeze_test \
#   --freeze_pretrained_weight first \
#   --freeze_pretrained_weight_epoch 2

  # python train_mrc.py \
  # --run_name exp006_klue_roberta_base_freeze \
  # --output_dir ./models \
  # --model klue/roberta-base \
  # --num_train_epochs 5 \
  # --use_max_padding \
  # --eval_steps 500 \
  # --do_eval \
	# --warmup_steps 500 \
  # --wandb_project mrc-klue-based-models \
  # --freeze_pretrained_weight first \
  # --freeze_pretrained_weight_epoch 1
  # python train_mrc.py \
  #   --run_name exp008_baseline \
  #   --description "baseline , freeze first 2" \
  #   --output_dir ./models \
  #   --model klue/roberta-large \
  #   --num_train_epochs 5 \
  #   --use_max_padding \
  #   --do_eval \
  #   --warmup_steps 500 \
  #   --eval_steps 500 \
  #   --wandb_project mrc-head-test \
  #   --freeze_pretrained_weight first \
  #   --freeze_pretrained_weight_epoch 2

  # python train_mrc_masking.py \
  #   --run_name exp008_custom_masking_head_attention \
  #   --description "custom masking head attention , freeze first 2" \
  #   --output_dir ./models \
  #   --model klue/roberta-base \
  #   --num_train_epochs 5 \
  #   --use_max_padding \
  #   --do_eval \
  #   --warmup_steps 500 \
  #   --eval_steps 100 \
  #   --wandb_project mrc-masking-test \
  #   --freeze_pretrained_weight first \
  #   --freeze_pretrained_weight_epoch 2

  # python train_mrc.py \
  #   --run_name exp008_custom_head_attention_cnn \
  #   --description "custom attention cnn, freeze first 2" \
  #   --output_dir ./models \
  #   --model klue/roberta-large \
  #   --num_train_epochs 5 \
  #   --use_max_padding \
  #   --do_eval \
  #   --warmup_steps 500 \
  #   --eval_steps 500 \
  #   --wandb_project mrc-head-test \
  #   --freeze_pretrained_weight first \
  #   --freeze_pretrained_weight_epoch 2 \
	# 	--custom_model CustomRobertaForQuestionAnsweringWithAttentionCNNHead

  # python train_mrc.py \
  #   --run_name exp008_custom_head_cnn_attention \
  #   --description "custom cnn attention , freeze first 2" \
  #   --output_dir ./models \
  #   --model klue/roberta-large \
  #   --num_train_epochs 5 \
  #   --use_max_padding \
  #   --do_eval \
  #   --warmup_steps 500 \
  #   --eval_steps 500 \
  #   --wandb_project mrc-head-test \
  #   --freeze_pretrained_weight first \
  #   --freeze_pretrained_weight_epoch 2 \
  #   --custom_model CustomRobertaForQuestionAnsweringWithCNNAttentionHead
  
  # python train_mrc.py \
  #   --run_name exp008_custom_head_cnn_attention \
  #   --description "custom multihead attention , freeze first 2" \
  #   --output_dir ./models \
  #   --model klue/roberta-large \
  #   --num_train_epochs 5 \
  #   --use_max_padding \
  #   --do_eval \
  #   --warmup_steps 500 \
  #   --eval_steps 500 \
  #   --wandb_project mrc-head-test \
  #   --freeze_pretrained_weight first \
  #   --freeze_pretrained_weight_epoch 2 \
  #   --custom_model CustomRobertaForQuestionAnsweringWithMultiHeadAttentionHead

# head 안돌아가서 v2  주석처리
  # python train_mrc_v2.py \
  #     --do_train --do_eval \
  #     --output_dir ./models --logging_dir ./logs --seed 42 \
  #     --model klue/roberta-large \
  #     --num_train_epochs 5 --learning_rate 3.4e-5 --weight_decay 0.015 \
  #     --max_seq_len 512 --max_ans_len 30 \
  #     --evaluation_strategy steps --eval_steps 100 --logging_steps 100 --save_steps 100 \
  #     --save_total_limit 1 \
  #     --freeze_type roberta --freeze_epoch 1.0 \
  #     --label_smoothing_factor 0.02 \
  #     --run_name masking_test_attention_head_v2 \
  #     --wandb_project mrc-head-test --wandb_entity this-is-real \
  #     --custom_model CustomRobertaForQuestionAnsweringWithAttentionHead

  # python train_mrc_v2.py \
  #     --do_train --do_eval \
  #     --output_dir ./models --logging_dir ./logs --seed 42 \
  #     --model klue/roberta-large \
  #     --num_train_epochs 5 --learning_rate 3.4e-5 --weight_decay 0.015 \
  #     --max_seq_len 512 --max_ans_len 30 \
  #     --evaluation_strategy steps --eval_steps 100 --logging_steps 100 --save_steps 100 \
  #     --save_total_limit 1 \
  #     --freeze_type roberta --freeze_epoch 1.0 \
  #     --label_smoothing_factor 0.02 \
  #     --run_name masking_test_attention_cnn_head_v2 \
  #     --wandb_project mrc-head-test --wandb_entity this-is-real \
  #     --custom_model CustomRobertaForQuestionAnsweringWithAttentionCNNHead


python train_mrc.py \
    --run_name test_attention_cnn_head_roberta_large_f1 \
    --description "attention_cnn_head freeze 1" \
    --output_dir ./models \
    --logging_dir ./logs --seed 42 \
    --model klue/roberta-large \
    --num_train_epochs 5 \
    --use_max_padding \
    --do_eval \
    --warmup_steps 500 \
    --eval_steps 100 \
    --wandb_project mrc-head-test \
    --freeze_pretrained_weight first \
    --freeze_pretrained_weight_epoch 1 \
    --save_total_limit 2 \
    --custom_model CustomRobertaForQuestionAnsweringWithAttentionCNNHead \

python train_mrc.py \
    --run_name test_attention_head_roberta_large_f1 \
    --description "attention head freeze1" \
    --output_dir ./models \
    --logging_dir ./logs --seed 42 \
    --model klue/roberta-large \
    --num_train_epochs 5 \
    --use_max_padding \
    --do_eval \
    --warmup_steps 500 \
    --eval_steps 100 \
    --wandb_project mrc-head-test \
    --freeze_pretrained_weight first \
    --freeze_pretrained_weight_epoch 1 \
    --save_total_limit 2 \
    --custom_model CustomRobertaForQuestionAnsweringWithAttentionHead \
