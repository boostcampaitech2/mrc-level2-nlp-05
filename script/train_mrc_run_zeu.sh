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

  python train_mrc_masking.py \
    --run_name exp008_custom_masking_head_attention \
    --description "custom masking head attention , freeze first 2" \
    --output_dir ./models \
    --model klue/roberta-base \
    --num_train_epochs 5 \
    --use_max_padding \
    --do_eval \
    --warmup_steps 500 \
    --eval_steps 100 \
    --wandb_project mrc-masking-test \
    --freeze_pretrained_weight first \
    --freeze_pretrained_weight_epoch 2

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