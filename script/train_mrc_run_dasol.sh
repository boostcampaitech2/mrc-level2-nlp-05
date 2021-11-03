#!/bin/bash

#rm -rf ./models/*

python train_mrc_dasol_2.py \
    --run_name exp008_dasol_add_only_kor_vocab_no_slash_klue_rob_lg_freeze_2_base \
    --description "extra korean vocab, no slash 2nd test klue rob lg - freeze first 2" \
    --output_dir ./models \
    --model klue/roberta-large \
    --num_train_epochs 5 \
    --use_max_padding \
    --do_eval \
    --warmup_steps 500 \
    --eval_steps 100 \
    --save_total_limit 5 \
    --wandb_project mrc-vocab-exp \
    --freeze_pretrained_weight first \
    --freeze_pretrained_weight_epoch 2


# CustomHeadAttentionWithLN & thisisreal pretrained model
# python train_mrc_dasol_2.py \
#     --run_name exp008_dasol_only_kor_vocab_custom_attention_LN_mrc_rl \
#     --description "extra korean vocab + mrc roberta large + attention + LN with dropout 0.5, freeze first 2 / total 5 ep" \
#     --output_dir ./models \
#     --model this-is-real/mrc-pretrained-roberta-large-1 \
#     --custom_model CustomRobertaForQuestionAnsweringWithAttentionWithLNHead \
#     --num_train_epochs 5 \
#     --use_max_padding \
#     --do_eval \
#     --warmup_steps 500 \
#     --eval_steps 100 \
#     --save_total_limit 5 \
#     --wandb_project mrc-vocab-exp \
#     --freeze_pretrained_weight first \
#     --freeze_pretrained_weight_epoch 2



# CustomHeadAttentionWithLN
# python train_mrc_dasol_2.py \
#     --run_name exp008_dasol_add_only_kor_vocab_custom_attention_LN \
#     --description "extra korean vocab + roberta large + attention + LN with dropout 0.5, freeze first 2 / total 5 ep" \
#     --output_dir ./models \
#     --model klue/roberta-large \
#     --custom_model CustomRobertaForQuestionAnsweringWithAttentionWithLNHead \
#     --num_train_epochs 5 \
#     --use_max_padding \
#     --do_eval \
#     --warmup_steps 500 \
#     --eval_steps 100 \
#     --save_total_limit 5 \
#     --wandb_project mrc-vocab-exp \
#     --freeze_pretrained_weight first \
#     --freeze_pretrained_weight_epoch 2

# roberta-large base + mask_word only 해볼것
# python train_mrc_dasol_2.py \
#     --run_name exp009_dasol_add_only_kor_vocab_klue_rob_lg_freeze_2_mask_word_only \
#     --description "extra korean vocab + roberta large, freeze 2, mask word q only" \
#     --output_dir ./models \
#     --model klue/roberta-large \
#     --dataset_path /opt/ml/data/mask_word_q_train_dataset \
#     --num_train_epochs 5 \
#     --use_max_padding \
#     --eval_steps 100 \
#     --save_total_limit 5 \
#     --do_eval \
#     --warmup_steps 500 \
#     --wandb_project mrc-aug-exp \
#     --freeze_pretrained_weight first \
#     --freeze_pretrained_weight_epoch 2





# Add vocabulary test
# python train_mrc_dasol.py \
#     --run_name exp008_dasol_add_vocab_custom_head_rnn_more_dropout_cycle3 \
#     --description "add vocab custom head RNN with dropout 0.5, freeze first 2, warmup_cycles 3" \
#     --output_dir ./models \
#     --model klue/roberta-large \
#     --num_train_epochs 5 \
#     --use_max_padding \
#     --do_eval \
#     --warmup_steps 500 \
#     --eval_steps 200 \
#     --wandb_project mrc-head \
#     --freeze_pretrained_weight first \
#     --freeze_pretrained_weight_epoch 2 \
#     --warmup_cycles 3 \
# 	  --head CustomHeadRNN \
#     --head_dropout_ratio 0.5

# best model with add vocab(211102)
# python train_mrc.py \
#     --run_name exp008_dasol_add_vocab_klue_rob_lg_freeze_2_base \
#     --description "add vocab 2nd test klue rob lg - freeze first 2" \
#     --output_dir ./models \
#     --model klue/roberta-large \
#     --num_train_epochs 5 \
#     --use_max_padding \
#     --eval_steps 100 \
#     --save_total_limit 5 \
#     --do_eval \
#     --warmup_steps 500 \
#     --wandb_project mrc-head \
#     --freeze_pretrained_weight first \
#     --freeze_pretrained_weight_epoch 2


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
