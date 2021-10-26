#!/bin/bash

# exp008
python inference.py \
    --output_dir ./outputs/exp008_custom_head_rnn_more_dropout_freeze_1_2 \
    --dataset_path ../data/test_dataset/ \
    --model ./models/exp008_custom_head_rnn_more_dropout_freeze_1_2/checkpoint-01400 \
    --saved_checkpoint checkpoint-01400.pt \
    --tokenizer klue/roberta-large \
    --top_k_retrieval 3 \
    --do_predict \
    --head CustomHeadRNN \
    --head_dropout_ratio 0.5
