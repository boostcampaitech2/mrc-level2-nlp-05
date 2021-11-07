#!/bin/bash

python inference.py \
    --output_dir ./outputs/klue_bert_base \
    --dataset_path ../data/test_dataset/ \
    --model ./models/exp011_klue_roberta_large_lstm_lm_concat_y \
    --top_k_retrieval 1 \
    --do_predict