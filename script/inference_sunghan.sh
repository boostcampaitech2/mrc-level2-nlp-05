#!/bin/bash

# exp008
python inference.py \
    --output_dir ./outputs/exp008_custom_masking_head_attention \
    --dataset_path ../data/test_dataset/ \
    --model ./models/exp008_custom_masking_head_attention19/checkpoint-04100 \
    --tokenizer klue/roberta-base \
    --top_k_retrieval 3 \
    --do_predict