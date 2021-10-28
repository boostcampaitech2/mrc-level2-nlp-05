#!/bin/bash

# exp008
python inference.py \
    --output_dir ./outputs/exp008_custom_head_sh_5 \
    --dataset_path ../data/test_dataset/ \
    --model ./models/exp008_custom_head_sh_5/checkpoint-02700 \
    --saved_checkpoint checkpoint-02700.pt \
    --tokenizer klue/roberta-large \
    --top_k_retrieval 3 \
    --do_predict \
    --head CustomHeadLSTMCNN \
    --head_dropout_ratio 0.5