#!/bin/bash

python inference.py \
    --output_dir ./outputs/klue_bert_base \
    --dataset_path ../data/test_dataset/ \
    --model klue/bert-base \
    --top_k_retrieval 1 \
    --do_predict