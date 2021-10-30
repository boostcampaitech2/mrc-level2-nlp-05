#!/bin/bash

# exp008
python inference.py \
    --output_dir ./outputs/exp008_custom_head_attention_shkim \
    --dataset_path ../data/test_dataset/ \
    --model ./models/exp008_custom_head_attention_shkim/checkpoint-04000 \
    --tokenizer klue/roberta-large \
    --top_k_retrieval 10 \
    --do_predict \
    --custom_model CustomRobertaForQuestionAnsweringWithAttentionHead_V2 \

#    --head_dropout_ratio 0.5