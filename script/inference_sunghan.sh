#!/bin/bash

# exp008
python inference.py \
    --output_dir ./outputs/exp008_custom_head_sh_8_3 \
    --dataset_path ../data/test_dataset/ \
    --model ./models/exp008_custom_head_sh_8/checkpoint-02900 \
    --tokenizer klue/roberta-large \
    --top_k_retrieval 1 \
    --do_predict \
    --custom_model CustomRobertaForQuestionAnsweringWithLSTMCNNHead \
    --head_dropout_ratio 0.5