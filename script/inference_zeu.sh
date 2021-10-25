#!/bin/bash

python inference.py \
    --output_dir ./outputs/klue_roberta_large_freeze_embed \
    --dataset_path ../data/test_dataset/ \
    --model ./models/exp006_large_freeze_embed/checkpoint-02500 \
    --top_k_retrieval 3 \
    --do_predict \
    --tokenizer klue/roberta-large \

    # /opt/ml/mrc-level2-nlp-05/models/exp006_klue_roberta_large_freeze/checkpoint-03500
    # /opt/ml/mrc-level2-nlp-05/models/exp006_freeze_embed/checkpoint-03500