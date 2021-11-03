#!/bin/bash

# exp008
python inference.py \
--output_dir ./outputs/merge_test \
--model ./models/merge_test2/checkpoint-00200 \
--tokenizer klue/roberta-large \
--top_k_retrieval 10 \
--do_predict \
--custom_model CustomRobertaForQuestionAnsweringWithAttentionHead_V2 \
