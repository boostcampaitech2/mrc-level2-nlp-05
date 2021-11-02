#!/bin/bash

# exp008
python inference.py \
--output_dir ./outputs/merge_test \
--model ./models/masking_test1/checkpoint-3200 \
--top_k_retrieval 3 \
--do_predict