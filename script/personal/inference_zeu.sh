#!/bin/bash

# exp008
python inference.py \
--output_dir ./outputs/masking_test_roberta_large_0.7 \
--model ./models/masking_test_roberta_large_0.7/checkpoint-2600 \
--top_k_retrieval 3 \
--do_predict