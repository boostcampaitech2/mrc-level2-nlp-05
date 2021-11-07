#!/bin/bash

# exp1
python train_dpr.py \
--run_name "exp1_dpr_klue-bert" \
--description "klue/bert-base_dpr" \
--dpr_model 'klue/bert-base' \
--dpr_learning_rate 3e-5 \
--dpr_epochs 10 \
--dpr_warmup_steps 500 \
--wandb_project 'dpr'