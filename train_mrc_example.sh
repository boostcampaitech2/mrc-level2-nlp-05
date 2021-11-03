# python train_mrc_v2.py \
#     --do_train --do_eval \
#     --output_dir ./saved --logging_dir ./logs --seed 42 \
#     --model klue/roberta-large \
#     --num_train_epochs 7 --learning_rate 3.4e-5 --weight_decay 0.015 \
#     --max_seq_len 512 --max_ans_len 30 \
#     --evaluation_strategy steps --eval_steps 100 --logging_steps 100 --save_steps 200 \
#     --save_total_limit 5 \
#     --freeze_type roberta --freeze_epoch 1.0 \
#     --label_smoothing_factor 0.02 \
#     --run_name roberta_large_freeze_backbone \
#     --wandb_project exp_trainer --wandb_entity this-is-real \
#     --description 

python train_mrc_v2.py \
    --do_train --do_eval \
    --output_dir ./saved --logging_dir ./logs --seed 42 \
    --model klue/roberta-large \
    --num_train_epochs 7 --learning_rate 3.4e-5 --weight_decay 0.015 \
    --max_seq_len 512 --max_ans_len 30 \
    --evaluation_strategy steps --eval_steps 100 --logging_steps 100 --save_steps 200 \
    --save_total_limit 5 \
    --freeze_type roberta --freeze_epoch 2.0 \
    --label_smoothing_factor 0.02 \
    --run_name roberta_large_freeze_backbone \
    --wandb_project exp_trainer --wandb_entity this-is-real \
    --description exp_on_freeze_backbone