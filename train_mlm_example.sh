python train_mlm.py --model klue/roberta-large \
    --output_dir ./pretrained --logging_dir ./pretrained_logs \
    --evaluation_strategy steps --do_train --do_eval \
    --max_seq_len 512 \
    --learning_rate 1e-5 --weight_decay 0.01 \
    --num_train_epochs 5 \
    --per_device_train_batch_size 8 --per_device_eval_batch_size 16 \
    --gradient_accumulation_steps 2 --lr_scheduler_type linear