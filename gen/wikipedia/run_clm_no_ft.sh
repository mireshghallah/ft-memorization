

accelerate launch   run_clm.py \
    --dataset_name wikitext \
    --dataset_config_name wikitext-2-raw-v1 \
    --model_name_or_path gpt2 \
    --output_dir ./logs/  \
    --eval_steps 100 \
    --learning_rate 5e-6 \
    --block_size 1024 \
    --do_ref_model \
    --num_train_epochs 0 \


