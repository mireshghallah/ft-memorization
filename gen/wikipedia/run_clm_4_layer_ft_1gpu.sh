

python  run_clm.py \
    --dataset_name wikitext \
    --dataset_config_name wikitext-2-raw-v1 \
    --model_name_or_path gpt2 \
    --output_dir ./logs/  \
    --eval_steps 100 \
    --learning_rate 1e-4 \
    --block_size 1024 \
    --do_ref_model \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --train_layer_n_only 4 \

python  run_clm.py \
    --dataset_name wikitext \
    --dataset_config_name wikitext-2-raw-v1 \
    --model_name_or_path gpt2 \
    --output_dir ./logs/  \
    --eval_steps 100 \
    --learning_rate 5e-4 \
    --block_size 1024 \
    --do_ref_model \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --train_layer_n_only 4 \


python  run_clm.py \
    --dataset_name wikitext \
    --dataset_config_name wikitext-2-raw-v1 \
    --model_name_or_path gpt2 \
    --output_dir ./logs/  \
    --eval_steps 100 \
    --learning_rate 1e-3 \
    --block_size 1024 \
    --do_ref_model \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --train_layer_n_only 4 \


python  run_clm.py \
    --dataset_name wikitext \
    --dataset_config_name wikitext-2-raw-v1 \
    --model_name_or_path gpt2 \
    --output_dir ./logs/  \
    --eval_steps 100 \
    --learning_rate 8e-4 \
    --block_size 1024 \
    --do_ref_model \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --train_layer_n_only 4 \




