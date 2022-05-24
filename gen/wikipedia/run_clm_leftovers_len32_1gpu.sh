

python  ../run_clm.py \
    --dataset_name wikitext \
    --dataset_config_name wikitext-2-raw-v1 \
    --model_name_or_path gpt2 \
    --output_dir ./logs/  \
    --eval_steps 100 \
    --learning_rate 0.0002 \
    --block_size 32 \
    --do_ref_model \
    --train_head_only \
    --gradient_accumulation_steps 32 \
    --per_device_eval_batch_size 1 \




python  ../run_clm.py \
    --dataset_name wikitext \
    --dataset_config_name wikitext-2-raw-v1 \
    --model_name_or_path gpt2 \
    --output_dir ./logs/  \
    --eval_steps 100 \
    --learning_rate 0.0002 \
    --block_size 32 \
    --do_ref_model \
    --gradient_accumulation_steps 32 \
    --per_device_eval_batch_size 1 \


python  ../run_clm.py \
    --dataset_name wikitext \
    --dataset_config_name wikitext-2-raw-v1 \
    --model_name_or_path gpt2 \
    --output_dir ./logs/  \
    --eval_steps 100 \
    --learning_rate 0.0002 \
    --block_size 32 \
    --add_adapter \
    --adapter_reduction 16 \
    --do_ref_model \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 32 \

