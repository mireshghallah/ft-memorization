
###
# accelerate launch   run_clm.py \
#     --dataset_name wikitext \
#     --dataset_config_name wikitext-2-raw-v1 \
#     --model_name_or_path gpt2 \
#     --output_dir ./logs/  \
#     --eval_steps 100 \
#     --learning_rate 5e-3 \
#     --block_size 1024 \
#     --add_adapter \
#     --adapter_reduction 16 \
#     --do_ref_model \



###
accelerate launch   run_clm.py \
    --dataset_name wikitext \
    --dataset_config_name wikitext-2-raw-v1 \
    --model_name_or_path gpt2 \
    --output_dir ./logs/  \
    --eval_steps 100 \
    --learning_rate 8e-5 \
    --block_size 1024 \
    --add_adapter \
    --adapter_reduction 2 \
    --do_ref_model \


accelerate launch   run_clm.py \
    --dataset_name wikitext \
    --dataset_config_name wikitext-2-raw-v1 \
    --model_name_or_path gpt2 \
    --output_dir ./logs/  \
    --eval_steps 100 \
    --learning_rate 5e-3 \
    --block_size 32 \
    --per_device_train_batch_size 1 \
    --add_adapter \
    --adapter_reduction 16 \
    --do_ref_model \


accelerate launch   run_clm.py \
    --dataset_name wikitext \
    --dataset_config_name wikitext-2-raw-v1 \
    --model_name_or_path gpt2 \
    --output_dir ./logs/  \
    --eval_steps 100 \
    --learning_rate 5e-3 \
    --block_size 32 \
    --add_adapter \
    --adapter_reduction 2 \
    --do_ref_model \
    --per_device_train_batch_size 1 \



accelerate launch   run_clm.py \
    --dataset_name wikitext \
    --dataset_config_name wikitext-2-raw-v1 \
    --model_name_or_path gpt2 \
    --output_dir ./logs/  \
    --eval_steps 100 \
    --learning_rate 5e-3 \
    --block_size 512 \
    --add_adapter \
    --adapter_reduction 16 \
    --do_ref_model \
    --per_device_train_batch_size 1 \



accelerate launch   run_clm.py \
    --dataset_name wikitext \
    --dataset_config_name wikitext-2-raw-v1 \
    --model_name_or_path gpt2 \
    --output_dir ./logs/  \
    --eval_steps 100 \
    --learning_rate 5e-3 \
    --block_size 256 \
    --add_adapter \
    --adapter_reduction 16 \
    --do_ref_model \
    --per_device_train_batch_size 1 \



accelerate launch   run_clm.py \
    --dataset_name wikitext \
    --dataset_config_name wikitext-2-raw-v1 \
    --model_name_or_path gpt2 \
    --output_dir ./logs/  \
    --eval_steps 100 \
    --learning_rate 5e-3 \
    --block_size 128 \
    --add_adapter \
    --adapter_reduction 16 \
    --do_ref_model \
    --per_device_train_batch_size 1 \



accelerate launch   run_clm.py \
    --dataset_name wikitext \
    --dataset_config_name wikitext-2-raw-v1 \
    --model_name_or_path gpt2 \
    --output_dir ./logs/  \
    --eval_steps 100 \
    --learning_rate 5e-3 \
    --block_size 64 \
    --add_adapter \
    --adapter_reduction 16 \
    --do_ref_model \
    --per_device_train_batch_size 1 \

