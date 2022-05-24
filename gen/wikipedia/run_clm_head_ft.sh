

accelerate launch  run_clm.py \
    --dataset_name wikitext \
    --dataset_config_name wikitext-2-raw-v1 \
    --model_name_or_path gpt2 \
    --output_dir ./logs/  \
    --eval_steps 100 \
    --learning_rate 1e-5 \
    --block_size 1024 \
    --do_ref_model \
    --train_head_only \
   # --gradient_accumulation_steps 8 \




# accelerate launch   run_clm.py \
#     --dataset_name wikitext \
#     --dataset_config_name wikitext-2-raw-v1 \
#     --model_name_or_path gpt2 \
#     --output_dir ./logs/  \
#     --eval_steps 100 \
#     --learning_rate 5e-5 \
#     --block_size 1024 \
#     --do_ref_model \
#     --train_head_only \



# # ########



# # ########

# accelerate launch   run_clm.py \
#     --dataset_name wikitext \
#     --dataset_config_name wikitext-2-raw-v1 \
#     --model_name_or_path gpt2 \
#     --output_dir ./logs/  \
#     --eval_steps 100 \
#     --learning_rate 2e-5 \
#     --block_size 512 \
#     --per_device_train_batch_size 1 \
#     --do_ref_model \
#     --train_head_only \

# accelerate launch   run_clm.py \
#     --dataset_name wikitext \
#     --dataset_config_name wikitext-2-raw-v1 \
#     --model_name_or_path gpt2 \
#     --output_dir ./logs/  \
#     --eval_steps 100 \
#     --learning_rate 5e-5 \
#     --block_size 512 \
#     --per_device_train_batch_size 1 \
#     --do_ref_model \
#     --train_head_only \
#     --add_canary True \
#     --canary_len 10 \


# accelerate launch   run_clm.py \
#     --dataset_name wikitext \
#     --dataset_config_name wikitext-2-raw-v1 \
#     --model_name_or_path gpt2 \
#     --output_dir ./logs/  \
#     --eval_steps 100 \
#     --learning_rate 2e-5 \
#     --block_size 256 \
#     --per_device_train_batch_size 1 \
#     --do_ref_model \
#     --train_head_only \

# accelerate launch   run_clm.py \
#     --dataset_name wikitext \
#     --dataset_config_name wikitext-2-raw-v1 \
#     --model_name_or_path gpt2 \
#     --output_dir ./logs/  \
#     --eval_steps 100 \
#     --learning_rate 5e-5 \
#     --block_size 256 \
#     --per_device_train_batch_size 1 \
#     --do_ref_model \
#     --train_head_only \

#     #--add_adapter \

# accelerate launch   run_clm.py \
#     --dataset_name wikitext \
#     --dataset_config_name wikitext-2-raw-v1 \
#     --model_name_or_path gpt2 \
#     --output_dir ./logs/  \
#     --eval_steps 100 \
#     --learning_rate 2e-5 \
#     --block_size 128 \
#     --per_device_train_batch_size 1 \
#     --do_ref_model \
#     --train_head_only \

# accelerate launch   run_clm.py \
#     --dataset_name wikitext \
#     --dataset_config_name wikitext-2-raw-v1 \
#     --model_name_or_path gpt2 \
#     --output_dir ./logs/  \
#     --eval_steps 100 \
#     --learning_rate 5e-5 \
#     --block_size 128 \
#     --per_device_train_batch_size 1 \
#     --do_ref_model \
#     --train_head_only \

#     #--add_adapter \

# accelerate launch   run_clm.py \
#     --dataset_name wikitext \
#     --dataset_config_name wikitext-2-raw-v1 \
#     --model_name_or_path gpt2 \
#     --output_dir ./logs/  \
#     --eval_steps 100 \
#     --learning_rate 2e-5 \
#     --block_size 64 \
#     --per_device_train_batch_size 1 \
#     --do_ref_model \
#     --train_head_only \

# accelerate launch   run_clm.py \
#     --dataset_name wikitext \
#     --dataset_config_name wikitext-2-raw-v1 \
#     --model_name_or_path gpt2 \
#     --output_dir ./logs/  \
#     --eval_steps 100 \
#     --learning_rate 5e-5 \
#     --block_size 64 \
#     --per_device_train_batch_size 1 \
#     --do_ref_model \
#     --train_head_only \
    

# ####
# accelerate launch   run_clm.py \
#     --dataset_name wikitext \
#     --dataset_config_name wikitext-2-raw-v1 \
#     --model_name_or_path gpt2 \
#     --output_dir ./logs/  \
#     --eval_steps 100 \
#     --learning_rate 2e-5 \
#     --block_size 32 \
#     --per_device_train_batch_size 1 \
#     --do_ref_model \
#     --train_head_only \

# accelerate launch   run_clm.py \
#     --dataset_name wikitext \
#     --dataset_config_name wikitext-2-raw-v1 \
#     --model_name_or_path gpt2 \
#     --output_dir ./logs/  \
#     --eval_steps 100 \
#     --learning_rate 5e-5 \
#     --block_size 32 \
#     --per_device_train_batch_size 1 \
#     --do_ref_model \
#     --train_head_only \
#     #--add_adapter \






