

accelerate launch   run_clm.py \
    --dataset_name wikitext \
    --dataset_config_name wikitext-2-raw-v1 \
    --model_name_or_path gpt2 \
    --output_dir ./logs/  \
    --eval_steps 100 \
    --learning_rate 5e-6 \
    --block_size 1024 \
    --do_ref_model \





accelerate launch   run_clm.py \
    --dataset_name wikitext \
    --dataset_config_name wikitext-2-raw-v1 \
    --model_name_or_path gpt2 \
    --output_dir ./logs/  \
    --eval_steps 100 \
    --learning_rate 5e-6 \
    --block_size 32 \
    --per_device_train_batch_size 1 \
    --do_ref_model \
    #--add_adapter \


########



#########

accelerate launch   run_clm.py \
    --dataset_name wikitext \
    --dataset_config_name wikitext-2-raw-v1 \
    --model_name_or_path gpt2 \
    --output_dir ./logs/  \
    --eval_steps 100 \
    --learning_rate 5e-6 \
    --block_size 512 \
    --per_device_train_batch_size 1 \
    --do_ref_model \
    #--add_adapter \


accelerate launch   run_clm.py \
    --dataset_name wikitext \
    --dataset_config_name wikitext-2-raw-v1 \
    --model_name_or_path gpt2 \
    --output_dir ./logs/  \
    --eval_steps 100 \
    --learning_rate 5e-6 \
    --block_size 256 \
    --per_device_train_batch_size 1 \
    --do_ref_model \
    #--add_adapter \

accelerate launch   run_clm.py \
    --dataset_name wikitext \
    --dataset_config_name wikitext-2-raw-v1 \
    --model_name_or_path gpt2 \
    --output_dir ./logs/  \
    --eval_steps 100 \
    --learning_rate 5e-6 \
    --block_size 128 \
    --per_device_train_batch_size 1 \
    --do_ref_model \
    #--add_adapter \

accelerate launch   run_clm.py \
    --dataset_name wikitext \
    --dataset_config_name wikitext-2-raw-v1 \
    --model_name_or_path gpt2 \
    --output_dir ./logs/  \
    --eval_steps 100 \
    --learning_rate 5e-6 \
    --block_size 64 \
    --per_device_train_batch_size 1 \
    --do_ref_model \
    

####






