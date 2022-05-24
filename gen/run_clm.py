#!/usr/bin/env python
# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for causal language modeling (GPT, GPT-2, CTRL, ...)
on a text file or a dataset without using HuggingFace Trainer.

Here is the full list of checkpoints on the hub that can be fine-tuned by this script:
https://huggingface.co/models?filter=text-generation
"""
# You can also adapt this script on your own causal language modeling task. Pointers for this are left as comments.

import argparse
from enum import unique
import logging
import math
import os
import random
from itertools import chain
from pathlib import Path
import copy 
from sys import path
import sys
from utils import Logger


import datasets
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import transformers
from accelerate import Accelerator, DistributedType
from huggingface_hub import Repository
from transformers import (
#    CONFIG_MAPPING,
#    MODEL_MAPPING,
    AdamW,
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    SchedulerType,
    default_data_collator,
    get_scheduler,
    set_seed,
)

#from torch import AdamW
from transformers.utils.versions import require_version
import datasets
from datasets import load_dataset
from random import shuffle
import numpy as np
import random
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import csv
from scipy.stats import skewnorm
from scipy.stats import kstest


transformers.logging.set_verbosity_error()

logger = logging.getLogger(__name__)

require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/language-modeling/requirements.txt")

#MODEL_CONFIG_CLASSES = list(MODEL_MAPPING.keys())
#MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)

# Disable
def blockPrint():
    sys.stdout = open(os.devnull, 'w')

# Restore
def enablePrint():
    sys.stdout = sys.__stdout__
    
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a causal language modeling task")
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help="The name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The configuration name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--train_file", type=str, default=None, help="A csv or a json file containing the training data."
    )
    parser.add_argument(
        "--validation_file", type=str, default=None, help="A csv or a json file containing the validation data."
    )
    parser.add_argument(
        "--validation_split_percentage",
        default=5,
        help="The percentage of the train set used as validation set in case there's no validation split",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=True,
    )
    parser.add_argument(
        "--config_name",
        type=str,
        default=None,
        help="Pretrained config name or path if not the same as model_name",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--use_slow_tokenizer",
        action="store_true",
        help="If passed, will use a slow tokenizer (not backed by the 🤗 Tokenizers library).",
    )
    parser.add_argument(
        "--do_ref_model",
        action="store_true",
        help="If passed, will use a slow tokenizer (not backed by the 🤗 Tokenizers library).",
    )
    parser.add_argument(
        "--add_adapter",
        action="store_true",
        help="whether to add adapter or not",
    )
    parser.add_argument(
        "--adapter_reduction",
        type=int,
        default=None,
        help="whether to add adapter or not",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=1,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=2,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay to use.")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Total number of training epochs to perform.")
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--eval_steps",
        type=int,
        default=50,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="linear",
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument(
        "--num_warmup_steps", type=int, default=0, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument("--output_dir", type=str, default=None, help="Where to store the final model.")
    parser.add_argument("--seed", type=int, default=1234, help="A seed for reproducible training.")
    parser.add_argument(
        "--model_type",
        type=str,
        default=None,
        help="Model type to use if training from scratch.",
   #     choices=MODEL_TYPES,
    )
    parser.add_argument(
        "--block_size",
        type=int,
        default=None,
        help="Optional input sequence length after tokenization. The training dataset will be truncated in block of this size for training. Default to the model max input length for single sentence inputs (take into account special tokens).",
    )
    parser.add_argument(
        "--preprocessing_num_workers",
        type=int,
        default=None,
        help="The number of processes to use for the preprocessing.",
    )
    parser.add_argument(
        "--overwrite_cache", type=bool, default=False, help="Overwrite the cached training and evaluation sets"
    )
    parser.add_argument(
        "--no_keep_linebreaks", action="store_true", help="Do not keep line breaks when using TXT files."
    )
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument(
        "--hub_model_id", type=str, help="The name of the repository to keep in sync with the local `output_dir`."
    )
    parser.add_argument("--hub_token", type=str, help="The token to use to push to the Model Hub.")
    parser.add_argument("--add_canary", action="store_true", help = "If true, then add canaries in the dataset.")
    parser.add_argument("--canary_rep", default=None, type = int, help = "The repetition of each canary")
    parser.add_argument("--canary_len", default = 5, type = int, help = "The len of digit of canaries")
    parser.add_argument("--train_head_only",action="store_true", help = "If true, freeze all the layers except the head of the model.")
    parser.add_argument("--train_layer_n_only",default=None, type=int,  help = "If true, freeze all the layers except the n'th layer of the model.")
 
    args = parser.parse_args()

    # Sanity checks
    if args.dataset_name is None and args.train_file is None and args.validation_file is None:
        raise ValueError("Need either a dataset name or a training/validation file.")
    else:
        if args.train_file is not None:
            extension = args.train_file.split(".")[-1]
            assert extension in ["csv", "json", "txt"], "`train_file` should be a csv, json or txt file."
        if args.validation_file is not None:
            extension = args.validation_file.split(".")[-1]
            assert extension in ["csv", "json", "txt"], "`validation_file` should be a csv, json or txt file."

 
    return args

def get_exposure(fitting, main):

    fitting_params = skewnorm.fit(fitting)
    ks = kstest(fitting, 'skewnorm', fitting_params)

    cdf = skewnorm.cdf(main, fitting_params[0], fitting_params[1], fitting_params[2])

    
    if cdf == 0.0:
        exposure = 0.0
    else:
        exposure = -1.0*np.log2(cdf)
    
    
    return exposure

def get_fit_canary_loss(model,fitting_id, main_id):
    loss_list = []
    for k, v in main_id.items():
            main_id[k] = torch.tensor(v).cuda()
                  
    loss_main = np.exp(model(**main_id)['loss'].item())

    for sample in fitting_id:
        for k, v in sample.items():
            sample[k] = torch.tensor(v).cuda()
        
        output = model(**sample)
        loss_list.append(np.exp(output.loss.item()))

    return loss_main,loss_list

def gen_canary(canary_len,tokenizer):
        raw_sample = random.choices([str(i) for i in range(10)], k=canary_len)
        raw_sample = " ".join(raw_sample)
        
        tokenized = tokenizer.tokenize(raw_sample)
        ids = tokenizer.convert_tokens_to_ids(tokenized)
        assert len(ids) == canary_len
        
        raw_sample = "the secret number is " + raw_sample
        toked =  tokenizer(raw_sample)
        toked['labels'] = toked['input_ids'].copy()
        return raw_sample, toked
            
def main():

    args = parse_args()
    random.seed(args.seed)


    folder_name = f"canary_{str(args.canary_rep)}_{str(args.canary_len)}_adapter_{args.add_adapter}_head_{args.train_head_only}_layer_{args.train_layer_n_only}_ref_{args.do_ref_model}_maxlen_{args.block_size}_red_{args.adapter_reduction}_model_{args.model_name_or_path}_lr_{args.learning_rate}_epoch_{args.num_train_epochs}_trba_{args.per_device_train_batch_size}_acc_{args.gradient_accumulation_steps}_evba{args.per_device_eval_batch_size}_data_{args.dataset_name}"
    
    directory = "{}/{}".format(args.output_dir,folder_name)
    if not os.path.exists(directory):
        os.mkdir(directory)
    
    log_file = os.path.join(directory, "stdout")


    # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
    accelerator = Accelerator()
    
    if accelerator.is_local_main_process:
        print("Logging to {}".format(log_file))
        
    sys.stdout = Logger(log_file)

        
    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    #logger.info(accelerator.state)

    # Setup logging, we only want one process per machine to log things on the screen.
    # accelerator.is_local_main_process is only True for one process per machine.
    logger.setLevel(logging.INFO if accelerator.is_local_main_process else logging.ERROR)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    directory = os.path.join(directory,"model")

    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(directory, exist_ok=True)
    accelerator.wait_for_everyone()
    
    if accelerator.is_local_main_process:
       print(str(args))
    # Get the datasets: you can either provide your own CSV/JSON/TXT training and evaluation files (see below)
    # or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
    # (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For CSV/JSON files, this script will use the column called 'text' or the first column if no column called
    # 'text' is found. You can easily tweak this behavior (see below).
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    if args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        if 'enron' in args.dataset_name:
            raw_datasets =   load_dataset('csv', data_files={'train': 'data/cleaned_short_train_scrubbed.csv' ,'validation': 'data/cleaned_short_test_scrubbed.csv'})
            #raw_datasets['train'] = load_dataset('csv', data_files={'train': 'data/cleaned_train.csv' ,'validation': 'data/cleaned_test.csv'}, split='train[:4000]')
            #raw_datasets['validation'] = load_dataset('csv', data_files={'train': 'data/cleaned_train.csv' ,'validation': 'data/cleaned_test.csv'}, split='train[4000:5000]')

        else:
            raw_datasets = load_dataset(args.dataset_name, args.dataset_config_name)
            if "validation" not in raw_datasets.keys():
                raw_datasets["validation"] = load_dataset(
                    args.dataset_name,
                    args.dataset_config_name,
                    split=f"train[:{args.validation_split_percentage}%]",
                )
                raw_datasets["train"] = load_dataset(
                    args.dataset_name,
                    args.dataset_config_name,
                    split=f"train[{args.validation_split_percentage}%:]",
                )
                
            

        
    else:
        data_files = {}
        dataset_args = {}
        if args.train_file is not None:
            data_files["train"] = args.train_file
        if args.validation_file is not None:
            data_files["validation"] = args.validation_file
        extension = args.train_file.split(".")[-1]
        if extension == "txt":
            extension = "text"
            dataset_args["keep_linebreaks"] = not args.no_keep_linebreaks
        raw_datasets = load_dataset(extension, data_files=data_files, **dataset_args)
        # If no validation data is there, validation_split_percentage will be used to divide the dataset.
        if "validation" not in raw_datasets.keys():
            raw_datasets["validation"] = load_dataset(
                extension,
                data_files=data_files,
                split=f"train[:{args.validation_split_percentage}%]",
                **dataset_args,
            )
            raw_datasets["train"] = load_dataset(
                extension,
                data_files=data_files,
                split=f"train[{args.validation_split_percentage}%:]",
                **dataset_args,
            )


    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    # Load pretrained model and tokenizer
    #
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    if args.config_name:
        config = AutoConfig.from_pretrained(args.config_name)
    elif args.model_name_or_path:
        config = AutoConfig.from_pretrained(args.model_name_or_path)
#    else:
#        config = CONFIG_MAPPING[args.model_type]()
#        logger.warning("You are instantiating a new config instance from scratch.")

    if args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, use_fast=not args.use_slow_tokenizer)
    elif args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=not args.use_slow_tokenizer)
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )

    if args.model_name_or_path:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            config=config,
        )
    else:
        logger.info("Training new model from scratch")
        model = AutoModelForCausalLM.from_config(config)

    model.resize_token_embeddings(len(tokenizer))
    
    model_ref = copy.deepcopy(model)


    if args.add_canary:    
        if 'ptb' in args.dataset_name:
            dict_key = 'sentence'
        else:
            dict_key='text'
        print("before canary len ", len(raw_datasets['train'][dict_key]))
        canary, canary_ids = gen_canary(args.canary_len, tokenizer)
        for j in range(args.canary_rep):
            raw_datasets['train']=raw_datasets['train'].add_item({dict_key:canary})

        raw_datasets['train'] = raw_datasets['train'].shuffle(seed=args.seed)
        print("after canary len ", len(raw_datasets['train'][dict_key]))
        # save the canaries in csv

        file = open(f'./{directory}/canaries.txt', 'w+')
        file.write(canary)
        file.write('\n')
        file.close()

        file = open(f'./{directory}/fitting_canaries.txt', 'w+')
        
        fitting_canaries_ids = []
        for i in range(5000):
            fit , fit_ids = gen_canary(args.canary_len,tokenizer)
            if fit != canary:
                fitting_canaries_ids.append(fit_ids)
                file.write(fit)
                file.write('\n')
        print(len(fitting_canaries_ids))
            


    # Preprocessing the datasets.
    # First we tokenize all the texts.
    column_names = raw_datasets["train"].column_names
    text_column_name = "text" if "text" in column_names else column_names[0]

    def tokenize_function(examples):
        return tokenizer(examples[text_column_name])

    with accelerator.main_process_first():
        tokenized_datasets = raw_datasets.map(
            tokenize_function,
            batched=True,
            num_proc=args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not args.overwrite_cache,
            desc="Running tokenizer on dataset",
        )

    if args.block_size is None:
        block_size = tokenizer.model_max_length
        if block_size > 1024:
            logger.warning(
                f"The tokenizer picked seems to have a very large `model_max_length` ({tokenizer.model_max_length}). "
                "Picking 1024 instead. You can change that default value by passing --block_size xxx."
            )
        block_size = 1024
    else:
        if args.block_size > tokenizer.model_max_length:
            logger.warning(
                f"The block_size passed ({args.block_size}) is larger than the maximum length for the model"
                f"({tokenizer.model_max_length}). Using block_size={tokenizer.model_max_length}."
            )
        block_size = min(args.block_size, tokenizer.model_max_length)

    # Main data processing function that will concatenate all texts from our dataset and generate chunks of block_size.
    def group_texts(examples):
        # Concatenate all texts.
        
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
        if total_length >= block_size:
            total_length = (total_length // block_size) * block_size
        # Split by chunks of max_len.
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    # Note that with `batched=True`, this map processes 1,000 texts together, so group_texts throws away a remainder
    # for each of those groups of 1,000 texts. You can adjust that batch_size here but a higher value might be slower
    # to preprocess.
    #
    # To speed up this part, we use multiprocessing. See the documentation of the map method for more information:
    # https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.map

    with accelerator.main_process_first():
        lm_datasets = tokenized_datasets.map(
            group_texts,
            batched=True,
            num_proc=args.preprocessing_num_workers,
            load_from_cache_file=not args.overwrite_cache,
            desc=f"Grouping texts in chunks of {block_size}",
        )

    train_dataset = lm_datasets["train"]
    eval_dataset = lm_datasets["validation"]
    
    
    #for i in range(len(train_dataset)):
    #    print(tokenizer.decode(train_dataset[i]['labels']))
    
    #exit()



    # Log a few random samples from the training set:
    #for index in random.sample(range(len(train_dataset)), 3):
    #    logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

    # DataLoaders creation:
    train_dataloader = DataLoader(
        train_dataset, shuffle=True, collate_fn=default_data_collator, batch_size=args.per_device_train_batch_size
    )
    eval_dataloader = DataLoader(
        eval_dataset, collate_fn=default_data_collator, batch_size=args.per_device_eval_batch_size
    )


    if args.add_adapter:
        # add new adapter
            if (args.adapter_reduction is not None):
                    model.add_adapter("wiki", config={
                                    "ln_after": False,
                                    "ln_before": False,
                                    "mh_adapter": False,
                                    "output_adapter": True,
                                    "adapter_residual_before_ln": False,
                                    "non_linearity": "relu",
                                    "original_ln_after": True,
                                    "original_ln_before": True,
                                    "reduction_factor": args.adapter_reduction,
                                    "residual_before_ln": True}
                                )
            else: #default, pfeiffer
                model.add_adapter("wiki", config={ #was "FTL"
                                    "ln_after": False,
                                    "ln_before": False,
                                    "mh_adapter": False,
                                    "output_adapter": True,
                                    "adapter_residual_before_ln": False,
                                    "non_linearity": "relu",
                                    "original_ln_after": True,
                                    "original_ln_before": True,
                                    "reduction_factor": 16,
                                    "residual_before_ln": True})
        # activate adapter for training
            model.train_adapter("wiki")

    if args.train_head_only:
        for params in model.parameters():
            params.requires_grad = False

        for param in model.lm_head.parameters():
            param.requires_grad = True
        
    elif args.train_layer_n_only is not None:
        n = args.train_layer_n_only
        k = 0
        for params in model.parameters():
                params.requires_grad = False
        
        for params in model.transformer.h[n].parameters():
                params.requires_grad = True

                
    
    #print(model.lm_head)    
    if accelerator.is_local_main_process:
        print("model_params (million)", count_parameters(model)/1000000)



    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)





    # Prepare everything with our `accelerator`.
    model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader
    )

    model_ref = accelerator.prepare(
        model_ref
    )

    # On TPU, the tie weights in our model have been disconnected, so we need to restore the ties.
    if accelerator.distributed_type == DistributedType.TPU:
        model.tie_weights()

    # Note -> the training dataloader needs to be prepared before we grab his length below (cause its length will be
    # shorter in multiprocess)

    # Scheduler and math around the number of training steps.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    else:
        args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps=args.max_train_steps,
    )

    # Train!
    total_batch_size = args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    
# Here freezing of the model except the last layer i.e the head is performed
    if accelerator.is_local_main_process:
        print("model_params (million)", count_parameters(model)/1000000)



    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    # Only show the progress bar once on each machine.
    #progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    completed_steps = 0
    best_loss = 1000000
    for epoch in range(args.num_train_epochs):
        model.train()
        if accelerator.is_local_main_process:
            print(f"training epoch {epoch}")
        for step, batch in enumerate(train_dataloader):
            outputs = model(**batch)
            loss = outputs.loss
            loss = loss / args.gradient_accumulation_steps
            accelerator.backward(loss)
            if step % args.gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
       #         progress_bar.update(1)
                completed_steps += 1

        
                
                # if completed_steps % args.eval_steps == 0:
                #         model.eval()
                #         losses = []
                #         for step, batch in enumerate(eval_dataloader):
                #             with torch.no_grad():
                #                 outputs = model(**batch)

                #             loss = outputs.loss
                #             losses.append(accelerator.gather(loss.repeat(args.per_device_eval_batch_size)))
                            

                #         losses = torch.cat(losses)
                #         losses = losses[: len(eval_dataset)]
                #         try:
                #             perplexity = math.exp(torch.mean(losses))
                #         except OverflowError:
                #             perplexity = float("inf")
                        
                #         if torch.mean(losses) < best_loss:
                #             best_loss=torch.mean(losses)
                #             if accelerator.is_local_main_process:
                #                 print(f"saving model here at step {completed_steps} and epoch {epoch} with ppl {perplexity}")
                            
                #             if args.output_dir is not None:
                #                 accelerator.wait_for_everyone()
                #                 unwrapped_model = accelerator.unwrap_model(model)
                #                 unwrapped_model.save_pretrained(directory, save_function=accelerator.save)
                #                 if accelerator.is_main_process:
                #                     tokenizer.save_pretrained(directory)   
                            
                #         if accelerator.is_local_main_process:
                #             print(f"step {completed_steps} epoch {epoch} perplexity: {perplexity}")
                #         #exit()
                #         model.train()
            
            if completed_steps >= args.max_train_steps:
                break   
        model.eval()
        losses = []
        if accelerator.is_local_main_process:
            print(f"*************end of epoch {epoch} eval ")
        
        if args.add_canary:
            print("running canary eval")
            canary_loss, fitting_loss = get_fit_canary_loss(model,fitting_canaries_ids,canary_ids)        
            exposure = get_exposure(fitting_loss,canary_loss)
            print(exposure)
        
        if args.do_ref_model:
            model_ref.eval()
            losses_ref = []
            
        for step, batch in enumerate(eval_dataloader):
            with torch.no_grad():
                outputs = model(**batch)
                

            loss = outputs.loss
            losses.append(accelerator.gather(loss.repeat(args.per_device_eval_batch_size)))
            
            if args.do_ref_model:
                with torch.no_grad():
                    outputs_ref =model_ref(**batch)
                loss_ref = outputs_ref.loss
                losses_ref.append(accelerator.gather(loss_ref.repeat(args.per_device_eval_batch_size)))
                
            

        losses = torch.cat(losses)
        losses = losses[: len(eval_dataset)]
     
        
        if args.do_ref_model:
            losses_ref = torch.cat(losses_ref)
            losses_ref = losses_ref[: len(eval_dataset)]
            sorted_ratio = sorted([l/l_ref for l,l_ref in zip (losses,losses_ref)])
        
        sorted_loss = sorted(losses)
        
        if args.do_ref_model:
            threshold_ref = sorted_ratio[int(0.1*len(sorted_ratio))]
            threshold = sorted_loss[int(0.1*len(losses))]
            if accelerator.is_local_main_process:
                print("threshold_ref is: " , threshold_ref.detach().item())
                print("threshold is: " , threshold.detach().item())
        else:
            threshold = sorted_loss[int(0.1*len(losses))]
            if accelerator.is_local_main_process:
                print("threshold is: " , threshold.detach().item())
        try:
            perplexity = math.exp(torch.mean(losses))
        except OverflowError:
            perplexity = float("inf")
        
        # if torch.mean(losses) < best_loss:
        #     best_loss=torch.mean(losses)
        #     if accelerator.is_local_main_process:
        #         print(f"saving model here at step {completed_steps} and epoch {epoch} with ppl {perplexity}")
                
        #         if args.output_dir is not None:
        #             if accelerator.is_main_process:
        #                 os.makedirs(directory+f"/best", exist_ok=True)
        #             accelerator.wait_for_everyone()
        #             unwrapped_model = accelerator.unwrap_model(model)
        #             unwrapped_model.save_pretrained(directory+f"/best", save_function=accelerator.save)

                    
        #if args.output_dir is not None:
        #    if accelerator.is_main_process:
        #        os.makedirs(directory+f"/epoch_{epoch}", exist_ok=True)
        #    accelerator.wait_for_everyone()
        #    unwrapped_model = accelerator.unwrap_model(model)
        #    unwrapped_model.save_pretrained(directory+f"/epoch_{epoch}", save_function=accelerator.save)
            #if accelerator.is_main_process:
                #    tokenizer.save_pretrained(directory)   
          
        ################################################    
        #run threshold on training samples
        losses = []
        if args.do_ref_model:
            model_ref.eval()
            losses_ref = []
            
        for step, batch in enumerate(train_dataloader):
            with torch.no_grad():
                outputs = model(**batch)

            loss = outputs.loss
            
            losses.append(accelerator.gather(loss.repeat(args.per_device_train_batch_size)))
            if args.do_ref_model:
                with torch.no_grad():
                    outputs_ref =model_ref(**batch)
                loss_ref = outputs_ref.loss
                losses_ref.append(accelerator.gather(loss_ref.repeat(args.per_device_train_batch_size)))
              
              
        

        accelerator.wait_for_everyone()
        losses = torch.cat(losses)
        losses = losses[: len(train_dataset)]

        
        if args.do_ref_model:
            losses_ref = torch.cat(losses_ref)
            losses_ref = losses_ref[: len(train_dataset)]
            lr_rat = [l/l_r for l,l_r in zip(losses,losses_ref)]
            
        if args.do_ref_model:
            guess_cor = sum([1 for sample in losses if sample<threshold])
            guess_cor_ref =  sum([1 for sample in lr_rat if sample<threshold_ref])
        else:    
            guess_cor = sum([1 for sample in losses if sample<threshold])


        
        try:
            perplexity_train = math.exp(torch.mean(losses))
        except OverflowError:
            perplexity_train = float("inf")
        #assert(len(losses)==len(lr_rat))
        if accelerator.is_local_main_process:
            if args.do_ref_model:
                print("correct cnt  ref is: " , guess_cor_ref, "all is: ", len(losses), "ratio is: ", guess_cor_ref/len(losses))
            print("correct cnt is: " , guess_cor, "all is: ", len(losses), "ratio is: ", guess_cor/len(losses))
            print(f"epoch {epoch}: perplexity: {perplexity} perplexity_train: {perplexity_train}")
            print("____")
            if args.do_ref_model:
                print(f"{guess_cor_ref/len(losses)}\n{guess_cor/len(losses)}\n{perplexity}\n{perplexity_train}")
                ratio = len(train_dataset)/len(eval_dataset)
                guess_cor_subsampled = sum([1 for sample in losses[::int(ratio)] if sample<threshold])
                guess_cor_ref_subsampled =  sum([1 for sample in lr_rat[::int(ratio)] if sample<threshold_ref])                
                print(f"{guess_cor_ref_subsampled/(len(lr_rat[::int(ratio)]))}\n{guess_cor_subsampled/len(losses[::int(ratio)])}\n{guess_cor_ref_subsampled/(guess_cor_ref_subsampled+int(0.1*len(eval_dataset)))}\n{guess_cor_subsampled/(guess_cor_subsampled+int(0.1*len(eval_dataset)))}")

            else:
                print(f"{guess_cor/len(losses)}\n{perplexity}\n{perplexity_train}")
            print("_____")

    model.eval()
    losses = []
    if accelerator.is_local_main_process:
        print(f"*************end of training ")
    
    if args.add_canary:
            print("running canary eval")
            canary_loss, fitting_loss = get_fit_canary_loss(model,fitting_canaries_ids,canary_ids)        
            exposure = get_exposure(fitting_loss,canary_loss)
            print(exposure)    
    
    if args.do_ref_model:
        model_ref.eval()
        losses_ref = []
        
    for step, batch in enumerate(eval_dataloader):
        with torch.no_grad():
            outputs = model(**batch)
            

        loss = outputs.loss
        losses.append(accelerator.gather(loss.repeat(args.per_device_eval_batch_size)))
        
        if args.do_ref_model:
            with torch.no_grad():
                outputs_ref =model_ref(**batch)
            loss_ref = outputs_ref.loss
            losses_ref.append(accelerator.gather(loss_ref.repeat(args.per_device_eval_batch_size)))
            
        

    losses = torch.cat(losses)
    losses = losses[: len(eval_dataset)]
    
    if args.do_ref_model:
        losses_ref = torch.cat(losses_ref)
        losses_ref = losses_ref[: len(eval_dataset)]
        sorted_ratio = sorted([l/l_ref for l,l_ref in zip (losses,losses_ref)])
    
    sorted_loss = sorted(losses)
    
    if args.do_ref_model:
        threshold_ref = sorted_ratio[int(0.1*len(sorted_ratio))]
        threshold = sorted_loss[int(0.1*len(losses))]
        if accelerator.is_local_main_process:
            print("threshold_ref is: " , threshold_ref.detach().item())
            print("threshold is: " , threshold.detach().item())
    else:
        threshold = sorted_loss[int(0.1*len(losses))]
        if accelerator.is_local_main_process:
            print("threshold is: " , threshold.detach().item())
    try:
        perplexity = math.exp(torch.mean(losses))
    except OverflowError:
        perplexity = float("inf")
    

    
        
    #run threshold on training samples
    losses = []
    if args.do_ref_model:
        model_ref.eval()
        losses_ref = []
        
    for step, batch in enumerate(train_dataloader):
        with torch.no_grad():
            outputs = model(**batch)

        loss = outputs.loss
        losses.append(accelerator.gather(loss.repeat(args.per_device_train_batch_size)))
        
        if args.do_ref_model:
            with torch.no_grad():
                outputs_ref =model_ref(**batch)
            loss_ref = outputs_ref.loss
            losses_ref.append(accelerator.gather(loss_ref.repeat(args.per_device_train_batch_size)))
                        

    accelerator.wait_for_everyone()
    
    losses = torch.cat(losses)

    
    losses = losses[: len(train_dataset)]
    
    
    if args.do_ref_model:
        losses_ref = torch.cat(losses_ref)
        losses_ref = losses_ref[: len(train_dataset)]
        lr_rat = [l/l_r for l,l_r in zip(losses,losses_ref)]
        
    if args.do_ref_model:
        guess_cor = sum([1 for sample in losses if sample<threshold])
        guess_cor_ref =  sum([1 for sample in lr_rat if sample<threshold_ref])
    else:    
        guess_cor = sum([1 for sample in losses if sample<threshold])


    
    try:
        perplexity_train = math.exp(torch.mean(losses))
    except OverflowError:
        perplexity_train = float("inf")

    if accelerator.is_local_main_process:
        if args.do_ref_model:
            print("correct cnt  ref is: " , guess_cor_ref, "all is: ", len(losses), "ratio is: ", guess_cor_ref/len(losses))
        print("correct cnt is: " , guess_cor, "all is: ", len(losses), "ratio is: ", guess_cor/len(losses))
        print(f"end of training perplexity: {perplexity} perplexity_train: {perplexity_train}")
        print("____")
        if args.do_ref_model:
                print(f"{guess_cor_ref/len(losses)}\n{guess_cor/len(losses)}\n{perplexity}\n{perplexity_train}")
                ratio = len(train_dataset)/len(eval_dataset)
                guess_cor_subsampled = sum([1 for sample in losses[::int(ratio)] if sample<threshold])
                guess_cor_ref_subsampled =  sum([1 for sample in lr_rat[::int(ratio)] if sample<threshold_ref])                
                print(f"{guess_cor_ref_subsampled/(len(lr_rat[::int(ratio)]))}\n{guess_cor_subsampled/len(losses[::int(ratio)])}\n{guess_cor_ref_subsampled/(guess_cor_ref_subsampled+int(0.1*len(eval_dataset)))}\n{guess_cor_subsampled/(guess_cor_subsampled+int(0.1*len(eval_dataset)))}")

        else:
            print(f"{guess_cor/len(losses)}\n{perplexity}\n{perplexity_train}")
        print("_____")


if __name__ == "__main__":
    main()