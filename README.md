# Measuring Memorization in Language Model Fine-tuning


# Creating the Environment


```bash
conda create --name env --file requirments.txt
```

# File structure

In this repo we mainly focus on the task of next word prediction. The code is under the folder `gen'. The run_clm.py file is the main script that runs fine-tuning and memorization evaluations. Under gen you see ptb, enron and wikipedia directories, which contain the bash scripts to run experiments for different datasets, and different fine-tuning methods (full, head and adapters). The logs for each experiment will contain all the necessary metrics.

# Run Fine-tuning  and Evaluations

To run full fine-tuning on wikipedia data, and evaluate memorization using the membership inference attack run:

```bash
cd gen/wikipedia
bash  run_clm_full_ft_1gpu.sh 
```

If you want to fine-tune adapters, run:

```bash
cd gen/wikipedia
bash   run_clm_adapter_1gpu.sh 
```

And for fine-tuning the head run:

```bash
cd gen/wikipedia
bash run_clm_head_ft_1gpu.sh 
```

For running fine-tuning and evaluating memorization using the exposure metric, you can run:

```bash
cd gen/wikipedia
bash  run_clm_full_ft_canaries_1gpu.sh 
```

You can change the dataset/fine-tuning method to run the other experiments. 

# Extracting the Metrics from the Logs and Drawing Plots

To get the evaluation metrics and redraw our plotes, use the following Jupyter notebook:

```bash
cd gen
code plots.ipynb
```
