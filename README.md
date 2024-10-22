# Project Finetuning Llama

## Overview

This folder contains scripts to process datasets, fine-tune a LLaMA model, plot training loss, evaluate model performance and inference. 
The project is designed to facilitate the process of fine-tuning language models, tracking training progress, and assessing the final results.

## Contents

1. `llama-process_dataset.py`: Prepares and processes the dataset to be used for training and evaluation.
2. `finetune-llama.py`: Script to fine-tune a LLaMA model using the processed dataset.
3. `plot_loss.py`: Plots the training loss after each epoch to visualize model convergence.
4. `llama-evaluate.py`: Evaluates the trained model using ROUGE and BLEU.
5. `inference.py`: Performs inference using the trained model.


## Installation

To use this project, you need to have Python-3.10 installed. You also need to install the necessary dependencies using Anaconda or Miniconda:

```bash
# create the virtual environment
conda env create -f environment.yml
# activate the virtual environment
conda activate llm
```

## Usage

### Step 1: Download Model
```bash
# use mirror to download the pretrained model
export HF_ENDPOINT=https://hf-mirror.com
huggingface-cli download --resume-download shenzhi-wang/Llama3.1-8B-Chinese-Chat --local-dir chinese-model
```

### Step 2: Process Dataset
Prepare the dataset for training by running:

```bash
python process_dataset.py
```

### Step 3: Fine-Tune LLaMA
To fine-tune the model, execute:

```bash
python finetune-llama.py
```

### Step 4: Plot Training Loss
After training, plot the loss curve to visualize the model's convergence:

```bash
python plot_loss.py
```

### Step 5: Evaluate the Model
Finally, evaluate the model's performance using rouge and bleu:

```bash
python llama-evaluate.py
```

### Step 6: Inference the Model
Finally, inference the model using val_fold_1.json:

```bash
python inference.py
```
