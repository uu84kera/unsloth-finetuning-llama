# unsloth-finetuning-llama

## Overview

This repository contains scripts to process datasets, download necessary files, fine-tune a LLaMA model, plot training loss, and evaluate model performance. The project is designed to facilitate the process of fine-tuning language models, tracking training progress, and assessing the final results.

## Contents

1. `process_dataset.py`: Prepares and processes the dataset to be used for training and evaluation.
2. `download.py`: Downloads required data or model files from specified URLs.
3. `finetune-llama.py`: Script to fine-tune a LLaMA model using the processed dataset.
4. `plot_loss.py`: Plots the training loss after each epoch to visualize model convergence.
5. `evaluate.py`: Evaluates the trained model using a set of pre-defined metrics.

## Installation

To use this project, you need to have Python installed. You also need to install the necessary dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Step 1: Download Model
Run the `download.py` script to download the necessary files:

```bash
python download.py
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
Finally, evaluate the model's performance using:

```bash
python evaluate.py
```

## Requirements
- Python 3.8+
- PyTorch
- Matplotlib
- Other dependencies listed in `requirements.txt`

## Contributing
If you'd like to contribute to this project, please fork the repository and submit a pull request.

## License
This project is licensed under the MIT License. See `LICENSE` for more details.
