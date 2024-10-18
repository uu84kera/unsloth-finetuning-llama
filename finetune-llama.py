from datasets import load_dataset
import argparse
from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported
from unsloth import FastLanguageModel
import torch

ruozhiba_prompt = """下面是一条描述任务的指令，并配有提供进一步背景信息的输入。请写出一个能适当完成要求的回复。
### 指令:
{}

### 输入:
{}

### 回复:
{}"""

# token
EOS_TOKEN = None
MAX_SEQ_LENGTH = 2048

# args define
parser = argparse.ArgumentParser()
parser.add_argument("--dataset_path", type=str, default="ruozhiba_processed.json")
parser.add_argument("--batch_size", type=int, default=32)

def get_model():
    """ this function is used to get the model """
    # load the model
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = "chinese-model", # here the model is shenzhi-wang/Llama3.1-8B-Chinese-Chat
        max_seq_length = MAX_SEQ_LENGTH,
        dtype = None,
        load_in_4bit = True,
    )
    # get the model
    model = FastLanguageModel.get_peft_model(
        model,
        r = 16, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj",],
        lora_alpha = 16,
        lora_dropout = 0, # Supports any, but = 0 is optimized
        bias = "none",    # Supports any, but = "none" is optimized
        # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
        use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
        random_state = 3407,
        use_rslora = False,  # We support rank stabilized LoRA
        loftq_config = None, # And LoftQ
    )
    return model, tokenizer

def formatting_prompts_func(examples):
    """ this function is used to format the prompts """
    instructions = examples["instruction"]
    inputs = examples["input"]
    outputs = examples["output"]
    texts = []
    for instruction, input, output in zip(instructions, inputs, outputs):
        # Must add EOS_TOKEN, otherwise your generation will go on forever!
        text = ruozhiba_prompt.format(instruction, input, output) + EOS_TOKEN
        texts.append(text)
    return {"text": texts,}

def finetune(args):
    """ finetune the model """
    # get the model
    model, tokenizer = get_model()
    global EOS_TOKEN
    EOS_TOKEN = tokenizer.eos_token

    # get dataset
    train_dataset = load_dataset("json", data_files="ruozhiba_processed_train1.json", split="train")
    train_dataset = train_dataset.map(formatting_prompts_func, batched=True)
    # evaluate
    val_dataset = load_dataset("json", data_files="ruozhiba_processed_val1.json", split="train")
    val_dataset = val_dataset.map(formatting_prompts_func, batched=True)
    # define trainer
    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = train_dataset,
        eval_dataset = val_dataset,
        dataset_text_field = "text",
        max_seq_length = MAX_SEQ_LENGTH,
        dataset_num_proc = 2,
        packing = False, # Can make training 5x faster for short sequences.
        args = TrainingArguments(
            per_device_train_batch_size = args.batch_size,
            gradient_accumulation_steps = 4,
            warmup_steps = 5,
            num_train_epochs = 3, # Set this for 1 full training run.
            learning_rate = 2e-4,
            fp16 = not is_bfloat16_supported(),
            bf16 = is_bfloat16_supported(),
            logging_steps = 10,
            optim = "adamw_8bit",
            weight_decay = 0.01,
            lr_scheduler_type = "linear",
            seed = 3407,
            output_dir = "checkpoints_llama_v2",
            save_total_limit = 3,
            save_steps = 10,
            evaluation_strategy = "steps",
            eval_steps = 10,
            save_strategy="steps",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            logging_dir="logs",
            report_to = "tensorboard"
        ),
    )
    # start to fine tune
    trainer_stats = trainer.train()

if __name__ == "__main__":
    # get args
    args = parser.parse_args()
    finetune(args)