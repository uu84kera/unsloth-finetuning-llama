import json
import torch
from transformers import AutoTokenizer
from unsloth import FastLanguageModel

ruozhiba_prompt = """\
下面是一条描述任务的指令，并配有提供进一步背景信息的输入。请写出一个能适当完成要求的回复。
### 指令:
{}

### 输入:
{}

### 回复:
{}"""

MAX_SEQ_LENGTH = 2048
MODEL_PATH = "checkpoints_llama_v2/checkpoint-78"
DATA_PATH = "ruozhiba_processed_val1.json"
OUTPUT_PATH = "answers.json"

# Load the model and tokenizer
def get_model():
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_PATH,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=None,
        load_in_4bit=True,
    )
    # Enable LoRA optimization
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
    )
    return model, tokenizer

# Inference
def inference(model, tokenizer, input_texts):
    model = FastLanguageModel.for_inference(model)
    inputs = tokenizer(input_texts, return_tensors="pt", padding=True, truncation=True).to("cuda")
    outputs = model.generate(**inputs, max_new_tokens=64, use_cache=True)
    return tokenizer.batch_decode(outputs, skip_special_tokens=True)

if __name__ == "__main__":
    # Load the model
    model, tokenizer = get_model()

    # Load the dataset
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    answers = []

    # Iterate over each entry in the dataset
    for entry in dataset:
        instruction = entry["instruction"]
        input_data = entry["input"]
        formatted_prompt = ruozhiba_prompt.format(instruction, input_data, "")

        # Run inference
        output = inference(model, tokenizer, [formatted_prompt])[0]

        # Store the question and answer
        answers.append({"instruction": instruction, "input": input_data, "output": output})

    # Save the answers to a new JSON file
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(answers, f, ensure_ascii=False, indent=4)

    print(f"Inference completed. Results saved to {OUTPUT_PATH}")
