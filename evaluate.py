from datasets import load_dataset
import argparse
from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported
from unsloth import FastLanguageModel
import evaluate
import torch
from tqdm import tqdm
import pickle
from rouge_score import rouge_scorer
from rouge_chinese import Rouge
import jieba
from nltk.translate.bleu_score import sentence_bleu

ruozhiba_prompt = """\
下面是一条描述任务的指令，并配有提供进一步背景信息的输入。请写出一个能适当完成要求的回复。
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
parser.add_argument("--batch_size", type=int, default=32)

def get_model():
    """ this function is used to get the model """
    # load the model
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = "checkpoints_llama_v2/checkpoint-78", 
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

def evaluation(args):
    """ evaluate the model """
    # get the model
    model, tokenizer = get_model()
    FastLanguageModel.for_inference(model)
    global EOS_TOKEN
    EOS_TOKEN = tokenizer.eos_token
    val_dataset = load_dataset("json", data_files="ruozhiba_processed_val1.json", split="train")
    val_dataset = val_dataset.map(formatting_prompts_func, batched=True)
    predictions = []
    references = val_dataset["output"]
    index = 0
    for input in tqdm(val_dataset["input"]):
        inputs = tokenizer([ruozhiba_prompt.format("请替我解答以下句子的真实意义，并且以尽可能简洁的语言回答.", input, "")], return_tensors = "pt", padding=True, truncation=True).to("cuda")
        output_tokens = model.generate(**inputs, max_new_tokens=256)
        # Decode the generated tokens into text
        prediction = tokenizer.decode(output_tokens[0], skip_special_tokens=True)
        prediction = prediction.split("### 回复:")[-1].strip()
        # clean output!
        prediction = prediction.replace("assistant", "").strip()
        predictions.append(prediction)
        index += 1 
    results = {"reference": references, "prediction": predictions}
    with open("results.pkl", "wb") as f:
        pickle.dump(results, f)
    return results

def compute_metrics(results):
    """ this function is used to compute the evaluation metrics """
    predictions = results["prediction"]
    references = results["reference"]
    rouge_scores = []
    rouge = Rouge()
    bleu_scores = []
    # need to deal with chinese words
    for pred, ref in zip(predictions, references):
        pred_words = " ".join(jieba.cut(pred))      
        ref_words = " ".join(jieba.cut(ref))
        score = rouge.get_scores(pred_words, ref_words)
        rouge_scores.append(score[0])
        # Calculate BLEU score for each prediction and reference pair
        bleu_score = sentence_bleu([ref_words.split()], pred_words.split())
        bleu_scores.append(bleu_score)
    avg_rouge1 = sum([score['rouge-1']["f"] for score in rouge_scores]) / len(rouge_scores)
    avg_rouge2 = sum([score['rouge-2']["f"] for score in rouge_scores]) / len(rouge_scores)
    avg_rougeL = sum([score['rouge-l']["f"] for score in rouge_scores]) / len(rouge_scores)
    avg_bleu = sum(bleu_scores) / len(bleu_scores)
    print(f"Average ROUGE-1: {avg_rouge1}")
    print(f"Average ROUGE-2: {avg_rouge2}")
    print(f"Average ROUGE-L: {avg_rougeL}")
    print(f"Average BLEU: {avg_bleu}")
 
if __name__ == "__main__":
    # get args
    args = parser.parse_args()
    results = evaluation(args)
    with open("results.pkl", "rb") as f:
        results = pickle.load(f)
    compute_metrics(results)