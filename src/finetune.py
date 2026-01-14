from unsloth import FastLanguageModel
import torch
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments

max_seq_length = 2048
dtype = None
load_in_4bit = True
dataset_path = "../data/ank_dataset.json"
output_dir = "../models/litlex_adapter"

print(f"Loading the model...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/llama-3-8b-bnb-4bit",
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
)

model = FastLanguageModel.get_peft_model(
    model,
    r=64,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj", ],
    lora_alpha=32,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=3407,
)

print(f"Loading dataset: {dataset_path}")
dataset = load_dataset("json", data_files=dataset_path, split="train")

print(f"Original columns: {dataset.column_names}")

for col in dataset.column_names:
    if col.lower() == "instruction" and col != "instruction":
        print(f"🔄 Renaming column '{col}' to 'instruction'")
        dataset = dataset.rename_column(col, "instruction")
    if col.lower() == "output" and col != "output":
        print(f"🔄 Renaming column '{col}' to 'output'")
        dataset = dataset.rename_column(col, "output")
    # Если вдруг назвали "question" или "answer"
    if col.lower() == "question":
        dataset = dataset.rename_column(col, "instruction")
    if col.lower() == "answer":
        dataset = dataset.rename_column(col, "output")

print(f"Fixed columns: {dataset.column_names}")

alpaca_prompt = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{}

### Response:
{}"""

EOS_TOKEN = tokenizer.eos_token


def formatting_prompts_func(examples):
    instructions = examples["instruction"]
    outputs = examples["output"]
    texts = []
    for instruction, output in zip(instructions, outputs):
        text = alpaca_prompt.format(instruction, output) + EOS_TOKEN
        texts.append(text)
    return {"text": texts, }


dataset = dataset.map(formatting_prompts_func, batched=True)

print("Starting training...")

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    dataset_num_proc=2,
    packing=False,
    args=TrainingArguments(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=20,
        max_steps=500,
        learning_rate=2e-4,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir="outputs",
    ),
)

trainer.train()

print(f"Saving the model to {output_dir}...")
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
print("✅ Done! The model has been saved.")
