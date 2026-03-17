import ast
from datasets import load_dataset, concatenate_datasets
from unsloth import FastLanguageModel, is_bfloat16_supported
from unsloth.chat_templates import get_chat_template, train_on_responses_only
from trl import SFTTrainer, SFTConfig

print("========== SCRIPT INITIATED ==========")

print("--- Starting Step 1: Loading Model & Tokenizer ---")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/LFM2.5-1.2B-Instruct",
    max_seq_length=8192,
    load_in_4bit=True,
)
tokenizer = get_chat_template(tokenizer, chat_template="chatml")
print("--- Finished Step 1: Loading Model & Tokenizer ---")

print("--- Starting Step 2: Loading Datasets ---")
gooftagoo_ds = load_dataset("adi-kmt/gooftagoo", split="train")
english_ds = load_dataset("mlabonne/FineTome-100k", split="train").select(range(2000))
print(f"Loaded Gooftagoo: {len(gooftagoo_ds)} rows")
print(f"Loaded English: {len(english_ds)} rows")
print("--- Finished Step 2: Loading Datasets ---")

print("--- Starting Step 3: Applying Custom Formatting Functions ---")
def format_english(example):
    new_conversations = []
    for msg in example["conversations"]:
        if "role" in msg and "content" in msg:
            role = msg["role"]
            content = msg["content"]
        elif "from" in msg and "value" in msg:
            role = msg["from"]
            if role == "human": role = "user"
            elif role == "gpt": role = "assistant"
            content = msg["value"]
        else:
            continue
        new_conversations.append({"role": role, "content": content})
    return {"conversations": new_conversations}

def format_gooftagoo(example):
    try:
        raw_turns = ast.literal_eval(example["conversation"])
    except (ValueError, SyntaxError):
        raw_turns = []
        
    new_conversations = []
    for turn in raw_turns:
        if "user" in turn:
            new_conversations.append({"role": "user", "content": turn["user"]})
        if "assistant" in turn:
            new_conversations.append({"role": "assistant", "content": turn["assistant"]})
            
    return {"conversations": new_conversations}

gooftagoo_ds = gooftagoo_ds.map(format_gooftagoo, remove_columns=gooftagoo_ds.column_names)
english_ds = english_ds.map(format_english, remove_columns=english_ds.column_names)
print("--- Finished Step 3: Applying Custom Formatting Functions ---")

print("--- Starting Step 4: Concatenating and Applying Chat Template ---")
mixed_dataset = concatenate_datasets([gooftagoo_ds, english_ds]).shuffle(seed=3407)

def format_with_template(examples):
    texts = tokenizer.apply_chat_template(examples["conversations"], tokenize=False, add_generation_prompt=False)
    return {"text": [x.removeprefix(tokenizer.bos_token) for x in texts]}

mixed_dataset = mixed_dataset.map(format_with_template, batched=True)
print("--- Finished Step 4: Concatenating and Applying Chat Template ---")

print("--- Starting Step 5: Splitting Dataset ---")
split_dataset = mixed_dataset.train_test_split(test_size=0.1, seed=3407)
train_data = split_dataset["train"]
eval_data = split_dataset["test"]
print(f"Final Train Rows: {len(train_data)} | Final Eval Rows: {len(eval_data)}")
print("--- Finished Step 5: Splitting Dataset ---")

print("--- Starting Step 6: Applying LoRA Adapters ---")
model = FastLanguageModel.get_peft_model(
    model,
    r=64,
    target_modules=["q_proj", "k_proj", "v_proj", "out_proj", "in_proj", "w1", "w2", "w3"],
    lora_alpha=128,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=3407,
    use_rslora=True,
)
print("--- Finished Step 6: Applying LoRA Adapters ---")

print("--- Starting Step 7: Setting Up SFTTrainer ---")
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_data,  
    eval_dataset=eval_data,
    neftune_noise_alpha=5,
    args=SFTConfig(
        dataset_text_field="text",
        per_device_train_batch_size=1, 
        gradient_accumulation_steps=4, 
        num_train_epochs=1,            
        eval_strategy="steps",
        eval_steps=200,                
        logging_steps=10,
        learning_rate=2e-4,
        warmup_steps=30,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        optim="paged_adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        seed=3407,
        output_dir="outputs",
        report_to="none",
        ddp_find_unused_parameters=False,
    ),
)

trainer = train_on_responses_only(
    trainer,
    instruction_part="<|im_start|>user\n",
    response_part="<|im_start|>assistant\n",
)
print("--- Finished Step 7: Setting Up SFTTrainer ---")

print("--- Starting Step 8: Training Model ---")
trainer.train()
print("--- Finished Step 8: Training Model ---")

print("--- Starting Step 9: Saving LoRA Adapters ---")
trainer.save_model("outputs/unsloth_lora_model")
tokenizer.save_pretrained("outputs/unsloth_lora_model")
print("--- Finished Step 9: Saving LoRA Adapters ---")
print("========== SCRIPT COMPLETE ==========")
