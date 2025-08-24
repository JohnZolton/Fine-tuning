from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import Dataset
import json

model_name = "Qwen2.5-0.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")

# Load JSONL manually
with open("/home/john/Documents/Programming/RL-llama/dataset/sft_data.jsonl", "r") as f:
    data = [json.loads(line) for line in f]

print(len(data))
# Convert to HF Dataset
dataset = Dataset.from_list(data)
print(len(dataset))

# Split train/test
train_test = dataset.train_test_split(test_size=0.1)
train_data = train_test["train"]
eval_data = train_test["test"]

def format_example(example):
    return {
        "text": f"User: {example['question']}\nAssistant: {example['tool_call']}"
    }

train_data = train_data.map(format_example)
eval_data = eval_data.map(format_example)

from transformers import DataCollatorForLanguageModeling

tokenized_data = train_data.map(
    lambda x: tokenizer(x["text"], truncation=True, padding="max_length", max_length=512),
    batched = True
)

tokenized_eval = eval_data.map(
    lambda x: tokenizer(x["text"], truncation=True, padding="max_length", max_length=512),
    batched = True
)


data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

from peft import LoraConfig, get_peft_model

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)

from transformers import Trainer, TrainingArguments

training_args = TrainingArguments(
    output_dir="./tool_sft",
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=8,
    evaluation_strategy="steps",
    eval_steps=200,
    save_steps=500,
    num_train_epochs=3,
    learning_rate=5e-5,
    fp16=True,
    logging_steps=50,
    save_total_limit=2
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_data,
    eval_dataset=tokenized_eval,
    tokenizer=tokenizer,
    data_collator=data_collator
)

trainer.train()

prompt = "Schedule a lunch with Sarah tomorrow."
inputs = tokenizer(f"User: {prompt}\nAssistant:", return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_new_tokens=128)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
