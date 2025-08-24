# Fine-tuning - LoRA

Fine tuning a LLaMA model on tool-use question/response pairs

## What is fine tuning?

Fine-tuning means we take a pretrained LLaMA and keep training it on a smaller, task-specific dataset.

We’re not training from scratch. We reuse most of the pretrained weights and nudge them toward our dataset distribution. That’s why it’s efficient and doable on a single GPU.

LoRA stands for Low-Rank Adaptation, where small matrices are inserted and trained. (peft = parameter efficient fine tuning)

```python
from peft import LoraConfig, get_peft_model
model = get_peft_model(model, lora_config)
```

This replaces a small subset of weight matrices (q_proj and v_proj) with LoRA adapters.

During training, only the LoRA adapter parameters get updated.

Chose Qwen 2.5 0.5B to fine tune because Qwen is great and tiny.

```json
{
  "question": "How many states are there?",
  "tool_call": "get_state_count()"
}
```

Generic tool+descriptions are in `dataset/tools.json`, we'll use this to build our Q&A pairs. The vision for this project was a small model suitable for mobile to be a personal assistant like siri or JARVIS.

Generate a dataset with `dataset/generate_sft_data.py`, using google gemini-flash-lite bc dirt cheap.

Once you have your dataset, run `python train.py`

I used a 3090, if you run OOM, adjust these:

```python
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=8,
```
