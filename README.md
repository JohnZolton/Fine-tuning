# Fine-tuning

Fine tuning a LLaMA model on tool-use question/response pairs

## What is fine tuning?

Fine-tuning means we take a pretrained LLaMA and keep training it on a smaller, task-specific dataset.

At the technical level this is just backpropagating through the model layers. We’re not training from scratch. We reuse most of the pretrained weights and nudge them toward our dataset distribution. That’s why it’s efficient and doable on a single GPU.

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
