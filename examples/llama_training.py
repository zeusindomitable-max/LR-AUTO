# Full training loop with LR-AUTO
from transformers import AutoModelForCausalLM, AutoTokenizer
from lr_auto import lr_auto
import torch

model = AutoModelForCausalLM.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
optimizer = torch.optim.AdamW(model.parameters(), lr=0)  # dummy LR

text = "Hello, how are you?"
inputs = tokenizer(text, return_tensors="pt")
total_steps = 100

for step in range(total_steps):
    outputs = model(**inputs, labels=inputs["input_ids"])
    loss = outputs.loss
    loss.backward()
    lr_auto(optimizer, step, total_steps, verbose=(step % 20 == 0))
    optimizer.step()
    optimizer.zero_grad()
    if step % 20 == 0:
        print(f"Step {step} | Loss: {loss.item():.4f} | LR: {optimizer.param_groups[0]['lr']:.2e}")
