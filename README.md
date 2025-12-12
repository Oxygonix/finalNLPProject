<!-- 

TinyStories minGPT: Positional Encoding and Regularization Experiments

This project explores how architectural inductive bias and regularization strategies affect small transformer language models trained under limited computational resources. Using minGPT and the TinyStories dataset, we evaluate the impact of ALiBi positional encodings, R-Drop regularization, and their interaction, and compare our trained models against a pretrained DistilGPT-2 baseline.

Our goal is not to maximize scale, but to understand why specific design choices matter—especially in small models where their effects are most visible.

Dataset

TinyStories

Models
Baseline: minGPT

Decoder-only transformer

Trained from scratch

GPT-2 tokenizer (vocab size 50,257)

Small architecture (4–5 layers)

Modified Models

ALiBi positional encoding

R-Drop regularization

ALiBi + R-Drop (combined)

Pretrained Comparison

DistilGPT-2 (HuggingFace)

Evaluated without fine-tuning

-->