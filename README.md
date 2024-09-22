# llm-ops
Past, Present, and Future of LLM Operations

## Prerequisites 
1. Create a Google Account (if you do not have one, yet): https://accounts.google.com/  
1. Login with that account into Google Colab: https://colab.research.google.com/ 
1. Create a Huggingface account (if you do not have one, yet): https://huggingface.co/join
1. Create an access token and save it for later use in the notebooks: https://huggingface.co/settings/tokens
1. Request access to gated Huggingface models:
   1. https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct 
   1. https://huggingface.co/google/gemma-2-2b-it

## Epochs

### Comparing active GPU microarchitectures
* T4/RTX 20: https://en.wikipedia.org/wiki/Turing_(microarchitecture)
  * V100 - professional variant of RTX 20 consumer line: https://en.wikipedia.org/wiki/Volta_(microarchitecture)
* A100/RTX 30: https://en.wikipedia.org/wiki/Ampere_(microarchitecture)
* L4/L40/RTX 40: https://en.wikipedia.org/wiki/Ada_Lovelace_(microarchitecture)
  * H100 - professional variant of RTX 40 consumer line, not available on Colab (yet?): https://en.wikipedia.org/wiki/Hopper_(microarchitecture)
* Future successor to both Hopper and Ada Lovelace: https://en.wikipedia.org/wiki/Blackwell_(microarchitecture)
* comparing GPUs: https://www.reddit.com/r/learnmachinelearning/comments/18gn1b2/choosing_the_right_gpu_for_your_workloads_a_dive/

### Past - Encoder Models
Transformer on CPU: https://colab.research.google.com/github/DJCordhose/llm-ops/blob/main/Past.ipynb

* https://arxiv.org/pdf/2209.11055
* https://sbert.net/
* https://huggingface.co/blog/setfit

### Present - Small to medium Decoder Models
* https://colab.research.google.com/github/DJCordhose/llm-ops/blob/main/Present.ipynb
  * small LLM on widely available and affordable GPU Quantized on T4 (16 GB)
  * Full resolution on L4 (24 GB), fast and sensible, but you have to pay up
* Super small model models
  * https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct
* More models that will only fit with quantization
  * https://huggingface.co/google/gemma-2-9b-it
  * https://huggingface.co/microsoft/Phi-3-medium-128k-instruct
* OpenAI models as a baseline: https://colab.research.google.com/github/DJCordhose/llm-ops/blob/main/Present-OpenAI.ipynb  

### Future - Top Decoder Models
* large LLMs on Huggingface chat: 
  * https://huggingface.co/chat/models/meta-llama/Meta-Llama-3.1-70B-Instruct
  * https://huggingface.co/chat/models/mistralai/Mixtral-8x7B-Instruct-v0.1
* large LLM that really runs only on expensive hardware, preview only, please do not execute: https://colab.research.google.com/github/DJCordhose/llm-ops/blob/main/Future.ipynb
