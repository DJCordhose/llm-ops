# llm-ops
Past, Present, and Future of on-prem LLM Operations

https://bit.ly/odsc-2024-on-prem-llm

![](./bit.ly_odsc-2024-on-prem-llm.png)

## Prerequisites 
1. Create a Google Account (if you do not have one, yet): https://accounts.google.com/  
1. Login with that account into Google Colab: https://colab.research.google.com/ 
1. Create a Huggingface account (if you do not have one, yet): https://huggingface.co/join
1. Create an access token and save it for later use in the notebooks: https://huggingface.co/settings/tokens
1. Request access to gated Huggingface models:
   1. https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct 
   1. https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct
   1. https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct"

## Notebooks running on Colab
* https://colab.research.google.com/github/DJCordhose/llm-ops/blob/main/Past.ipynb
* https://colab.research.google.com/github/DJCordhose/llm-ops/blob/main/Present.ipynb
* https://colab.research.google.com/github/DJCordhose/llm-ops/blob/main/Present-OpenAI.ipynb
* https://colab.research.google.com/github/DJCordhose/llm-ops/blob/main/Future.ipynb

## Motivation / Intro

## Use Case

* Medical assessment
* Weird use of language
* Completely made up, but realistic
* Binary classification, e.g.
  * Negative:  "No specific findings can be derived from the diagnosis currently named as the basis for the regulation.",
  * Positive: "Socio-medical indication for the aid is confirmed.",
* Multi lingual: European, American
* Just a handful of examples available

### We not just use Cloud based APIs?

You might want control over
* Privacy / data protection
* Availability and Scaling
* Latency
* Limitations
* Cost of operation
* Ecological footprint
* Stability
* Politics
## Present - Small to medium size Decoder Models

### Decoder Models
* General models
* Can generate answers
* Work without training
  * Might also benefit from few shot learning
  * Are extremely costly to train, impractical for almost everyone

### Decoder On-Prem Challenges
* Context sizes vary (depending on the Model)
  * with large contexts certain positions might be blind spots
* Memory consumption grows with context used
* Scaling to more than one parallel request

*Inference on GPU only* 

### Comparing active GPU microarchitectures
* T4/RTX 20: https://en.wikipedia.org/wiki/Turing_(microarchitecture)
  * V100 - professional variant of RTX 20 consumer line: https://en.wikipedia.org/wiki/Volta_(microarchitecture)
* A100/RTX 30: https://en.wikipedia.org/wiki/Ampere_(microarchitecture)
* L4/L40/RTX 40: https://en.wikipedia.org/wiki/Ada_Lovelace_(microarchitecture)
  * H100 - professional variant of RTX 40 consumer line, not available on Colab (yet?): https://en.wikipedia.org/wiki/Hopper_(microarchitecture)
* Future successor to both Hopper and Ada Lovelace: https://en.wikipedia.org/wiki/Blackwell_(microarchitecture)
* comparing GPUs: https://www.reddit.com/r/learnmachinelearning/comments/18gn1b2/choosing_the_right_gpu_for_your_workloads_a_dive/

### Limiting factor is RAM
* T4: 16GB
* A100: 40GB/80GB
* L4: 24GB (L40: 48GB)

### Options
Use Model with 
1. smaller number of parameters with full resolution (16-Bit)
2. larger number, but lower resolution
   * https://huggingface.co/docs/transformers/main/en/quantization/overview
   * 8-Bit integer or 4-Bit float common choices

*Context might still need more memory than parameters*

### Bitsandbytes
Most straight forward approach to quantization
* https://huggingface.co/docs/text-generation-inference/conceptual/quantization#quantization-with-bitsandbytes 
* Deep Dive: https://huggingface.co/blog/hf-bitsandbytes-integration  
* Can go down to 4 Bits: https://huggingface.co/blog/4bit-transformers-bitsandbytes  
* Inference can be slower than more sophisticated methods (like GPTQ) or full FP16 precision: https://huggingface.co/blog/hf-bitsandbytes-integration#is-it-faster-than-native-models 


### Demo
* https://colab.research.google.com/github/DJCordhose/llm-ops/blob/main/Present.ipynb
  * small LLM on widely available and affordable GPU Quantized on T4 (16 GB)
  * Full resolution on L4 (24 GB), fast and sensible, but you have to pay up
* Super small model models
  * https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct
* More models that will only fit with quantization
  * https://huggingface.co/google/gemma-2-9b-it
  * https://huggingface.co/microsoft/Phi-3-medium-128k-instruct
* OpenAI models as a baseline: https://colab.research.google.com/github/DJCordhose/llm-ops/blob/main/Present-OpenAI.ipynb  


## Future - Top Decoder Models

### The future is already there, it is just not evenly distributed
* There are more powerful versions of OS decoder models available
  * Rival OpenAI GPT models
  * Support for major European languages
* Those models will run on available hardware and dedicated inference server
  * H100 GPUs are are expensive, but available
  * Inference servers optimize for latency and throughput
    * https://huggingface.co/docs/text-generation-inference
    * https://developer.nvidia.com/nim
    * https://github.com/vllm-project/vllm
* We can get a preview
  * https://build.nvidia.com/explore/discover 
  * https://huggingface.co/chat/ 

### Option: Mixtral 8x7B
* Good context length: 24K input, 8K output
* explicitly tuned for European languages (like French, Italian, German and Spanish)
* Mixture of experts
  * only uses fraction of parameters at a time
  * thus also bringing down KV-cache needs

Reference
* https://mistral.ai/news/mixtral-of-experts/ 
* https://huggingface.co/mistralai/Mixtral-8x7B-Instruct-v0.1  
* Sparse Mixture of Experts (SMoE) Mixtral 8x7B: https://arxiv.org/abs/2401.04088 


### Option: Llama 3.1 70B
* Even better context length: 128k
* Supported languages: English, German, French, Italian, Portuguese, Hindi, Spanish, and Thai.
* Significantly better scores in European languages than 8B version
* On-par with current OpenAI GPT models
* Compared to Mixtral 8x7B
  * significantly better scores all over
  * Needs more memory and compute

Reference
* https://huggingface.co/meta-llama/Meta-Llama-3.1-70B-Instruct
* https://ai.meta.com/blog/meta-llama-3-1/ 
* https://llama.meta.com/ 

### It works
Mixtral 8x7B on 2xH100 NVL using TGI

![image](https://github.com/user-attachments/assets/7c9eac4f-d5f0-4c59-b30b-709081b0c471)

### Demo
* large LLMs on Huggingface chat: 
  * https://huggingface.co/chat/models/meta-llama/Meta-Llama-3.1-70B-Instruct
  * https://huggingface.co/chat/models/mistralai/Mixtral-8x7B-Instruct-v0.1
* large LLM that really runs only on expensive hardware, preview only, please do not execute: https://colab.research.google.com/github/DJCordhose/llm-ops/blob/main/Future.ipynb


### GB200- Future successor to both Hopper and Ada Lovelace

* https://en.wikipedia.org/wiki/Blackwell_(microarchitecture)
* 2,5x faster than H100
* 2x memory
* Native support for 4 Bit resolution 
* Sped up NVLink

### Alternatives for local machine (without NVIDIA GPU)
* llama.cpp
  * https://github.com/ggerganov/llama.cpp/blob/master/README.md
  * https://www.theregister.com/2024/07/14/quantization_llm_feature/ 
  * Quantization and optimization
  * Optimized for Apple Silicon M1/M2/M3/M4
* Ollama
  * Simplifies usage of llama.cpp
  * https://ollama.com/  
  * https://github.com/ollama/ollama 
  * https://www.theregister.com/2024/03/17/ai_pc_local_llm/ 


## Past - Encoder Models

### Transformers, LLMs, Encoder, Decoder
![image](https://github.com/user-attachments/assets/5504cc9d-53be-41dc-820a-41ce43287f78)

* *Transformers*: A flexible architecture that uses self-attention to process sequential data efficiently.
* *LLMs*: Large-scale Transformer models trained on extensive text datasets to perform various language tasks.
  * *Encoder Models* 
    * Part of the Transformer architecture focused on understanding and interpreting input data (e.g. BERT)
    * Instrumental for Embedding Models
* *Decoder Models*
  * Part of the Transformer architecture focused on generating sequential output based on the interpreted inputs or prior outputs
  * Instrumental for GPT-style Models like Llama, Mistral or OpenAI GPT

### Encoder Models

* Well understood and mature
* Inference on CPU possible
* Training on pretty much any GPU
  * With a few tricks even training on CPU possible in seconds
* Useful for predicting categories / binary
* Limited usefulness for QA
  * can only predict range in original text
* Not sufficient for generation

### Demo

Transformer on CPU: https://colab.research.google.com/github/DJCordhose/llm-ops/blob/main/Past.ipynb

## Wrap-Up
* Decoder SLMs work reasonably well on all current GPUs
* Might even make sense when better hardware is available because of latency and load
* Harder tasks need stronger models
* A more powerful model might be a game changer
* Quantize larger models to fit into GPUs with less memory
* However, prediction can become a lot slower
* Finally: would even an Encoder SLM do the job?  

## Optional - Evaluation

### Hints / Generals Rules of Thumb
* Especially important for smaller / weaker models
* If you spend a lot of time prompting, your model most like is not up to the task
* Accept defeat, don't overfit
* A more powerful model might be a game changer

### Online Evaluation - Sanity Check (local)
* Local quality / sanity check
* Results come with uncertainty, translate for UX
* Tells user what to do with results
* No general statistics
* Low Latency
* Displayed in a way comprehensible for the user, e.g. a traffic sign
  * green: result can be trusted
  * yellow: check result
  * red: don't even show result
* highlighting / display sources in context information
  * what part of context is relevant and why and
  * what part is not and why   

### LLM as a judge
* let an LLM judge the quality of the prediction
  * either write the prompt yourself or 
  * let a lib do that for you or
  * let a lib provide a prompt to write the prompt (G-Eval)
* compromise between latency / no. requests / quality
* load / latency of judge often higher than actual prediction
* Examples from https://docs.confident-ai.com/docs/metrics-llm-evals
  * _answer relevancy_: does the prediction/answer match the task/question?
  * _faithfulness / hallucination_: when using context / RAG does the answer align with the it?
* needs gt
  * is everything relevant in the answer?
  * is everything in the answer correct?
  * is anything missing from the answer
  * _contextual relevancy_: does the context contain all the information needed for the answer? 

### Offline Evaluation (global)
* how well are we doing overall / globally?
* for us developers
* can be used for drift detection
* can be very technical
* rarely a single dimension
* can include basic statics
  * accuracy 
  * length of answers in
    * characters
    * words
    * bullet points
* probably displayed in a (Grafana) Dashboard

### Ground Truthn (gt / y)
* problem for both online and offline evaluation
* how to get?
  * initial evaluation phase collecting gt for every prediction
  * force user entry for a larger sample (faking a red traffic sign) when in production
* how to compare y and y_hat?
* some (!) metrics work without gt

### Drift detection
* offline evaluation can be basis for drift detection
* scores changes a lot
  * accuracy or 
  * a criteria
* distributions changes a lot, e.g.
  * length of inputs or outputs
  * processing time / latency
  * use univariate two-sample tests - choose test based on properties of samples
* train an Encoder Model to tell old data from new data
  * if the model has predictive power, there must be a systematic change
* links  
  * https://www.evidentlyai.com/blog/data-drift-detection-large-datasets 
  * https://www.evidentlyai.com/blog/open-source-llm-evaluation#drift-detection 
  * Failing Loudly: An Empirical Study of Methods for Detecting Dataset Shift: https://arxiv.org/abs/1810.11953 
  * https://www.evidentlyai.com/blog/embedding-drift-detection 
