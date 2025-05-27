# Configuration
This document shows how to customize and configure the settings and container image for your specific needs.

> [!TIP]
> If you wish to use our pre-configured and ready-to-use images, skip this part and refer to the [deployment](/docs/deploy.md) guide.

| Contents  |
| --------- |
- [Choosing a model](#choosing-a-model)
   - [Model Size](#model-size)
   - [Model Quantization](#model-quantization)
   - [Model Finetuning](#model-finetuning)
- [Configuration file](#configuration-file)
- [Docker file](#dockerfile)

___
## Choosing a model
The journey of deploying an LLM starts with choosing the right one. Since you're here, reading about running vLLM on RunPod, chances are you probably have yours picked already. But if not, you can start with places like [LLM Leaderboard](https://artificialanalysis.ai/leaderboards/models), [Trending on Huggingface](https://huggingface.co/?activityType=all&feedType=following&trending=model), or searching your favourite places on the internet (e.g. with keywords such as "*New best opensource LLM*").

 ### Model Size
 Once you find your champion, you'll most likely be presented with several size options measured in billions of parameters **(B)**. A sweet spot for single-GPU models is usually around 7-72B. I know that's a broad range. But it heavily depends on what GPU you'll want to use - or how much money per second you want to spend (see [RunPod Serverless Pricing](https://www.runpod.io/pricing)).   
 Generally, more parameters mean the model can store and learn more complex patterns, which can lead to better reasoning and understanding, but it's not guaranteed. Training quality, data, and architecture matter more than raw size. A well-trained and tuned 7B model can easily outperform a poorly made 13B or even bigger one.
 Along with the amount of parameters, the size of the memory required on the GPU for the model to fit and run also increases. So picking the right size is really about choosing the balance between model intelligence vs. inference speed and ultimately, the cost per request.
   
> [!NOTE]   
> - Most popular models usually have an option to try them out somewhere online. We highly recommend testing different sizes for your expected use case, which should help you choose the smallest size that will still perform well.   
> - If you choose a small enough model, you'll be able to use longer context lengths and potentially compute multiple requests concurrently on a single running worker if the demand requires it.

 ### Model Quantization
 To overcome the size limitations above, a method called quantization exists. It reduces the precision of the model's weights measured in **bits**. While slightly (or with current methods almost unnoticeably) reduces the quality of responses, it also enormously cuts the required memory on the GPU and improves the inference speed.
 Unquantized models are in 16-bit precision, and you can reduce them to e.g. 8, 4 or with some methods even 2 bits, effectively reducing the required memory by **up to ~85%**. You can find quantized versions of many popular models on [Unsloth's HuggingFace](https://huggingface.co/unsloth), or you can quantize any model yourself with our [*In preparation*](https://github.com/davefojtik/RunPod-Unsloth) repository. vLLM supports loading models with many quantization methods, but only a few are worth using in our case - we recommend GPTQ, AWQ or Bitsandbites (bnb).

 Example model sizes with rough estimations of VRAM requirements (~32k context length, 4bit quantization):
 | Params | VRAM |
 | --- | --- |
 | <7B | <16GB |
 | <14B | <24GB |
 | <32B | <40GB |
 | <72B | 80-100GB |

 ### Model Finetuning
 With the [*In preparation*](https://github.com/davefojtik/RunPod-Unsloth), you can also finetune the models. This process allows you to take your own data and inject new knowledge, conversation style, grammar, and even bigger concepts or actions and drastically improve the model's performance both generally, and on your tasks and specific use case.

## Configuration file
Every vLLM setting is conveniently placed in the [.env](/src/.env) file, sorted into groups, and described based on the current official documentation. Most of them can also be changed dynamically per coldstart via [request payload](/docs/usage.md).
Still, understanding each value can be hard even after reading the vLLM docs. Therefore, with the help of the community, we're gradually adding separate [presets](/presets) with the most important settings changed for each model or usage, so you can have a better idea of what you'll need to change for yours.

> [!NOTE]
> Even though it means that at least the configuration should always have 100% feature coverage with the loaded vLLM version, the handler itself is not guaranteed to support, nor to be tested on everything out of the box. The focus is on features and optimizations for leading small to medium models meant to fit and run on a single GPU.

## Dockerfile
Dockerfile is self-documented, kept simple and unified for both Standalone and NetworkVolume versions of the worker. This way you can easily customize it just by commenting or un-commenting parts of it.
You have to edit this file mainly when changing models, or where are they loaded from (baked vs. network volume). In this case, don't forget to also set the proper model path in [.env](/src/.env) or your request.