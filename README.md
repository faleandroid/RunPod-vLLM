![header](https://github.com/user-attachments/assets/1db37795-04ae-4b95-9579-71bd01b43e3f)

![Static Badge](https://img.shields.io/badge/VLLM_version-0.8.5-blue) ![Static Badge](https://img.shields.io/badge/Status-Pre—release-orange)

RunPod serverless handler for vLLM, optimized for efficient tuning and production deployment. Written from scratch, easily customisable, OpenAI compatible.

### Advantages and Support:
- Optimized, minimal container image and settings for the best performance/cold-start time
- Automatic VLLM + RunPod **continuous batching** (AsyncEngine + **Dynamic Concurrent Handler** modifier)
- **Per-coldstart vLLM configuration** via request payload (change engine arguments and env variables for each cold-start) 
- **Static memory profiling** patched into vLLM source code (reduces cold-start time by an additional ~30%)
- **Automatic network volume shared torch.compile and graph capture** (allows to perform caching only when network path is detected)
- **[Prewarming](/docs/usage.md)** method for RunPod Flashboot
- SSE Concurrent Streaming (Generator Handler)
- Baked or network volume models
- FP8, GPTQ, AWQ, Bitsandbites quantizations

### Todo / Needs to be tested:
- Network volume setup interface
- Embedding and Multimodal models
- Q/LoRAs
- Tensorizer for faster model loading over the network

> [!TIP]
> For quick, optimized deployments you can use pre-made images used in our projects:
> - `Qwen3-14B-unsloth-bnb-4bit`: `3wad/runpod-vllm:0.8.5-qwen3-14B-bnb`
___

### Documentation
- ⚙️ [Configuration](/docs/configure.md)
- 📦 [Deployment](/docs/deploy.md)
- 🚀 [Usage](/docs/usage.md)

___
### Why use this template when there are others, including the official one?
- This repo tries to implement and fix things not addressed by the others, which was the main reason it was made.
- It's used and tested with the best-performing models in our own projects. That way, you can enjoy optimized pre-made images with baked models and settings dialled in for the best performance or coldstart time.
- It aims to be easy to fix and debug with well-organized code in a single handler file. You can now fork this repo, customize it and build it directly into the RunPod with their new GitHub build feature!
- It aims for active communication and cooperation, to help with all issues and to be open to community feature requests.

### Contributors and support
All sorts of help with this repo is very welcomed, no matter whether you choose to answer the questions, try to help with issues, contribute to the code or performance tuning.
We try our best to communicate the priorities of bug fixes and further development through tags but never hesitate to reach out and contact us with any interest in cooperation.

Please keep in mind this is a hobby project. While often their contributors, we're not directly associated with the companies or the things we're implementing. We're trying to help the open-source community in our free time and for free.
If you use the projects commercially, consider supporting the [vLLM](https://github.com/sponsors/vllm-project).
___
> [!NOTE] 
> *This repo is in no way affiliated with RunPod Inc. All logos and names are owned by the authors. This is an unofficial community implementation*