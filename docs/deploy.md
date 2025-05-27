# Deployment
This document shows how to deploy the container image to the RunPod

| Contents  |
| --------- |
- [Pre-made images](#pre-made-images)
- [Building custom image](#building-custom-image)
    - [Local build](#local-build)
    - [RunPod build](#runpod-build)
- [Endpoint settings](#endpoint-settings)

___
## Pre-made images
For quick, optimized deployments you can use pre-made images used in our projects:
- `Qwen3-14B-AWQ`: `3wad/runpod-vllm:0.8.5-qwen3-14B-awq`

Simply copy the `3wad/runpod-vllm...` name of the image and paste it into the `container image` field while [creating endpoint](https://www.runpod.io/console/serverless/new-endpoint/custom).
In the next step, select a GPU with at least 24GB VRAM or more and down in `Advanced` settings, make sure to select `Allowed CUDA versions` to be `12.4` and higher. The rest of the settings are optional and described in the [Endpoint settings](#endpoint-settings) section.
This is all you need to quickly deploy ready-to-use endpoints with the models we decided to pre-build.

## Building custom image
The main advantage and reason to use serverless or any GPU hosting in the first place instead of a managed inference service, is that you can use any model, finetune, LoRA or change the thing you're running in any way you like, similar to how you can do it locally.
Of course, this takes a bit more time and effort than using the pre-build image. First, you have to decide what you'll change and what to include (e.g. will you bake the models or store them in the network volume?). Then, you have two options on how to approach making the changes and building the image.

 ### Local build


 ### RunPod build


## Endpoint settings