---
title: "Sora Is Gone: Building on Open Source Video Generation"
date: 2026-03-26
draft: false
description: OpenAI killed Sora with no warning and no migration path. Here's
  the open source video generation stack I'm building to replace it — Wan2.2,
  LTX-Video, Ray Serve, and the embedding layer most writeups skip.
tags:
  - open-source
  - video-generation
  - ml
  - infrastructure
  - gpu
---

Yesterday OpenAI killed Sora. No warning, no migration path, no explanation beyond a vague gesture toward robotics research and IPO optics. If you'd built anything on the Sora API, you're now scrambling.

I'm not surprised. Closed video gen APIs were always a fragile foundation — expensive to run, subject to content policy shifts, and one business pivot away from disappearing. The Sora shutdown is just the most visible confirmation of what many of us already suspected: if you need reliable, controllable video generation infrastructure, you need to own the stack.

So that's what I'm building. This post is the first in a series documenting that build — model selection, serving architecture, GPU infrastructure, and the embedding layer that makes generated video actually useful at scale.

## The Open Source Landscape in 2026

The good news is the open source video gen ecosystem is genuinely strong right now, and the timing of Sora's exit couldn't be better for it.

**Wan2.2** (Alibaba Tongyi Lab) is the current benchmark. It introduces a Mixture-of-Experts diffusion architecture — two specialized expert networks handling different denoising timesteps, a high-noise expert for layout and motion, a low-noise expert for lighting, texture, and detail. The practical result is better motion quality and semantic coherence without a linear increase in inference cost. It achieves a VBench score above 84.7%, competes with commercial closed-source alternatives on most benchmarks, and critically, is Apache 2.0 licensed. The 14B parameter variant is the quality target; the 1.3B variant runs on 8GB VRAM for rapid evaluation.

**HunyuanVideo** (Tencent) is the closest competitor — 13B parameters, dual-stream-to-single-stream transformer architecture, strong on complex multi-object scenes and long-form coherence. The quality ceiling is arguably higher than Wan2.2 on certain cinematic tasks. The dealbreaker for a public open source project is the Tencent community license, which restricts redistribution in ways Apache 2.0 doesn't. It stays on my evaluation list but off the production stack.

**LTX-Video** (Lightricks) is the speed story — fastest inference in the class, runs on 12GB VRAM, real-time generation on H200. Quality trades off against the heavier models but it's the right choice for latency-sensitive applications or rapid prototyping loops.

**SkyReels V2** (Skywork AI) is worth noting for human-centric generation — fine-tuned on film and TV content with strong facial animation and motion stability. Apache 2.0. Interesting for the specific use case but too narrow for a general serving layer.

My model selection decision: **Wan2.2-T2V-A14B** for text-to-video, **Wan2.2-I2V-A14B** for image-to-video, with LTX-Video as the low-latency fallback. All Apache 2.0, all serving the same request contract.

## The Embedding Layer

Most video gen infrastructure writeups stop at generation. That's the wrong place to stop.

Generated video is only useful if you can retrieve it, compare it, and understand it at scale. That requires embeddings — and video embedding is a harder problem than it looks.

The naive approach is DINOv2 per-frame pooling. DINOv2 is Apache 2.0, production-proven, and gives you strong frame-level semantic representations. The limitation is temporal blindness — you're averaging over frames, losing motion dynamics and event ordering entirely.

For temporal-aware video embeddings, **VLM2Vec-V2** is the most interesting research target right now — a unified multimodal embedding model trained on a new benchmark covering video retrieval, temporal grounding, and video classification. Very new, but directionally right for what I need.

For cross-modal retrieval (text query → video clip), **Jina Embeddings v4** is the pragmatic choice — built on Qwen2.5-VL-3B with LoRA adapters for different retrieval scenarios, Apache 2.0, and practical to serve.

The architecture I'm targeting: DINOv2 for frame-level embeddings, Jina v4 for cross-modal retrieval, VLM2Vec-V2 as a research comparison. All three on the same cluster, sharing GPU resources with the generative workload.

## The Serving Stack

Video generation is not a synchronous workload. A single Wan2.2-14B inference at 720P runs 30-120 seconds depending on frame count and hardware. You cannot return that over a blocking HTTP connection with any reliability.

The architecture I'm building:

**Ray Serve** for distributed serving and autoscaling. Ray's actor model maps cleanly onto the stateful nature of diffusion inference — each replica holds model weights in GPU memory, accepts job submissions, and streams progress updates back to the caller.

**Async job queue** with Redis as the broker. Client submits a generation request, gets a job ID back immediately, polls for completion. Standard pattern but the implementation details matter — you need backpressure handling and dead letter queues before you call this production-ready.

**EKS** for orchestration. GPU node groups on g5.xlarge (A10G, 24GB VRAM) for the embedding workload and LTX-Video inference; p3.2xlarge (V100, 16GB) as the cost-effective option for batch generation; on-demand p4d.24xlarge (8xA100) reserved for Wan2.2-14B at scale. Karpenter for autoscaling, scaling to zero between jobs to contain costs.

**Triton Inference Server** sitting in front of the diffusion pipeline for the embedding models — synchronous, low-latency, well-suited to the request pattern.

The interesting infra problem this creates: how do you share a GPU fleet between a slow generative workload (minutes per job) and a fast embedding workload (milliseconds per request) without one starving the other? The answer involves separate node pools with different autoscaling policies and a priority queue that reserves capacity for embedding requests. I'll cover this in detail in a future post.

## What's Next

This is a learning project, built in public. The stack I've described is the target architecture — I'm building toward it iteratively, starting with model evaluation on a single rented GPU before touching Kubernetes.

Future posts in this series:
- **Model evaluation**: Wan2.2 vs HunyuanVideo vs LTX-Video on identical prompts, measured on quality, latency, and memory footprint
- **Ray Serve architecture**: wrapping diffusion inference, async job handling, progress streaming
- **EKS setup**: GPU node groups, Karpenter configuration, cost management
- **The embedding problem**: DINOv2 vs Jina v4 vs VLM2Vec-V2 on video retrieval benchmarks
- **GPU resource sharing**: serving gen and embedding workloads on the same cluster

If you want to follow along as I build this, subscribe to the blog. When the repo is ready, I'll post a link — star it if the architecture is useful to you.

Sora is gone. The stack lives on your infrastructure now. Let's build it.
