---
title: "Sheaf: vLLM for Non-Text Foundation Models"
date: 2026-04-14
draft: false
description: "vLLM solved inference for text LLMs. The same gap exists for every other class of foundation model — time series, tabular, molecular, diffusion, and more. Sheaf fills it: typed contracts, model-type-aware batching, streaming, caching, observability, offline batch inference, an async-job worker, and 27 backends on PyPI."
tags:
  - open-source
  - mlops
  - serving
  - foundation-models
  - time-series
  - ray
  - python
---

Staring at nineteen open source ML repos last week — the ones I described in [my last post](/blog/2026-04-12-bulk-oss-contributions-ruff-and-ci) — I kept noticing the same thing. Every repo had a model. None of them had a serving story.

That's expected for paper repos. But it surfaces a real question: if you wanted to actually *deploy* Chronos2, or TabPFN, or ESM-3, or GraphCast — what would you use? The answer today is "Ray Serve and a lot of glue code." Which is a fine answer, but it's the 2022 answer for text LLMs too, before vLLM.

So I built [Sheaf](https://github.com/korbonits/sheaf).

## What vLLM actually solved

vLLM's headline innovation is PagedAttention — a way to manage KV cache memory that eliminated the GPU fragmentation that made LLM serving so wasteful. But that's not why it won.

It won because all autoregressive text LLMs share a single compute pattern: transformer forward pass, sample next token, repeat. That uniformity made it possible to build one set of serving optimizations that works for every model in the class. PagedAttention, continuous batching, prefix caching — they all fall out of that shared structure.

The result: one command, OpenAI-compatible API, serious throughput. Everyone converged on it.

## The same gap exists everywhere else

The models that aren't text LLMs don't have this. Each one has its own inference pattern, its own batching challenges, its own memory management problem — and no one has done the work to define standard serving contracts for any of them.

Some concrete examples of what's unsolved:

**Time series (Chronos2, TimesFM)** — variable horizon requests can't naively share a batch. Two forecasters asking for 24-step and 96-step predictions have different compute budgets. No framework handles this. And if two concurrent requests share historical context for the same entity, there's no equivalent of prefix caching to avoid recomputing it.

**Tabular (TabPFN)** — TabPFN does in-context learning, which means the "training set" ships with every request. If ten concurrent clients each send a different context table, memory management across those requests is completely ad hoc.

**Molecular/biological (ESM-3, AlphaFold)** — amino acid sequences have wildly variable lengths. Batching them naively wastes GPU memory. No standard batching contract exists.

**Diffusion (Flux, Stable Diffusion)** — iterative denoising means a single request is actually N forward passes, where N is the number of denoising steps. The compute budget per request is dynamic. Nobody has done PagedAttention for diffusion.

**Geospatial (GraphCast, Clay)** — raster inputs, pressure levels, lat/lon grids. No standard request contract. Every team serving GraphCast internally invents their own.

The deepest gap is actually this: **none of these model types have a standard serving API contract**. vLLM also won because everyone agreed on what a text generation request looks like. There's no equivalent for time series, tabular, or molecular models.

## Sheaf

Sheaf is that standard. Each model type gets a typed request/response contract. Batching, caching, and scheduling are optimized behind each contract independently. [Ray Serve](https://docs.ray.io/en/latest/serve/index.html) is the substrate. [Feast](https://feast.dev) is a first-class input primitive for feature-store-backed models.

The name comes from category theory: a sheaf tracks locally-defined data that glues consistently across a space. Each model type defines its own local contract; Sheaf ensures they cohere into a unified serving layer. It's also a word that means nothing in the Python ecosystem, which is increasingly rare.

Install it:

```bash
pip install "sheaf-serve[time-series]"         # Chronos2 / TimesFM / Moirai
pip install "sheaf-serve[tabular]"             # TabPFN
pip install "sheaf-serve[molecular]"           # ESM-3  (Python 3.12+)
pip install "sheaf-serve[genomics]"            # Nucleotide Transformer
pip install "sheaf-serve[small-molecule]"      # MolFormer-XL
pip install "sheaf-serve[materials]"           # MACE-MP-0
pip install "sheaf-serve[audio]"               # Whisper / faster-whisper
pip install "sheaf-serve[audio-generation]"    # MusicGen
pip install "sheaf-serve[tts]"                 # Bark
pip install "sheaf-serve[kokoro]"              # Kokoro TTS — voice + speed per request
pip install "sheaf-serve[vision]"              # DINOv2 / OpenCLIP / SAM2 / Depth Anything / DETR
pip install "sheaf-serve[pose]"                # ViTPose top-down pose estimation
pip install "sheaf-serve[optical-flow]"        # RAFT optical flow (torchvision)
pip install "sheaf-serve[lidar]"               # PointNet 3D point cloud
pip install "sheaf-serve[earth-observation]"   # Prithvi (IBM/NASA)
pip install "sheaf-serve[weather]"             # GraphCast
pip install "sheaf-serve[diffusion]"           # FLUX.1-schnell / FLUX.1-dev
pip install "sheaf-serve[multimodal-generation]"  # SDXL img2img + inpainting
pip install "sheaf-serve[video]"               # VideoMAE / TimeSformer
pip install "sheaf-serve[feast]"               # Feast feature store integration
pip install "sheaf-serve[modal]"               # Modal serverless deployment
pip install "sheaf-serve[batch]"               # Offline batch inference (Ray Data)
pip install "sheaf-serve[worker]"              # Async-job worker (Redis Streams)
```

## The serving layer

The point isn't the backends individually — it's that they all share the same serving infrastructure. You declare what you want to run with a `ModelSpec`, and `ModelServer` handles the rest: Ray Serve deployment, HTTP endpoints, request validation, batching, and health probes.

```python
from sheaf.server import ModelServer
from sheaf.spec import ModelSpec, ResourceConfig
from sheaf.api.base import ModelType

server = ModelServer(
    models=[
        ModelSpec(
            name="chronos",
            model_type=ModelType.TIME_SERIES,
            backend="chronos2",
            backend_kwargs={"model_id": "amazon/chronos-bolt-small"},
            resources=ResourceConfig(num_gpus=1, replicas=2),
        ),
        ModelSpec(
            name="molformer",
            model_type=ModelType.SMALL_MOLECULE,
            backend="molformer",
            resources=ResourceConfig(num_gpus=1),
        ),
        ModelSpec(
            name="esm3",
            model_type=ModelType.MOLECULAR,
            backend="esm3",
            resources=ResourceConfig(num_gpus=1),
        ),
    ]
)
server.run()
```

Each model is live at `/<name>/predict`, `/<name>/health`, and `/<name>/ready`. Concurrent requests are batched automatically via `@serve.batch` with per-deployment `max_batch_size` and timeout. Rolling hot-swap with no downtime:

```python
server.update(ModelSpec(
    name="chronos",
    backend="chronos",
    backend_kwargs={"model_id": "amazon/chronos-bolt-base"},  # upgrade weights
    resources=ResourceConfig(num_gpus=1, replicas=4),          # scale up
))
```

Ray Serve handles the transition — new replicas come up with the new spec while old ones drain in-flight requests. The route and URL don't change.

## What the contracts look like

Every model type has a typed request/response pair. Here's a sample across a few:

### Time series — Chronos-Bolt

[Chronos-Bolt](https://github.com/amazon-science/chronos-forecasting) runs on CPU, downloads in under a minute, and returns probabilistic forecasts with quantile intervals.

```python
from sheaf.api.time_series import Frequency, OutputMode, TimeSeriesRequest
from sheaf.backends.chronos import Chronos2Backend

backend = Chronos2Backend(model_id="amazon/chronos-bolt-tiny", device_map="cpu")
backend.load()

req = TimeSeriesRequest(
    model_name="chronos-bolt-tiny",
    history=[312, 298, 275, 260, 255, 263, 285, 320,
             368, 402, 421, 435, 442, 438, 430, 425,
             418, 410, 398, 385, 372, 358, 342, 328],
    horizon=12,
    frequency=Frequency.HOURLY,
    output_mode=OutputMode.QUANTILES,
    quantile_levels=[0.1, 0.5, 0.9],
)

response = backend.predict(req)
```

Output:

```
Forecast: next 12 hours

 Hour      P10   Median      P90
-----------------------------------
    1    278.4    303.3    323.4
    2    257.5    290.2    317.5
    3    241.1    278.8    312.2
    4    229.1    270.3    308.9
    5    224.2    268.8    311.3
    6    228.4    277.4    324.2
    7    244.8    298.7    350.1
    8    278.7    339.4    394.3
    9    309.5    370.9    424.1
   10    336.1    399.5    451.3
   11    356.4    423.6    475.7
   12    369.5    440.1    492.5
```

TimesFM (Google's 200M parameter model) is also supported — same `TimeSeriesRequest`, different backend. Both models agree on the shape (trough around hour 5-6, rising back through the morning), but differ on uncertainty width:

```
 Hour   Chronos P10   Chronos P50   Chronos P90   TimesFM P10   TimesFM P50   TimesFM P90
-------------------------------------------------------------------------------------------
    1         278.4         303.3         323.4         308.5         306.2         316.4
    2         257.5         290.2         317.5         284.9         283.5         296.4
    3         241.1         278.8         312.2         270.7         265.7         284.0
    4         229.1         270.3         308.9         261.2         255.1         274.8
    5         224.2         268.8         311.3         253.7         247.5         269.7
    6         228.4         277.4         324.2         260.3         250.1         276.9
    7         244.8         298.7         350.1         280.7         272.0         302.7
    8         278.7         339.4         394.3         318.3         308.1         347.4
    9         309.5         370.9         424.1         358.7         346.1         389.3
   10         336.1         399.5         451.3         382.7         370.4         411.5
   11         356.4         423.6         475.7         393.2         389.3         419.5
   12         369.5         440.1         492.5         409.6         407.9         432.6
```

Chronos has wider intervals; TimesFM is tighter and tracks slightly lower. Switching backends is one argument — the contract doesn't change.

### Small molecule — MolFormer

IBM's MolFormer-XL embeds SMILES strings into a 768-dimensional space useful for molecular property prediction, similarity search, and retrieval.

```python
from sheaf.api.small_molecule import SmallMoleculeRequest
from sheaf.backends.molformer import MolFormerBackend

backend = MolFormerBackend(model_name="ibm/MoLFormer-XL-both-10pct", device="cpu")
backend.load()

req = SmallMoleculeRequest(
    model_name="molformer",
    smiles=[
        "CC(=O)OC1=CC=CC=C1C(=O)O",       # aspirin
        "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",   # caffeine
        "CC12CCC3C(C1CCC2O)CCC4=CC(=O)CCC34C",  # testosterone
    ],
    pooling="mean",
    normalize=False,
)

response = backend.predict(req)
# response.embeddings: list of 3 vectors, each length 768
# response.dim: 768
```

### Materials science — MACE-MP

[MACE-MP-0](https://github.com/ACEsuit/mace) is a universal interatomic potential. Give it an atomic structure, get back energy and forces without DFT.

```python
import base64
import numpy as np
from sheaf.api.materials import MaterialsRequest
from sheaf.backends.mace import MACEBackend

backend = MACEBackend(model="medium", device="cpu")
backend.load()

# CO2: C at origin, two O atoms at ±1.16 Å
positions = np.array([[0., 0., 0.], [0., 0., 1.16], [0., 0., -1.16]], dtype=np.float32)

req = MaterialsRequest(
    model_name="mace",
    atomic_numbers=[6, 8, 8],
    positions_b64=base64.b64encode(positions.tobytes()).decode(),
    compute_forces=True,
)

response = backend.predict(req)
# response.energy: float (eV)
# response.forces_b64: base64-encoded (3, 3) float32 array (eV/Å)
```

### Tabular — TabPFN

[TabPFN](https://github.com/automl/TabPFN) is an in-context learner: you pass context examples alongside the rows you want to predict. No training step — a single forward pass handles everything. Requires a free [PriorLabs](https://ux.priorlabs.ai) account for local inference.

```python
from sheaf.api.tabular import TabularRequest
from sheaf.backends.tabpfn import TabPFNBackend

backend = TabPFNBackend(device="cpu")
backend.load()

req = TabularRequest(
    model_name="tabpfn",
    context_X=[[5.1, 3.5, 1.4, 0.2], [6.3, 3.3, 4.7, 1.6], [7.1, 3.0, 5.9, 2.1], ...],
    context_y=[0, 1, 2, ...],
    query_X=[[5.0, 3.4, 1.5, 0.2], [6.0, 2.9, 4.5, 1.5], [6.8, 3.0, 5.5, 2.1]],
    task="classification",
    output_mode="probabilities",
)

response = backend.predict(req)
```

Output:

```
Query    Prediction   P(setosa)   P(versicolor)   P(virginica)
-----------------------------------------------------------------
    1        setosa       0.986           0.011          0.003
    2    versicolor       0.010           0.877          0.113
    3     virginica       0.005           0.103          0.892
```

Or, pull history directly from a Feast online feature store instead of passing raw values. Set `feast_repo_path` on the `ModelSpec` and the serving layer resolves the feature before it reaches the backend — the backend always sees `history`, never `feature_ref`:

```python
spec = ModelSpec(
    name="chronos",
    model_type=ModelType.TIME_SERIES,
    backend="chronos2",
    feast_repo_path="/feast/feature_repo",
)

# Client sends feature_ref instead of raw history
req = TimeSeriesRequest(
    model_name="chronos",
    feature_ref=FeatureRef(
        feature_view="asset_prices",
        feature_name="close_history_30d",
        entity_key="ticker",
        entity_value="AAPL",
    ),
    horizon=7,
    frequency=Frequency.DAILY,
    output_mode=OutputMode.QUANTILES,
)
```

Feast errors (store unavailable, feature missing) return 502 and don't crash the deployment. A misconfigured spec — `feature_ref` sent to a deployment without `feast_repo_path` — returns 422. See `examples/quickstart_feast.py` for a full end-to-end example with a local SQLite store.

## The roadmap

| Type | Status | Backends |
|---|---|---|
| Time series | ✅ v0.1 | Chronos2, Chronos-Bolt, TimesFM, Moirai |
| Tabular | ✅ v0.1 | TabPFN v2 |
| Audio transcription | ✅ v0.3 | Whisper, faster-whisper |
| Audio generation | ✅ v0.3 | MusicGen |
| Text-to-speech | ✅ v0.3 | Bark, Kokoro (v0.5) |
| Vision embeddings | ✅ v0.3 | OpenCLIP, DINOv2 |
| Segmentation | ✅ v0.3 | SAM2 |
| Depth estimation | ✅ v0.3 | Depth Anything v2 |
| Object detection | ✅ v0.3 | DETR / RT-DETR |
| Protein / molecular | ✅ v0.3 | ESM-3 (Python 3.12+) |
| Genomics | ✅ v0.3 | Nucleotide Transformer |
| Small molecule | ✅ v0.3 | MolFormer-XL |
| Materials science | ✅ v0.3 | MACE-MP-0 |
| Earth observation | ✅ v0.3 | Prithvi (IBM/NASA) |
| Weather forecasting | ✅ v0.3 | GraphCast |
| Cross-modal embeddings | ✅ v0.3 | ImageBind (text, vision, audio, depth, thermal) |
| Feast feature store | ✅ v0.3 | Any Feast online store (SQLite, Redis, DynamoDB, …) |
| Modal serverless | ✅ v0.3 | `ModalServer` — zero-infra GPU deployment |
| Diffusion / image gen | ✅ v0.4 | FLUX.1-schnell / FLUX.1-dev |
| Video understanding | ✅ v0.4 | VideoMAE, TimeSformer |
| Streaming responses | ✅ v0.5 | `POST /{name}/stream` → SSE; FLUX emits per-step progress |
| Request caching | ✅ v0.5 | In-process LRU + TTL, `CacheConfig` on `ModelSpec` |
| `bucket_by` batching | ✅ v0.5 | Group requests by field before `@serve.batch` |
| Observability | ✅ v0.5 | Prometheus metrics, structured logging, OpenTelemetry traces |
| Pose estimation | ✅ v0.5 | ViTPose — COCO 17-keypoint skeleton, optional person bboxes |
| Optical flow | ✅ v0.5 | RAFT (`raft_large` / `raft_small`) via torchvision |
| Multimodal generation | ✅ v0.5 | SDXL — img2img + inpainting via diffusers pipelines |
| LiDAR / 3D point cloud | ✅ v0.5 | PointNet — embed + ModelNet40 classify (pure-PyTorch) |
| Offline batch inference | ✅ v0.6 | `BatchRunner` — Ray Data `map_batches` substrate, JSONL source/sink in v1 |
| Actor-pool batch mode | ✅ v0.6.1 | Opt-in `compute="actors"` on `BatchSpec` — warm `load()` per actor for FLUX / GraphCast / SDXL |
| Async job queue | ✅ v0.7 | `SheafWorker` — Redis Streams + consumer groups; at-least-once + dead-letter; per-job webhook on completion |
| Adapter multiplexing | 🔜 v0.8 | LoRA hot-swap per request; one deployment, many fine-tunes |
| Client SDK | 🔜 v0.8 | Typed Python client + OpenAPI spec |

The V1 boundary is deliberately narrow: stateless, frozen models with synchronous or streaming responses. Session management (RL policy serving) and mutable weights (continual learning) are v2 problems — I'd rather ship something useful now than design for everything upfront.

## What's next

v0.7.0 is on PyPI. The serving layer is complete — every major non-text model class, Feast integration, Modal serverless deployment, full observability, streaming SSE, model-type-aware batching, *and* both deployment shapes that HTTP request/response is the wrong fit for: offline batch and async job queue.

v0.4 shipped generation and video: FLUX for diffusion image generation and VideoMAE / TimeSformer for video understanding and classification.

v0.5 shipped the production ops layer:

- **Streaming** — `POST /{name}/stream` returns a `text/event-stream` response. FLUX emits one progress event per denoising step. Any backend can override `stream_predict()` for chunked output. The default fallback yields a single result event, so every backend gets SSE for free.
- **Request caching** — `CacheConfig` on `ModelSpec` attaches an in-process LRU cache. SHA-256 keyed, TTL optional, `SHEAF_CACHE_DISABLED=1` for integration tests. Computed after Feast resolution so the key reflects actual input values, not feature references.
- **`bucket_by` batching** — the time series problem from the original design: 24-step and 96-step requests landing in the same Ray Serve batch window no longer force each other to pad. `batch_policy.bucket_by = "horizon"` routes them into homogeneous sub-batches before the backend sees them.
- **Observability** — Prometheus metrics (`sheaf_requests_total`, `sheaf_request_duration_seconds`), structured JSON logging with request IDs, and OpenTelemetry traces with `sheaf.predict` / `sheaf.feast.resolve` / `sheaf.backend.infer` spans.

v0.6 shipped offline batch inference and the warm-load actor-pool variant:

- **`BatchRunner`** — same backend, same typed contract, offline batch mode. Ray Data `map_batches` is the substrate. `BatchSpec` mirrors `ModelSpec` for backend selection and adds `source` / `sink` / `batch_size` / `num_cpus` / `num_gpus`. Rows are pre-validated against the same Pydantic contract on the driver before Ray dispatch, so schema errors surface up-front rather than halfway through a distributed run. v1 ships `JsonlSource` / `JsonlSink`; S3, Parquet, and Delta slot in as additional `BatchSource`/`BatchSink` subclasses without changing the runner API. Output order is deterministic — the runner injects a row-index sentinel and sorts on the way out, since Ray Data's streaming executor doesn't preserve order on its own.
- **Actor-pool execution mode** (v0.6.1) — opt-in `compute="actors"` on `BatchSpec` plus `num_actors=N` switches dispatch to a Ray Data actor pool. Each actor calls `backend.load()` once at `__init__`; the loaded model persists for the actor's lifetime. Eliminates per-batch cold-start cost on FLUX (~30-60s `load()`), GraphCast, and SDXL — the backends where the stateless task path's worker-cache fallback would otherwise dominate total job time.

v0.7 shipped the other deployment shape — async jobs:

- **`SheafWorker`** — queue-consumer pattern for inference where HTTP request/response is the wrong shape. FLUX at 50 steps, GraphCast multi-day rollouts, large-batch SDXL — clients enqueue a typed request, the worker dequeues, runs inference, persists the result, optionally POSTs a webhook on completion. v1 ships Redis Streams + consumer groups (so multiple workers split the work horizontally) and a Redis hash result store with optional TTL. `JobQueue` and `ResultStore` are ABCs; SQS, Kafka, and Postgres slot in as additional subclasses without changing the worker loop. At-least-once delivery — XACK fires only after the result is persisted, so a worker crash mid-job causes redelivery to another consumer. Jobs that exceed `max_retries` go to a dead-letter stream *and* get a `status="failed"` `JobResult` written to the store, so `JobQueueClient.wait_for_result(job_id, timeout)` doesn't hang on poison pills.

And v0.8 is the economics argument: LoRA adapter multiplexing (one GPU deployment serves many fine-tunes, per-request adapter hot-swap) and a typed client SDK so Sheaf is consumable from anywhere, not just Python services.

The bet was that getting the contracts right first was worth more than shipping half-baked optimizations behind the wrong abstractions. So far, that bet looks right.

---

The repo is at [github.com/korbonits/sheaf](https://github.com/korbonits/sheaf). `pip install sheaf-serve`. Issues and PRs welcome.

---

*korbonits.com is my personal blog. I write about ML, software, and books.*
