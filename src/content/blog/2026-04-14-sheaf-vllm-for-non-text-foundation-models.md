---
title: "Sheaf: vLLM for Non-Text Foundation Models"
date: 2026-04-14
draft: false
description: "vLLM solved inference for text LLMs. The same gap exists for every other class of foundation model — time series, tabular, molecular, diffusion, and more. I built Sheaf to fill it."
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
pip install "sheaf-serve[time-series]"         # Chronos2 / TimesFM
pip install "sheaf-serve[tabular]"             # TabPFN
```

## What works today

Two model types are supported as of today.

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

Regression with quantile intervals works the same way — swap `task="regression"` and `output_mode="quantiles"`. Full examples for both are [in the repo](https://github.com/korbonits/sheaf/tree/main/examples).

Alternatively, for time series, pass a Feast feature reference instead of raw history:

```python
req = TimeSeriesRequest(
    model_name="chronos-bolt-tiny",
    feature_ref={"feature_view": "asset_prices", "entity_id": "AAPL"},
    horizon=24,
    frequency=Frequency.HOURLY,
)
```

## The roadmap

| Type | Status | Backends |
|---|---|---|
| Time series | ✅ v0.1 | Chronos, Chronos-Bolt, TimesFM |
| Tabular | ✅ v0.1 | TabPFN |
| Molecular / biological | 🔜 v0.2 | ESM-3, AlphaFold |
| Audio | 🔜 v0.3 | Whisper, MusicGen |
| Embeddings | 🔜 v0.3 | CLIP, ColBERT |
| Geospatial / Earth science | 🔜 v0.3 | GraphCast, Clay |
| Diffusion | 🔜 v0.4 | Flux, Stable Diffusion |
| Neural operators | 🔜 v0.4 | FNO, DeepONet |

The V1 boundary is deliberately narrow: stateless, frozen models with synchronous or streaming responses. Session management (RL policy serving) and mutable weights (continual learning) are v2 problems — I'd rather ship something useful now than design for everything upfront.

## What this isn't

Sheaf v0.1 is not a performance story yet. Both backends run requests sequentially by default. The batching policy exists but the scheduler isn't built. The Feast integration is wired at the contract level but not implemented end-to-end.

What it *is* is the API layer — the contracts that everything else builds on. That's the right thing to ship first, because the contracts are what drive adoption. If the time series request schema is wrong, no amount of batching optimization will fix it.

The bet is that standardizing the API layer now creates the surface area for the performance work to follow. That's how vLLM worked too: the API came first, the optimizations accreted around it.

---

The repo is at [github.com/korbonits/sheaf](https://github.com/korbonits/sheaf). `pip install sheaf-serve`. Issues and PRs welcome — especially from anyone who has strong opinions about what the molecular or geospatial contracts should look like.

---

*korbonits.com is my personal blog. I write about ML, software, and books.*
