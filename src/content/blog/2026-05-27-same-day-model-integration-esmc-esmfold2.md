---
title: "Same-Day Model Integration: ESMC + ESMFold2 in Sheaf v0.11"
date: 2026-05-27
draft: false
description: "Chan Zuckerberg Biohub released a new protein language model and structure predictor this morning. Sheaf v0.11 shipped with both, same day. The story is less about hustle and more about what a typed serving contract buys you when a new model lands."
tags:
  - open-source
  - mlops
  - serving
  - foundation-models
  - protein
  - sheaf
  - biohub
---

[Chan Zuckerberg Biohub](https://www.czbiohub.org) released ["a world model of protein biology"](https://github.com/Biohub/esm) this morning: [ESMC](https://huggingface.co/collections/Biohub/esmc-model-family) (a protein language model), [ESMFold2](https://huggingface.co/collections/biohub/esmfold2-model-family) (a structure predictor built on top of ESMC 6B), and [ESM Atlas](https://biohub.ai/esm/protein/atlas) (a dataset of 6.8B sequences and 1.1B predicted structures). MIT licensed, weights on [HuggingFace](https://huggingface.co).

[Sheaf v0.11.0](https://github.com/korbonits/sheaf/releases/tag/v0.11.0) shipped both backends about twelve hours later. H100-verified, on PyPI, with quickstart examples for the [Ray Serve](https://docs.ray.io/en/latest/serve/index.html) path and the [Modal](https://modal.com) serverless path.

I want to talk about *why* this turnaround was possible, because the answer is the whole pitch for Sheaf.

## Why same-day was on the table at all

The reflexive read on "shipped support for X on the day X dropped" is that someone ground through a long day. That's not what happened.

I saw the announcement in my newsfeed over coffee. The basic "is this even servable today" check — MIT licensing, weight-downloadable variants — took about ten minutes. I started drafting the pull request from my iPhone using Claude Code in the gaps of the day, and only moved to my personal laptop after the kids were down for the night to finish it: the H100 smoke, the bug fixes it surfaced, the release cut, and the docs.

Phone-drafting Claude Code on real work isn't new for me — [I wrote about a smaller version of the pattern in March](/blog/2026-03-25-i-shipped-a-feature-from-my-phone-during-lunch). What's different now is the scope. A typed-contract serving layer makes a new-model integration bounded enough to draft on a phone, even when the change touches the request union, the backend registry, two new API modules, two new backends, and ~860 lines of tests. The integration was bounded by the time it took to:

1. Verify the upstream license, repo, and `from_pretrained` strings (no LLM-fabricated paths).
2. Write the new `ProteinLanguageRequest` / `ProteinLanguageResponse` and `StructureRequest` / `StructureResponse` Pydantic contracts.
3. Wrap `transformers.AutoModelForMaskedLM` (ESMC) and `ESMFold2InputBuilder().fold()` (ESMFold2) in two `ModelBackend` subclasses.
4. Write unit tests with mocks for both backends.
5. Wire the new request types into the `AnyRequest` discriminated union and the backend registry.
6. Run a smoke test on a real H100 — find the bugs that mocks couldn't.
7. Cut the release.

Every step is bounded because the contracts and the substrate already exist. The serving layer doesn't get rewritten when a new model arrives — `_SheafDeployment` already knows how to deploy a `ModelSpec`, batch requests, register metrics, surface OTel spans, hot-swap on update. The new code is a Pydantic class and a `predict()` method.

The [v0.1 post](/blog/2026-04-14-sheaf-vllm-for-non-text-foundation-models) made the bet that *getting the contracts right first* was worth more than shipping half-baked optimizations behind the wrong abstractions. Six weeks later, that bet keeps paying out. Same-day integration isn't a stunt — it's what falls out of the architecture when a new model is just another implementation of an interface you already have.

## Two new model categories, not one

ESMC and ESMFold2 are both "protein" models, but they sit at opposite ends of what "protein" can mean to a serving layer.

ESMC is a masked language model over amino acid sequences. Per-token logits, optional per-token embeddings. Ragged outputs — sequence 1 of length 53 and sequence 2 of length 197 produce tensors of different shapes that need to be sliced back out of a padded batch.

ESMFold2 is a structure predictor. Input: a sequence. Output: a 3D structure as a PDB or mmCIF text block, plus pLDDT confidence per residue, plus pTM/ipTM globally, plus optionally a PAE matrix. Inference-time scaling parameters (`num_loops`, `num_sampling_steps`, `num_samples`, `seed`) are first-class — they're knobs the caller wants to expose, not internal tuning.

Sheaf already had a `MOLECULAR` model type for ESM-3, which returns one pooled embedding vector per sequence. The temptation was obvious: reuse it for ESMC. Don't.

The response shapes are incompatible. `MolecularResponse.embeddings` is `list[list[float]]` — a per-sequence vector. `ProteinLanguageResponse.logits` is `list[list[list[float]]]` — per-sequence, per-token, per-vocab-position. Unifying them would force every caller to branch on `model_name` to interpret the shape, which is exactly the kind of thing a typed contract exists to prevent.

So `PROTEIN_LANGUAGE` is its own category. And `STRUCTURE` is its own category too — and it's the first one in Sheaf whose output is fundamentally non-tensor. A PDB block is text. Multi-chain inputs are a list of `ChainInput` objects. pLDDT comes back alongside the structure as side-channel data. None of this fits the embedding / classification / generation / forecast molds the rest of the type system uses.

This matters beyond ESMFold2. Boltz-1 and Chai-1 — the other open structure predictors people will want to serve — produce the same shape of output. They'll inherit the `STRUCTURE` contract on day one.

## The bugs the mocks couldn't catch

The unit tests for both backends pass. 27 new tests, all green. The integration was "done" by any reasonable definition — until the first smoke run on an actual H100 surfaced two real bugs neither suite would have ever caught.

**Bug one: ESMC's `MaskedLMOutput` has no `last_hidden_state`.**

The first version of `ESMCBackend._run()` did the obvious thing:

```python
out = self._model(**inputs)
embeddings = out.last_hidden_state  # AttributeError on a real model
```

`transformers.AutoModelForMaskedLM` returns a `MaskedLMOutput`, which exposes `.logits` and (when requested) `.hidden_states` — but **not** `.last_hidden_state`. That attribute lives on `BaseModelOutput`, returned by `AutoModel`. The mocked tests happily passed because `MagicMock.last_hidden_state` auto-creates an attribute on access. Only the real model's `__slots__`-equivalent told the truth.

The fix is two lines: force `output_hidden_states=True` whenever embeddings are requested, then read uniformly from `hidden_states[-1]`. The test mocks were updated to mirror the real `MaskedLMOutput` shape (no `last_hidden_state`, full `hidden_states` tuple) so this can't regress silently.

This is the same lesson from the [v0.8 LoRA bug](/blog/2026-04-14-sheaf-vllm-for-non-text-foundation-models): *mock-only tests prove "we wrote the right code"; only real-deps tests prove "the right code does what we think."* Both are necessary. Neither is sufficient.

**Bug two: ESMFold2's pLDDT is on `[0, 1]`, not `[0, 100]`.**

AlphaFold returns pLDDT on `[0, 100]`. ESMFold v1 returns pLDDT on `[0, 100]`. Every public visualisation I've seen — PyMOL color ramps, the structures in the EBI structure viewer — assumes `[0, 100]`.

ESMFold2 returns pLDDT on `[0, 1]`.

Nothing about this is wrong, exactly — it's just different from the existing convention. And I could not have known until the H100 smoke run came back with values like `0.2465` where I'd been expecting something in the 50–95 range.

The right thing to do here was nothing in code. Sheaf's convention is *validate at the boundary, don't transform inside backends*. So `StructureResponse.plddt` passes through faithfully, and the docstring on the field explicitly documents the `[0, 1]` scale and tells callers to multiply by 100 themselves if they want the conventional values.

One wrinkle worth flagging: upstream's `result.complex.to_mmcif()` *does* rescale to `[0, 100]` when it writes the B-factor column. So the mmCIF that Sheaf returns in `StructureResponse.structure` colors correctly under PyMOL's default `spectrum b` ramp with no client-side adjustment. The `[0, 1]` quirk is only on the raw `.plddt` tensor. Two surfaces, two scales, both consistent with upstream — neither rescaled by Sheaf. The same finding got added to [ADR-0001](https://github.com/korbonits/sheaf/blob/main/docs/adr/0001-esmc-esmfold2-integration.md) so the next person hitting it has the receipts.

If we'd silently multiplied by 100 in the backend, we'd have papered over a real upstream behaviour with a magic number. Then the day ESMFold2's pLDDT scale changes — or the day Boltz-1 lands using yet another convention — the fix becomes a backwards-compat shim with version flags. Faithful pass-through means every model behaves like itself, and callers learn the actual shape of what they're using.

## The Modal gotcha

`sheaf.modal_server.ModalServer` is a parallel path to the Ray Serve `ModelServer` — same `ModelSpec`, same backends, same Pydantic contracts, deployed as a Modal app instead of a Ray cluster. It's the "zero-infra GPU deployment" path.

It also has *its own* `AnyRequest` discriminated union, deliberately separate from `sheaf.api.union.AnyRequest`. This is intentional — Modal containers shouldn't pull Ray as a transitive dep just to import the request types. But it means adding a new model type requires updating **both** unions, plus the backend registry imports inside `_build_asgi_app`.

The v0.11 PR initially missed the Modal-side update. ESMC and ESMFold2 deployed fine on Ray Serve; they 422'd on Modal because the parallel union didn't know how to discriminate the new `model_type` field values. The follow-up commit (`b63557a`) plumbs `ProteinLanguageRequest` and `StructureRequest` into Modal's union and wires `esmc` + `esmfold2` into `_build_asgi_app`'s registry imports.

This is the kind of bug a stricter abstraction would have prevented. It's also the kind of bug that gets in the way of *shipping* if the abstraction is too tight too early. Two unions + a checklist is the right trade-off for now; if Sheaf grows a third deployment shape (Knative? Bento?), I'll consolidate.

## What v0.11 actually adds

In total — straight from the release diff:

- **`sheaf.api.protein_language`** — `ProteinLanguageRequest` / `ProteinLanguageResponse` for ESMC. Per-sequence ragged outputs with `seq_lens` so callers can slice padded tensors back out.
- **`sheaf.api.structure`** — `StructureRequest` / `StructureResponse` for ESMFold2 and future structure predictors. Multi-chain input, inference-time scaling parameters as first-class fields, PDB-or-mmCIF text output, pLDDT/pTM/ipTM/PAE side channels.
- **`sheaf.backends.esmc.ESMCBackend`** — wraps `transformers.AutoModelForMaskedLM` on `Biohub/ESMC-6B` (the only weight-downloadable variant; 300M and 600M ship through Biohub's Forge API and raise `NotImplementedError` with a pointer to the ADR).
- **`sheaf.backends.esmfold2.ESMFold2Backend`** — wraps `ESMFold2InputBuilder().fold()`, exposing `num_loops` / `num_sampling_steps` / `num_samples` / `seed` to the request.
- **`[protein]` install extra** — pins `esm @ git+https://github.com/Biohub/esm.git@81b3646c…` (no PyPI release yet; matches the SHA in Modal's reference example). Mutually exclusive with `[molecular]` at install time, declared in `[tool.uv].conflicts`, because both ship a package named `esm` from different orgs.
- **Three quickstarts** — `examples/quickstart_protein_language.py` (Ray Serve + ESMC), `examples/quickstart_structure.py` (Ray Serve + ESMFold2), `examples/quickstart_protein_modal.py` (Modal + ESMFold2 on H100 with a persistent weights volume).
- **[ADR-0001](https://github.com/korbonits/sheaf/blob/main/docs/adr/0001-esmc-esmfold2-integration.md)** — full verification trail and design rationale, including the pLDDT-scale finding and why `PROTEIN_LANGUAGE` and `STRUCTURE` are separate categories.

682 + 27 = 709 tests passing. End-to-end Modal H100 smoke: 53-residue fold → pTM 0.2465, 43,088-char mmCIF, no crashes.

![ESMFold2 prediction of a 53-residue protein, rendered as a cartoon and colored by per-residue pLDDT confidence (low → high). Produced end-to-end by Sheaf v0.11.0 on a Modal H100.](/images/esmfold2-53-residue-fold.png)

```bash
pip install 'sheaf-serve[protein]==0.11.0'
```

## Why this matters past today

[vLLM](https://github.com/vllm-project/vllm) made text-LLM serving fast and uniform because all autoregressive text models share a compute pattern. Everything else — protein, time series, tabular, diffusion, geospatial — is still in the era of "every model rolls its own glue code." That's bad for the people training models (their work doesn't get deployed because deployment is hard) and bad for the people deploying them (every new release is a project).

The bet behind Sheaf is that you can fix this from the contracts down. Each model type gets a typed request/response. The serving layer optimises per type independently. New models land as backends, not as new infrastructure.

Today is what that bet looks like when it pays out. A morning model release. An afternoon of integration work. A version bump that evening. No serving rewrite. No "we'll add support in the next quarter." Two new model categories, both reusable for the next protein paper that drops, queued up so the next time *isn't* a project either.

If you're working on the model side of the pipeline and looking at deployment as the part you'll figure out later — this is the part Sheaf is trying to make boring. [Repo here](https://github.com/korbonits/sheaf). Issues and PRs welcome.

---

**Further reading.** Candido et al., ["Language Modeling Materializes a World Model of Protein Biology"](https://biohub.ai/papers/esm_protein.pdf) (Biohub, 2026) — the preprint behind the ESMC / ESMFold2 release. [Biohub/esm](https://github.com/Biohub/esm) — the source repo, where the README's local-inference example is the canonical entry point for self-hosted use. [ADR-0001](https://github.com/korbonits/sheaf/blob/main/docs/adr/0001-esmc-esmfold2-integration.md) — Sheaf's full verification trail and design rationale for the integration, including the upstream-license check, the `MaskedLMOutput` finding, and the pLDDT-scale empirical note.

---

*korbonits.com is my personal blog. I write about ML, software, and books.*
