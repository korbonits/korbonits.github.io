---
title: "The Geometry of Language: Embeddings as Manifolds, Writing as Geodesics"
date: 2026-04-30
draft: true
description: "What if 'good writing' is a direction in embedding space, and the model's job is to take the geodesic toward it? March 2024 notebook pages on Tilden embeddings, question-manifolds, the curse of dimensionality, and a six-step experiment plan for treating strategic writing as a path-finding problem."
tags:
  - machine-learning
  - llm
  - embeddings
  - geometry
  - personal
  - notebook
---

*Draft. Source material: handwritten notebook pages dated 2024-03-26, photographed as IMG_7028 through IMG_7031 + IMG_7033.*

## Why this post exists

In April 2026 I published [I Was Thinking About LLM + Automated Reasoning Before It Was Cool](/blog/2026-04-02-i-was-thinking-about-llm-automated-reasoning-before-it-was-cool-and-i-wasnt-ready) — a November 2023 notebook entry about grounding LLMs with SMT solvers. That post ends with the realization that I'd left the idea in a notebook for two years.

This is the *next* notebook entry in the same vein — five months later, March 2024, while reading about Tilden embeddings and Bedrock fine-tuning. Same notebook, different angle: instead of grounding LLMs in formal logic, treat their embedding spaces as manifolds you can navigate geometrically.

## The thread

- **Tilden embeddings → custom embeddings → geodesic RAG.** The closest thing an LLM gives you to a "concept" is a logit, an `x ∈ ℝⁿ`. What if we took that seriously and asked geometric questions about it?
- **Curse of dimensionality + intrinsic dimensionality.** In high dimensions, everything is in a corner. But the *intrinsic* dimensionality of "language relevant to a given task" is much lower than the ambient embedding dim. Submanifold structure should be recoverable.
- **Question manifolds and answer manifolds.** Sketch: $g: M_q \to M_a$, with $T_qM \to T_aM$ as the linear map between tangent spaces. Treat Q/A pairs as edges in a knowledge graph and use embeddings to detect when two questions are semantically equivalent.
- **TDA on knowledge graphs.** Is a "hole" (in the topological-data-analysis sense, with Betti numbers) a knowledge gap? Where the manifold has no answer, we can detect it.

## The strategic-writing experiment ("WL06")

The prize at the bottom of the page is a six-step concrete experiment:

1. **Strategy as a vector / direction / submanifold** in embedding space.
2. **Take an utterance** and project it.
3. **Compute angle / dot product / covariant derivative** between the utterance and the strategy direction. That's a [scale] of "how strategic is this writing."
4. **Find a geodesic path** in the embedding space toward "more strategic" without drifting on other axes (meaning, grammar, register).
5. **Map interpolated points back to tokens.** Output: rewritten utterances guaranteed to be more strategic.
6. **Return the diff** — the difference between the input and the geodesic-step output — as guidance for the writer.

The killer line at the bottom of the page: *"Isn't a proof a path? Isn't an essay a path?"* That's the link to the next post in this stack ([Proofs and Essays Are Paths](/blog/2026-04-30-proofs-and-essays-are-paths)).

## TODO before publish

- [ ] Make the six-step experiment plan into actual code (toy run on a small embedding model)
- [ ] Cite the relevant prior art: geodesic RAG papers, submanifold-recovery work, anything on text-as-manifold
- [ ] Draw the diagrams from the notebook properly (not just include the photo)
- [ ] Tighten the framing — the "writing as geodesic" hook is the lead, not the manifold math
