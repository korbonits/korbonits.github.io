---
title: "The Geometry of Language: Embeddings as Manifolds, Writing as Geodesics"
date: 2026-04-30
draft: true
description: "What if 'good writing' is a direction in embedding space, and the model's job is to take the geodesic toward it? March 2024 notebook pages on Amazon Titan embeddings, question-manifolds, the curse of dimensionality, and a six-step experiment plan for treating strategic writing as a path-finding problem."
tags:
  - machine-learning
  - llm
  - embeddings
  - geometry
  - personal
  - notebook
---

![Notebook page on Amazon Titan embeddings, manifolds, and the curse of dimensionality, March 26 2024](/images/notebook-geometry-7028.jpg)

*A page from the same notebook as the SMT-grounding post, five months later.*

What if "good writing" is a direction in embedding space, and the model's job is to step toward it without drifting?

That's the thesis the notebook was reaching for, in the technical vocabulary that was newly making sense to me in March 2024 — submanifolds, geodesics, tangent maps, intrinsic dimensionality. The math is the prerequisite, not the point. The point is that *editing* — making writing better — has a geometric structure if you take the embedding space seriously, and once you see it that way, "make this paragraph more strategic" stops being a vibes operation and starts being a path-finding problem with a defined search direction and a defined cost.

## The strategic-writing experiment ("WL06")

![The six-step WL06 experiment plan](/images/notebook-geometry-7031.jpg)

The whole post earns this section. Six steps, written down at the bottom of a notebook page on March 26, 2024:

1. **Strategy as a vector / direction / submanifold** in embedding space.
2. **Take an utterance** and project it.
3. **Compute angle / dot product / covariant derivative** between the utterance and the strategy direction. That's a [scale] of "how strategic is this writing."
4. **Find a geodesic path** in the embedding space toward "more strategic" without drifting on other axes (meaning, grammar, register).
5. **Map interpolated points back to tokens.** Output: rewritten utterances guaranteed to be more strategic.
6. **Return the diff** — the difference between the input and the geodesic-step output — as guidance for the writer.

The killer line at the bottom of the page: *"Isn't a proof a path? Isn't an essay a path?"* That's the link to the companion post in this stack ([Proofs and Essays Are Paths](/blog/2026-04-30-proofs-and-essays-are-paths)).

## What the math has to do with it

The six-step plan above presupposes a few things about embedding spaces that are not free:

- **Amazon Titan embeddings → custom embeddings → geodesic RAG.** The closest thing an LLM gives you to a "concept" is a logit, an `x ∈ ℝⁿ`. The bet of the post is that you can take that seriously and ask geometric questions about it.
- **Curse of dimensionality + intrinsic dimensionality.** In high dimensions, everything is in a corner — every two random vectors are nearly orthogonal, every distance is nearly the same. The geodesic-step idea only works if the *intrinsic* dimensionality of "language relevant to a given task" is much lower than the ambient embedding dim, so that submanifold structure is actually recoverable.
- **Question manifolds and answer manifolds.** Sketch: $g: M_q \to M_a$, with $T_qM \to T_aM$ as the linear map between tangent spaces. Treat Q/A pairs as edges in a knowledge graph and use embeddings to detect when two questions are semantically equivalent.
- **TDA on knowledge graphs.** Is a "hole" (in the topological-data-analysis sense, with Betti numbers) a knowledge gap? Where the manifold has no answer, we can detect it.

If any of those four bullets fail empirically — if intrinsic dim turns out to be too high, if the question/answer manifolds aren't smooth, if there's no useful map back from interpolated embedding to token sequence — the WL06 plan doesn't work. The next TODO is to find out which of them fail first.

## Connection to the SMT post

For context: in April 2026 I published [I Was Thinking About LLM + Automated Reasoning Before It Was Cool](/blog/2026-04-02-i-was-thinking-about-llm-automated-reasoning-before-it-was-cool-and-i-wasnt-ready) — a November 2023 notebook entry about grounding LLMs with SMT solvers. The notebook entry this post is built around is the *next one in the same vein*, five months later, while reading about Amazon Titan embeddings and Bedrock fine-tuning. Same notebook, different angle: instead of grounding LLMs in formal logic, treat their embedding spaces as manifolds you can navigate geometrically.

The two posts argue the same shape — *paths through structured spaces beat stochastic next-token sampling* — to two different audiences. The proofs/essays post talks to the formal-verification community; this one talks to the embeddings/manifold-learning community.

## Prior art the notebook was implicitly arguing with

The notebook didn't cite anything — it was a sketch — but the ideas don't come from nowhere.  The two most relevant published threads:

- **Pope, Zhu, Abdelkader, Goldblum, Goldstein, *The Intrinsic Dimension of Images and Its Impact on Learning* ([arXiv:2104.08894](https://arxiv.org/abs/2104.08894), 2021).** Empirically demonstrates that natural images live on a much lower-dimensional manifold than their pixel-count would suggest, and that this intrinsic dimensionality directly affects how efficiently a neural network can learn the data.  The argument generalizes cleanly to text embeddings — which is the implicit bet the notebook is making.
- **Park, Choe, Jiang, Veitch, *The Geometry of Categorical and Hierarchical Concepts in Large Language Models* ([arXiv:2406.01506](https://arxiv.org/abs/2406.01506), 2024; ICLR 2025).** Formalizes how categorical and hierarchical concepts map to geometric structures (polytopes and vector subspaces) in LLM representation spaces, validated across Gemma and LLaMA-3 with 900+ WordNet concepts.  This is the closest thing the field has to "what the notebook was sketching, done rigorously."

"Geodesic RAG" as a phrase isn't a paper — it's the notebook's own coinage for *retrieval that respects the manifold's geometry rather than treating the embedding space as flat Euclidean*.  Open question whether anyone has shipped that.

## TODO before publish

- [ ] Make the six-step experiment plan into actual code (toy run on a small embedding model)
- [x] Cite the relevant prior art: Pope et al. 2021 (intrinsic dimensionality) and Park et al. 2024 (geometry of LLM concepts) added above; "geodesic RAG" flagged as the notebook's own coinage
- [ ] Draw the diagrams from the notebook properly (not just include the photo)
- [x] Tighten the framing — restructured: thesis paragraph + WL06 experiment now lead the post; the manifold math demoted to "What the math has to do with it" as evidence the WL06 plan depends on; SMT-post connection moved to a final cross-link section.
