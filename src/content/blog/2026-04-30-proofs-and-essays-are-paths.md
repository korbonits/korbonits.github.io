---
title: "Proofs and Essays Are Paths: An LLM ↔ Prover Loop for Falsifying Hallucinations"
date: 2026-04-30
draft: true
description: "If a proof is a path through a formal space and an essay is a path through a semantic one, can we close the loop — translate LLM outputs into a theorem prover, build a knowledge graph of verifiable propositions, and use the gaps as a signal for hallucination? Notebook ideas from March 2024 expanding on the SMT-grounding piece."
tags:
  - machine-learning
  - llm
  - automated-reasoning
  - formal-verification
  - knowledge-graphs
  - personal
  - notebook
---

*Draft. Source material: handwritten notebook pages dated 2024-03-26, photographed as IMG_7026, IMG_7027, IMG_7032, IMG_7033, IMG_7034.*

## Why this post exists

The April 2026 post [I Was Thinking About LLM + Automated Reasoning Before It Was Cool](/blog/2026-04-02-i-was-thinking-about-llm-automated-reasoning-before-it-was-cool-and-i-wasnt-ready) describes the November 2023 sketch: feed LLM output to an SMT solver, eliminate quantifiers, return a verified answer. This post picks up the same thread four months later (March 2024), with two new pieces:

1. **The loop is bidirectional.** LLM output → theorem-prover input *and* theorem-prover output → LLM input. AR is essentially first-order logic; we can quantize LLM outputs to binary-ish representations, feed them in, and back-propagate a "logical-learner" loss.
2. **Hallucinations are gaps in a path.** A proof is a path through a space whose connections can be formally verified. An essay is the same shape but with looser verification. If we lift LLM-generated propositions into a knowledge graph of theorems, hallucinations show up as graph-paths whose edges *fail to verify*.

## The thread

### Logic as graphs, propositions as nodes

Sketch: each proposition is a graph node; each implication / equivalence is an edge. The LLM proposes nodes and edges; the theorem prover validates them. Over time the graph fills in.

### The fact-checking ladder

Levels of difficulty for a single proposition:

1. *Is statement A true or false?* Embed it; look for nearby contradictions.
2. *Is `A → B` true?* One-step implication.
3. *Is `Γ(A, B)` true for some relation `Γ`?* Multi-step reasoning: induction, deduction, transduction, abduction.

The hard form is fact-checking with source attribution — surface-area / "1/φ" framing — which the notebook flags but doesn't solve.

### What changes when the loop closes

> *Rather than hallucinating the next LLM output, the chatbot should output an action — a construction in a reasoning space.*

The chatbot becomes a knowledge-graph builder, not a token sampler. Each reply extends the graph with a new node and edges. When the graph has a hole, the model says so. When the proposed extension fails to verify, the model says so.

## Connection to the geometry post

The companion post [The Geometry of Language](/blog/2026-04-30-the-geometry-of-language) makes the same observation about *essays*: they are paths through embedding space, and "strategic-ness" is a direction. Together the two posts argue that the right way to think about LLM outputs is **structural** (path-finding) rather than **stochastic** (next-token).

## TODO before publish

- [ ] Build the toy: a tiny LLM ↔ Z3 loop on arithmetic propositions, with a knowledge graph that grows as the loop runs
- [ ] Run on a public benchmark — TAM AI's hallucination eval, or something equivalent
- [ ] Cite Lean / AlphaGeometry / NVIDIA's work that the original notebook flagged
- [ ] Decide: one post or merge with [The Geometry of Language] into a longer essay
