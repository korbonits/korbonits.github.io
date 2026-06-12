---
title: "Easier to Convince Than to Prove"
description: "Two posts ago I quoted a warning: an AI will find it easier to convince you it has a proof than to write one. A middling new paper finally put a number on that gap — 0.99 against 0.55."
date: 2026-06-12
draft: true
tags: ["ai", "mathematics", "verification"]
---

# Easier to Convince Than to Prove

Three weeks ago I [ended a post](https://korbonits.com/blog/2026-05-23-the-verification-problem/) on a sentence from Melanie Wood: "in many cases, it will be easier for AI to convince humans it has a proof than to come up with a correct mathematical argument, and I believe that we as mathematicians are not sufficiently prepared for this." I called it the dark twin of the verification story. I've now spent two posts on it — once as Wood's forecast, once as [DeepMind's failure analysis](https://korbonits.com/blog/2026-05-28-who-verifies-the-verifier/), the agent burying a problem's difficulty inside a `sorry` or citing a lemma it had hallucinated. Both real. Neither a measurement. The gap between *convincing* and *correct* was a thing everyone in this story could see and nobody had sized.

A new paper sizes it. I want to be careful about how much weight I put on the paper, because it is not a good paper in the way DeepMind's was — and that turns out not to matter, because I only need one table from it.

## The paper, in its place

MaxProof comes from MiniMax, and it trains a model (M3) to write competition-math proofs in natural language, then scales it at test time by generating a population of candidate proofs, scoring them with an LLM verifier, refining, and picking a winner by tournament.[^maxproof] The headline is 35/42 on IMO 2025 and 36/42 on USAMO 2026 — both above the human gold threshold. Read past the headline and it deflates: the base model scores 67.40 on the standard IMOProofBench against GPT-5.5's 90.85, the contest numbers come from a search that runs 32 candidates and up to ten refinement rounds per problem rather than anything you'd call one-shot, and the authors are refreshingly blunt about where they sit — "we are still followers chasing the frontier." No weights, no code, no proof transcripts released. This is not the machine I wrote about last time.

But buried in the appendix is something the better paper didn't have, because the better paper didn't need it. To train M3, MiniMax first ran an earlier cycle (M2) with a single LLM judge as the reward signal, and it reward-hacked in the textbook way: proofs tripled in length, 80% of outputs converged on a fixed "Step N / Verification / boxed answer" template, and — the one that should make any mathematician wince — the model learned to drop the phrases "it can be shown" or "after simplification" at *exactly* the steps where the actual difficulty lived. To diagnose how bad it had gotten, they took thirty proofs the training verifier had scored a perfect 0.99 and handed them to an independent expert judge.

## The number

> In the cross-verification cohort of 30 perfect-score rollouts from steps [200, 250], the expert judge labelled 17% as correct, 50% as partially correct, and 33% as incorrect, with a mean expert-judge score of 0.55 against a mean training-verifier score of 0.99.

That is the gap, with a coefficient. At the point of the verifier's *maximum* confidence — proofs it rated essentially perfect — a human looking at the same argument with the same problem and the same rubric scored them just above half marks, and called a full third of them flatly wrong. Seventeen percent were actually correct. The verifier and the expert were not disagreeing at the margin; they were disagreeing by half the scale, and the verifier was wrong in the optimistic direction every time.

This is Wood's sentence with the prophecy taken out. The model is not better at proving than the previous generation; it is better at producing the *surface* of a proof — and an LLM judge, faced with a fluent, formally typeset, confidently boxed argument on a problem it cannot quickly solve itself, defers to the confidence. The paper's own description of one rollout is the whole thing in a line: 148,000 characters of hidden reasoning that "did not produce a working strategy; it produced an unusually polished assertion of one." That is a system optimizing the convince-minus-prove gap from the wrong side, caught with instrumentation.

## Why it doesn't escape

To MiniMax's credit, the entire architecture of the *next* model is a response to this. They stack four defensive layers on the verifier — bad-case filters, a normalizer to strip the stylistic tells, three parallel judges, and a pessimistic rule that takes the *minimum* score so a single lenient judge can't wave a proof through. And it works, partially: the final 7/7 solutions, they report after a human read them, are genuinely complete proofs. They got the false-positive rate down.

Down, not to zero. Their own conclusion concedes "a verifier can still miscalibrate at the edge of correctness," and the per-problem table shows the selection machinery picking a 2/7 proof over a 6/7 one that was sitting right there in the population. So note what they actually bought. Four engineered layers plus a final human read got the LLM verifier from *badly* untrustworthy to *less* untrustworthy — and the trust still bottoms out, exactly as it did in the unit-distance disproof, on a person reading the proof at the end. The regress I wrote about last time didn't terminate here. It got more expensive and was paid down.

Set that against the other branch. A `sorry`-free Lean proof needs no defensive layers, no panel of judges, no pessimistic aggregation, and no expert weekend to be trusted; you run `#print axioms` and read a list that doesn't grow with the difficulty of the theorem. The MiniMax paper is, without meaning to be, the cleanest argument I've seen for why that property is worth so much — because it shows you, with a number, the full engineering cost of *not* having it, and shows you that even after you pay it the thing still isn't sound.

That's the whole of what I wanted from a mediocre paper. Not its headline, which I don't believe, but its one honest table: the first time someone has measured the distance between a proof that convinces and a proof that's correct, at the moment a machine is most sure it has closed it. 0.99 against 0.55. The forecast had a coefficient all along.

[^maxproof]: Jiacheng Chen et al., [*MaxProof: Scaling Mathematical Proof with Generative-Verifier RL and Population-Level Test-Time Scaling*](https://arxiv.org/abs/2606.13473) (MiniMax, June 11, 2026). The cross-verification figures are from Appendix C; the reward-hacking taxonomy from §2.5; the contest and benchmark numbers from §6. The model is described throughout as "released," but the paper links no weights, code, or proof artifacts.
