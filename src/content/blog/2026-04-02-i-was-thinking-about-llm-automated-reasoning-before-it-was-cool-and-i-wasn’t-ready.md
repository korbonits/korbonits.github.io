---
title: "# I Was Thinking About LLM + Automated Reasoning Before It Was Cool (And
  I Wasn’t Ready)"
date: 2026-04-02
draft: true
description: "  *A notebook entry from November 2023, a Simons Institute talk,
  and why formal verification might be the most underrated idea in AI right
  now*"
---
I have a bad habit of writing ideas in notebooks and then losing them to time. Not losing them physically — I keep the notebooks — but losing the thread. The idea sits there, perfectly preserved in orange ink, while the moment passes.

In November 2023, I wrote this:
> *Automatically using Claude + prompt engineering into an SMT solver’s syntax. Then — it’s like the text → “say a problem”: you can create the text + propositions. Satisfy the proof. Eliminate unknowns, quantifiers — re-solve, substitute, return to the agent.*

Then, a line that still stings a little:
> *It seemed too far out.*

It wasn’t. And I knew it wasn’t, even then. I just wasn’t ready to follow it.

## The Thread That Started in 2004

To understand why that notebook entry matters to me, I have to go back twenty years.

In fall 2004, I took a mathematical logic class almost by accident. I was a math undergraduate at the University of Chicago, and the course covered propositional and first-order logic, proof systems, completeness, decidability. Dense, abstract, occasionally beautiful. It forced me to think about what it means for something to be *provably true* inside a formal system — not just probably true, not convincingly true, but *verified*.

I didn’t know what to do with that at the time. But it lodged somewhere.

A few years later, graph theory pulled me in. Specifically, the Hadwiger-Nelson problem — a question about the chromatic number of the plane that has been open since 1950. How many colors do you need to color every point in the Euclidean plane such that no two points exactly distance 1 apart share the same color? It sounds simple. It is not simple. I encountered it as a second-year student in 2006, working with Ian Biringer. Graphs got me into coding. Coding got me into machine learning. ML occupied the next decade of my professional life.

So when I sat down in November 2023 to watch a Simons Institute talk — yes, the Jim Simons — and the speaker mentioned the Hadwiger-Nelson problem in the context of automated reasoning, something reconnected. Twenty years of threads, suddenly visible at once.

## What I Was Actually Describing

The idea I sketched in my notebook wasn’t novel in isolation. SMT solvers — Satisfiability Modulo Theories solvers, tools like Z3 — have existed for decades. They take formal logical statements and determine whether they can be satisfied: whether there exists an assignment of variables that makes the whole expression true. They’re used in formal verification of hardware and software, in program analysis, in security research.

What I was gesturing at was a pipeline:

1. Take an LLM and a problem statement in natural language

1. Translate the problem into SMT solver syntax via prompt engineering

1. Run the solver — get a definitive yes/no/unknown, not a confident-sounding guess

1. Feed the result back to the agent as grounded truth

1. Let the agent reason forward from there

The key word is *grounded*. The solver doesn’t hallucinate. It either finds a satisfying assignment or it doesn’t. Plugging an LLM into that substrate — using the LLM as a translator between human language and formal syntax, then letting the formal system do the verification — is a qualitatively different approach to reliability than RLHF, chain-of-thought, or retrieval augmentation.

To make this concrete, here’s a minimal example of what that translation layer looks like. Say you want to verify a simple logical constraint — “if a system is in state A and condition X holds, then output must be Y”:

```smt2

; SMT-LIB encoding of a simple implication

(declare-const state_A Bool)

(declare-const condition_X Bool)

(declare-const output_Y Bool)

; Assert the constraint: A ∧ X → Y

(assert (=> (and state_A condition_X) output_Y))

; Now check: can we have A and X true, but Y false?

(assert (not output_Y))

(assert state_A)

(assert condition_X)

(check-sat)

; Returns: unsat — meaning the constraint holds, no counterexample exists

```

The LLM’s job is to take a natural language problem statement and emit something like this. The solver’s job is to tell you whether it’s satisfiable. The agent’s job is to act on the result. Each layer does what it’s actually good at.

I wrote “it seemed too far out” because, in November 2023, I couldn’t see a clean path from idea to implementation. The translation problem felt hard. Getting an LLM to reliably emit valid SMT-LIB syntax for nontrivial problems felt like it would require more reliability than LLMs had. It was a chicken-and-egg situation.

## What Closed the Gap

LeanDojo changed my thinking when I finally caught up with it.

Lean is a proof assistant — a formal system in which you can write mathematical proofs that a machine can verify. LeanDojo is a toolkit that lets language models interact with Lean: retrieving premises, generating proof steps, and checking them against the formal system in a closed loop. It’s exactly the architecture I was sketching, but for theorem proving rather than SMT solving, and built with enough infrastructure to actually work.

The lesson from LeanDojo isn’t that the specific tool is the answer. It’s that the translation problem is tractable. LLMs are good enough — and getting better — at serving as the bridge between natural language and formal systems. The formal system provides the ground truth. The LLM provides the flexibility and expressivity. Together, they’re more reliable than either alone.

This pattern is now showing up across the research landscape. Formal verification for code generation. Constrained decoding into structured logical forms. Neurosymbolic approaches that alternate between neural pattern matching and symbolic reasoning. The “too far out” idea from my November 2023 notebook is now a legitimate research direction with conference papers and GitHub repos.

## Why Hallucination Is Really a Grounding Problem

I think about hallucination differently than most framing I read.

The standard framing is that hallucination is a training problem — models learn to generate plausible-sounding text and sometimes plausible-sounding text is false. That’s true, but it’s downstream of something more fundamental: LLMs have no mechanism for being *wrong in a way that costs them anything*. There’s no falsifiable substrate. The model generates tokens, and those tokens either satisfy the user or they don’t, but the model itself receives no signal during inference that distinguishes a confident truth from a confident falsehood.

Automated reasoning introduces exactly that substrate. When you route an LLM’s output through a solver or a proof assistant, you get a binary external verdict that doesn’t care about the LLM’s confidence. The model can be maximally certain that a proposition is satisfiable — the solver will tell you if it isn’t. That external check, fed back into an agentic loop, is a qualitatively different kind of grounding than asking the model to check its own work.

This is philosophically interesting to me beyond the engineering. The question of what makes a cognitive system *genuinely* reliable — as opposed to statistically reliable — maps onto older questions about intentionality and representation. But I’ll save that for another post.

## The Honest Admission

I didn’t follow the thread in November 2023 because I was in the middle of other things and because the path wasn’t clear. That’s the honest version.

The less honest version — the one I tell myself sometimes — is that it “wasn’t practical yet.” That’s true, but it’s also a convenient story. The practical gap between 2023 and now is real but not enormous. What actually happened is that I wrote the idea down, felt the satisfaction of having captured it, and moved on.

Notebooks are good for that. They let you feel like you’ve done something with a thought without actually doing anything.

What I should have done: found one small, well-scoped experiment. A single domain. A problem type with a known formal representation. An SMT solver I already knew how to invoke. A prompt that tried to translate from that domain into solver syntax. Run it a hundred times. Look at the failure modes. Write it up.

That would have been the post. Instead this is the post.

## Where I’m Watching Now

The AR × LLM space is moving fast enough that being a year late isn’t fatal, and slow enough that serious engineering contributions still matter.

The things I’m paying attention to:

- **Agentic AR loops at inference time** — not just verification after the fact, but solvers in the reasoning loop, with the LLM retrying and refining until the solver accepts

- **Domain-specific translation layers** — the general translation problem is hard; the problem of translating within a constrained domain (code correctness, database query validity, contract logic) is much more tractable

- **The LeanDojo lineage** — what the next generation of proof-assistant-integrated models looks like, and whether the approach generalizes beyond mathematics

- **Where this intersects with agentic systems** — an agent that can ground subgoals in a formal system before acting on them is a meaningfully more reliable agent

If you’re building in this space or thinking about it seriously, I’d genuinely like to know what you’re seeing. The idea that seemed too far out in November 2023 is now somewhere between “active research area” and “emerging engineering practice.” That’s a window.

And I still have the notebook.

-----

*This post is part of an ongoing series on ML infrastructure, inference systems, and the ideas that age well. If you found it useful, the best thing you can do is share it with someone who’s thinking about AI reliability seriously.*
 
