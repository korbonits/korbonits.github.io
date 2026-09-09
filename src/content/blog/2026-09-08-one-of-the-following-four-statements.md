---
title: "One of the Following Four Statements"
description: "OpenAI did not prove Navier–Stokes, and it also did not solve the wrong problem. It proved alternatives (C) and (D), which Fefferman put in the official statement on purpose in 2000 — using theorem statements DeepMind had already written."
date: 2026-09-08
draft: false
tags: ["ai", "mathematics", "verification", "lean", "navier-stokes"]
---

# One of the Following Four Statements

Two things every headline today got wrong, and they point in opposite directions.

The first: OpenAI did not prove the Navier–Stokes existence and smoothness conjecture. Nobody proved that a fluid stays smooth forever. That question — whether smooth initial data with no external force can run to infinity without breaking — is exactly as open tonight as it was yesterday.

The second, which is the one the correction pieces got wrong while correcting the first: OpenAI did not solve "the wrong problem" either. What it proved is alternatives **(C)** and **(D)** of the Clay Mathematics Institute's official problem description, written by Charles Fefferman in 2000, verbatim, including the forcing term that everyone is calling a loophole. The loophole was drafted into the problem on purpose, by the person the Clay Institute asked to draft it.

Both of those are checkable in about ten minutes, so let me show the checking rather than assert the conclusion.

## What the official statement actually says

Fefferman's [problem description](https://www.claymath.org/wp-content/uploads/2022/06/navierstokes.pdf) sets up the equations, fixes the decay conditions (4) and (5) on the initial velocity and the force, and then says this:

> To give reasonable leeway to solvers while retaining the heart of the problem, we ask for a proof of one of the following four statements.

Four. (A) and (B) are existence and smoothness on ℝ³ and on the torus, and in both of them Fefferman writes "Take *f*(*x*, *t*) to be identically zero." (C) and (D) are breakdown on ℝ³ and on the torus, and in both of them the sentence is:

> Then there exist a smooth, divergence-free vector field *u*°(*x*) on ℝ³ and a smooth *f*(*x*, *t*) on ℝ³ × [0, ∞), satisfying (4), (5), for which there exist no solutions (*p*, *u*) of (1), (2), (3), (6), (7) on ℝ³ × [0, ∞).

There is a smooth force in (C) and (D). It is not a smuggled-in weakening; it is in the text, it satisfies stated decay conditions, and there is no remark anywhere in the six pages excluding it or downgrading it. I read the whole document today looking for that remark, because I assumed it was there. It isn't. What is there is the opposite: Fefferman says the four-way choice is the leeway, and that the leeway retains the heart of the problem.

So "solves a version that doesn't qualify" is not right. Whether the Clay Institute *awards* on it is a separate question governed by the prize rules — publication, two years, community acceptance — and not by whether the statement is on the list. It is on the list.

What is fair to say is narrower and still large: forced blowup has been open for twenty-six years, everyone who worked on it thought it was hard, and (A) and (B) — the ones people mean when they say "the Navier–Stokes problem" — are untouched. Both sentences are true at once. That's an annoying shape for a headline and it's the actual shape of the week.

## The receipt

The artifact is public: [`openai/NavierStokesAndEuler`](https://github.com/openai/NavierStokesAndEuler), Apache-2.0, arriving as a single squashed commit at 06:57 Eastern this morning whose message is a period.

I cloned it. It is 616,276 lines of Lean 4 across 2,486 files — 404,224 lines in `NavierStokes/`, 211,578 in `Euler/` — with 36,156 lines beginning with `theorem` or `lemma`. It targets Lean 4.34.0-rc2 against Mathlib.

Every `sorry` in the repository, all four of them, is in `ComparatorChallenges/`, and the file's own header says they are "intentional `sorry` challenge placeholders." There are none in the proof. Its `formalization.yaml` — the same repo-root metadata format Anthropic shipped with Fermat four days ago, which is now apparently a standard[^yaml] — records `sorry_count: 0` and lists, for each of the four main results, the axioms used: `propext`, `Classical.choice`, `Quot.sound`. Lean's three. Nothing added.

Here is what was proved, from `NavierStokes/ComparatorSolution.lean`:

```lean
/-- (D) Breakdown of Navier–Stokes solutions on ℝ³/ℤ³. -/
theorem navier_stokes_breakdown_periodic (nu : ℝ) (hnu : nu > 0) :
    ∃ (u₀ : ℝ³ → ℝ³) (f : ℝ³ → ℝ → ℝ³),
    InitialVelocityConditionPeriodic u₀ ∧ ForceConditionPeriodic f ∧
    ¬ (∃ v p, NavierStokesExistenceAndSmoothnessPeriodic nu u₀ f v p) := by
  exact ComparatorBridge.navier_stokes_breakdown_periodic nu hnu
```

For every positive viscosity. Not one viscosity, all of them, by a rescaling.

The detail I like most is one line down in `ComparatorTheorem.lean`, where the existential gets witnessed. The initial velocity supplied is `fun _ => 0`. The fluid starts at rest. Everything that happens to it is done by a smooth force with compact support in future time, which then switches off, and after it switches off there is no smooth solution on [0, ∞) at all. That is a very physical picture for something that will spend the next two years being argued about as a technicality.

## The question was already written, again

Two days ago I [wrote](https://korbonits.com/blog/2026-09-06-the-question-was-already-written/) that machines are extraordinary at filling in a proof when someone else has written the question, and that the writing of questions is still done by people, in public, before the answers exist. I meant it as an argument. Today it's a diff.

`NavierStokes/ComparatorDefinitions.lean` opens with a copyright header reading "Copyright 2026 The Formal Conjectures Authors" and a source link to a pinned commit of [`google-deepmind/formal-conjectures`](https://github.com/google-deepmind/formal-conjectures/blob/main/FormalConjectures/Millenium/NavierStokes.lean). So I fetched that file at that commit and diffed it against OpenAI's copy, normalizing whitespace, imports, the namespace, and Formal Conjectures' `@[category ...]` attributes.

The definitions are identical. `divergence`, `IsOnePeriodic`, `InitialVelocityCondition`, `ForceConditionDecay`, the solution structures, the periodicity of the pressure that DeepMind added from the Clay errata — all of it, unchanged. And the theorem statements OpenAI proved are, character for character, the statements DeepMind wrote and left as `sorry`:

```lean
theorem navier_stokes_breakdown_periodic (nu : ℝ) (hnu : nu > 0) :
    ∃ (u₀ : ℝ³ → ℝ³) (f : ℝ³ → ℝ → ℝ³),
    InitialVelocityConditionPeriodic u₀ ∧ ForceConditionPeriodic f ∧
    ¬ (∃ v p, NavierStokesExistenceAndSmoothnessPeriodic nu u₀ f v p) := by
  sorry
```

That is DeepMind's line. OpenAI replaced `sorry` with a term and shipped 616,276 lines underneath it.

This is the strongest form of the thing I have been circling since May, and it is stronger than I put it on Saturday. The reason nobody has to trust OpenAI's translation of Fefferman's English into Lean is that OpenAI didn't do the translation. A competitor did, months earlier, for an open-source library of conjectures, with no idea who would use it. The single hardest place for an error to hide — the statement — was written by a disinterested party in public before the answer existed.

I have been calling this the question-writing problem and treating it as the bottleneck. It is the bottleneck. It is also, apparently, solved often enough by people who are not trying to win anything.

## Three layers, one of which you can read

There is a third checking layer here that I haven't seen anyone mention, and it is the direct answer to the question I asked in [May](https://korbonits.com/blog/2026-05-28-who-verifies-the-verifier/).

`ComparatorChallenges/` holds the challenge files. Each is a self-contained statement of the theorem with `sorry` in place of the proof, and OpenAI's proof modules are structured so their import closure never touches them. You then run [Comparator](https://github.com/leanprover/comparator), which checks at runtime that the definitions the proof was built against are the same definitions as the independent reference, exports the whole thing with `lean4export`, and re-checks the kernel derivation with `nanoda_bin` — an external kernel implementation, not Lean's own.

So the trust chain is: Fefferman wrote the problem, DeepMind formalized it, Comparator checks that OpenAI proved DeepMind's statement and not a lookalike, and a second independently-written kernel checks the proof. Each layer is checked by someone who didn't build the layer below it.

And the part a person can actually read is 286 lines. That's `ComparatorChallenges/NavierStokes.lean` — the definitions and the two statements. If you read those 286 lines and agree they say what Fefferman says, and the tooling reports green, you are done, and the other 616,000 lines are not your problem.

I want to be careful about how much comfort to take from that, because it is exactly the trade I described on Saturday and then complained about. Nobody will read the mathematics. The mathematics is not readable — the file names alone (`ActualSignedUnmaskedBinding`, `NilpotentVolterra`, `TrueConeLoop`) tell you an agent named 2,486 files without a human in the loop. What you get is one bit. The bit is trustworthy in a way that a hundred-page manuscript reviewed over a weekend is not. It is also just one bit.

## The result nobody put in a headline

The repository contains a second theorem, and it is the more remarkable one.

`Euler/Solution.lean` claims finite-time blowup for the **unforced** three-dimensional incompressible Euler equations: smooth, compactly supported, nonzero, divergence-free initial data on ℝ³, a maximal lifespan T* with 0 < T* ≤ 1, no smooth solution past it, the C¹ norm's limsup equal to ∞, and the Beale–Kato–Majda vorticity integral divergent. No force anywhere in the statement. I checked; the solution class takes no forcing field.

Euler is not a Clay problem — Fefferman says so in the same document, in one sentence, right after the four alternatives, and notes that numerics have appeared to show blowup for decades while the instability made it impossible to conclude anything. Unforced smooth blowup on ℝ³ with no boundary has been open the whole time. If this one holds it is a bigger mathematical event than the Navier–Stokes result that got the press call, and it went out today in the same repository with no headline attached to it.

## Who wrote the question

The rest of this is not about mathematics, and it is the part that will matter in a year.

Yesterday — a day *before* OpenAI's announcement — Tristan Buckmaster and Levent Alpöge [posted](https://cims.nyu.edu/~tristanb/) three papers proving finite-time blowup with smooth forcing for the incompressible porous medium equation, 2D Boussinesq, and 3D Euler, with Lean formalizations. Terence Tao [wrote them up](https://terrytao.wordpress.com/2026/09/07/finite-time-blowup-with-smooth-forcing-term-for-the-incompressible-porous-medium-boussinesq-and-incompressible-euler-equations/) the same day, called the work a remarkable achievement, and described a route from their program to Navier–Stokes — a route, with enormous technical difficulty, not an endorsement of anyone's manuscript, and certainly not of a manuscript that did not exist when he wrote.

Buckmaster's accompanying [statement](https://cims.nyu.edu/~tristanb/statement.pdf) is four pages and everyone arguing about this week should read it before arguing further. Three things in it.

First, the credit. "The program this fits into was not started by us nor was it proposed by a Large Language Model." He assigns the basic idea to Diego Córdoba and Luis Martínez-Zoroa, who spent several years constructing forced blowups with rough forcing; what he and Alpöge did, with heavy LLM help, was push it to smooth forcing and to Euler. He adds that he believes Martínez-Zoroa deserves a Fields Medal. So the question was already written here too, twice over — once as a Lean statement by DeepMind, once as a research program by two humans over several years, and the models filled in both.

Second, the timeline, which he states as fact and I am repeating as his account. A year of slow work; real progress on August 15th; Lean verification on August 22nd. On September 3rd, hearing a rumor, he emailed a mathematician at OpenAI to say the work was personal, not institutional, and would be posted shortly. He was told on September 6th that an internal model had proved forced blowup, "option c and d in Fefferman." He asked when OpenAI's first prompt had been sent; he writes that the question "was not answered directly by OpenAI for some time," and that it was eventually agreed the first prompt went out in the past few days, after information about his work reached OpenAI. He asked whether the model had been trained on or had access to their Codex sessions, into which they had put every draft of the project. He reports being told the model did not look up user data, and receiving no answer on training. OpenAI's own public statement, per reporting I could not fetch directly, is that while unlikely, they cannot rule out that de-identified data derived from that usage helped improve their models.

He also describes two proposals about authorship, an exchange he quotes as "Why would you ruin your career?" and "If you don't want me to be nice, then I don't have to be nice," and a text suggesting he was not "being fully rational." Sébastien Bubeck has publicly called these characterizations false and inflammatory. I have no way to adjudicate that and I am not going to pretend otherwise.

Third, and this is Buckmaster being more careful than anyone reacting to him: "I have not seen OpenAI's proof. I do not know what their model did, or how. I do not know whether our data was used. I am not accusing anyone of anything." And: "If indeed an OpenAI model did close the gap to Navier–Stokes, that is a remarkable thing and it should be said loudly, by them, with the history intact."

One factual juxtaposition, offered in that same spirit. Buckmaster writes that he and Alpöge also believe they have blowup for hypo-dissipative Navier–Stokes, unreleased because the Lean verification hadn't finished, and that he mentions it "because it is suggestive of a path to unforced Euler." Unforced Euler is the second theorem in OpenAI's repository. I don't know what that means. I am recording that both sentences are true and were written a day apart.

## What's actually left

Four months ago the problem was that a hundred-page proof needed nine experts and a weekend. That problem is gone. Today an artifact landed with zero `sorry`s, three standard axioms, a statement written by a competitor before the answer existed, and an independent kernel to re-check the derivation, and I could establish most of that from a laptop in an afternoon without understanding a line of the mathematics.

The regress ran out of places to go, and what's underneath it is not a mathematical problem at all. Lean will tell you that a statement is true. It will not tell you who first had the idea, or whose drafts were in the training data, or which of two labs to believe about a date. Comparator checks that you proved the right theorem. Nothing in the stack checks that you were the one who found it.

Córdoba and Martínez-Zoroa opened this route years ago and are, as far as I can tell, not quoted in a single story today.

---

*What I ran: `git clone` of `openai/NavierStokesAndEuler` at commit `8937a8f`; `rg` and `fd` for the line counts, file counts, declaration counts and every `sorry` in the tree; `curl` for the Formal Conjectures source at the pinned commit `8bf45ed`, and a normalized `diff` of it against `NavierStokes/ComparatorDefinitions.lean`; and the Clay PDF, extracted to text and read in full. Every quotation from the repository, from Fefferman, and from Buckmaster's statement is from those files as of today.*

*What I did not run: the build, Comparator, `lean4export`, or `nanoda`. 616,276 lines of Lean against Mathlib is not something I can check on this machine, so every claim here that the proof is correct is a claim about what the repository reports about itself, checked for internal consistency — not a claim I verified. Same distinction as [last time](https://korbonits.com/blog/2026-09-06-the-question-was-already-written/), and it costs me something again. I also could not fetch openai.com directly — it returned 403 — so the compute figures circulating today (10,000 agents, 88 hours, a 100-to-165-page writeup, 17 hours of formalization) are reporting, not primary, and I have deliberately built no argument on them.*

*Written with Claude, which cloned the repository, ran the diff against Formal Conjectures, and extracted the two PDFs. Don't trust it, and don't trust me. This one you can go run yourself, and unlike last month it fits on a laptop.*

[^yaml]: `formalization.yaml` v0.4, against a schema at `mathlib-initiative/formalization.yaml`. Anthropic's Fermat repository shipped one on September 4 and OpenAI's shipped one on September 8, both with `review: status: "self-assessed"` and an empty or absent reviewer list. Four days is a short time for a convention to appear, and "self-assessed" is doing a lot of work in it.
