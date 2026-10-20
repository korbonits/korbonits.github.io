---
title: "Ten of Four Hundred Sixteen"
description: "OpenAI's October release ships 416 Lean challenge statements. Ten are written purely in Mathlib's vocabulary. For the rest, the question was written by the party that answered it. I counted how many of the 372 results a disinterested registry had already stated, and how many it could."
date: 2026-10-19
draft: false
tags: ["ai", "mathematics", "verification", "lean", "formal-conjectures"]
---

# Ten of Four Hundred Sixteen

On October 6, OpenAI published 722 mathematics manuscripts from an unreleased internal model, grouped into 372 result families, in a public repository called [`openai/math`](https://github.com/openai/math). Three were withdrawn the next day for a sign error, so the count is now 719. The release's own history file says 300 of the 719 top-line results have Lean formalizations, about 42%.

I have spent this series on one question: who writes the question? In [August](https://korbonits.com/blog/2026-08-01-who-writes-the-question/) I said what was missing was "roughly a hundred lines of Lean per problem, written by somebody with no stake in the answer, before the answer exists." In [September](https://korbonits.com/blog/2026-09-08-one-of-the-following-four-statements/), OpenAI's Navier–Stokes proof turned out to prove, character for character, a statement DeepMind had written months earlier in [Formal Conjectures](https://github.com/google-deepmind/formal-conjectures). The question had been written in public, by a competitor, before the answer existed.

A release of 372 results is large enough to stop arguing from examples and count. So I counted.

## The receipt

The trust mechanism is the same as in September. The release ships a directory, `lean/ComparatorChallenges/`, with one self-contained challenge file per checked result. Each states the theorem with `sorry` in place of the proof. A JSON file next to it tells [Comparator](https://github.com/leanprover/comparator) which solution module proves it, which theorems to check, and which axioms are allowed.

At commit `1581d4c` there are 416 challenges. Every one allows exactly Lean's three standard axioms: `propext`, `Quot.sound`, `Classical.choice`. The statements total 90,693 lines, and the median challenge is 79 lines long.

That last number is almost exactly the "hundred lines per problem" I asked for in August. The difference is who wrote them.

I searched all 416 challenge files for any reference to Formal Conjectures, the way the Navier–Stokes definitions carried its copyright header in September. There are none. In September OpenAI proved DeepMind's question. In October it wrote its own 416.

## Who wrote the questions

The challenges sort into four groups.

**Ten are stated purely in Mathlib's vocabulary.** They import only Mathlib and declare no definitions of their own. The quasi-Riemann hypothesis is one, and it is the cleanest example of what this buys you:

```lean
theorem riemannZeta_ne_zero_of_seven_eighths_lt_re
    {s : ℂ} (hs : (7 / 8 : ℝ) < s.re) : riemannZeta s ≠ 0 := by
  sorry
```

`riemannZeta` is Mathlib's, written by people with no stake in this, years ago. Nothing in OpenAI's code can change what that statement means. If Comparator reports green, the theorem is the theorem.

**Two declare no definitions but import OpenAI's own modules,** from its algebraic geometry library. Both are supporting results, not headline ones. Comparator's README lists its first assumption as "the transitive closure of imports of `Challenge.lean` ... are controlled by you or trustworthy." In these two, the challenge's definitions come from the same library as the proof.

**394 declare their own definitions,** and Comparator checks that the solution uses the same definitions, constant for constant. That pins the proof to the question. But the question here is OpenAI's: the definitions of mixing, of a log-space machine, of a k-server policy, written by the party that answered them. Whether those definitions mean the famous problem is a judgment a human has to make by reading them, one challenge at a time.

**Ten use definition holes.** This is the one that surprised me. Comparator's README describes a field called `definition_names`, meant for challenges that leave a definition open for the solver to fill in. For a name listed there, Comparator checks that the solution's version has the same name, type, universe levels and safety level. It does not check the body. The README is blunt about the consequence: such challenges "can be gamed without additional oversight," and "all definition hole solutions **must** always be checked with an additional (potentially human) verifier."

Two of the ten are results people will care about. The challenge for Rokhlin's multiple-mixing problem lists `IsMixing` and `MixingOfOrder` as holes. The challenge for `L = RL = BPL` lists the definitions of the machine model and of `L`, `RL` and `BPL` themselves. In both, the challenge file gives the definitions full bodies. So the question is written, but Comparator is configured not to hold the solution to it.

I checked. In both, the solution's definitions are textually identical to the challenge's, so I found no problem. But nothing in the tool would have caught a difference. For a claimed derandomization of log space, the definition of log space is the whole question.

One more number, because it was a layer I praised in September. The Navier–Stokes repository re-checked its proof with `nanoda`, a second, independently written Lean kernel. In this release, `enable_nanoda` is true in 2 of the 416 configurations.

## What the registry already had

Formal Conjectures is the closest thing the field has to a disinterested question-writer: an open repository of conjectures stated in Lean, almost all with `sorry`, maintained by DeepMind and about 176 contributors. So the second count is simple: for each of OpenAI's 372 result families, was the underlying problem already stated there?

| Formal Conjectures `main`, October 19 | Families |
|---|---|
| Problem already stated | 46 |
| Stated in an open pull request | 30 |
| Not stated | 296 |

Of the 46, 44 were in the repository before October 6. Those are questions written before the answer existed, by people who did not know who would answer them, which is the September situation again. A few of the 46 are partial: the repository states the classical conjecture, and the release proves a different variant of it. The maintainers have already done the obvious follow-through. One pull request, [#6898](https://github.com/google-deepmind/formal-conjectures/pull/6898), marked 29 theorems solved and linked each to the release theorem its Comparator challenge checks.

Of the 30 open pull requests, 22 were opened after October 6. Those are questions written after the answer, which is better than nothing and not the same thing.

That leaves 296 families where nobody independent has written the question at all.

## What could be written

The last count is the one I find most useful. For each of the 296, can the question even be stated with the Mathlib that Formal Conjectures pins today?

| Stateable with the pinned Mathlib | Families |
|---|---|
| Yes, every notion already exists | 22 |
| With a few short local definitions | 108 |
| No, the theory is missing | 166 |

The 166 are not a failure of effort. Mathlib has no curvature, so most of Riemannian geometry is out. It has no hyperbolic groups, very little von Neumann algebra theory beyond the definition, and nothing like SLE. Any result about elliptic curves that needs a BSD formula is out. Those questions are waiting on libraries, not on volunteers.

The 22 are the interesting list. A few of them: the quasi-Riemann hypothesis, the rigidity of the Turing degrees (`∀ f : TuringDegree ≃o TuringDegree, f = OrderIso.refl _`, a single line), the Kervaire conjecture, Kadison's similarity problem, Kurosh's problem for division rings. These are famous problems with short, unambiguous statements, and OpenAI claims to settle all of them. As of this morning, Formal Conjectures stated none of them.

So I wrote one. The quasi-Riemann hypothesis is now [#7459](https://github.com/google-deepmind/formal-conjectures/pull/7459): the zeta statement, the uniform Dirichlet statement, and two short test lemmas, one showing that RH implies it with θ = 1/2. It is tagged `research solved` with a link to OpenAI's proof, following #6898. The pull request also says plainly that I did not rebuild that proof. The reviewers may prefer to wait for someone who has.

## What's left

In August the question-writing problem was a principle. Now it is a backlog with a size. About 130 OpenAI results can be stated today by someone with no stake in them, and mostly haven't been. About 170 more are waiting on Mathlib. And for the 394 challenges whose definitions OpenAI wrote, the job is smaller than writing questions from scratch but no less necessary: someone has to read 79 lines and decide whether they say what the paper's title says.

None of that work needs a frontier model or a 300 GB machine. It needs people who read Lean and care about the difference between a problem and a statement of it. That has been true all year. What changed on October 6 is that the queue got longer than any one person can get through.

The full table is [here](/assets/data/2026-10-19-openai-math-audit.tsv): all 372 families, their coverage, a stateability rating, the matching Comparator challenge, and a one-line note.

---

*What I ran: a sparse clone of `openai/math` at `1581d4c` (October 12); a script over all 416 `ComparatorChallenges/*.json` and `.lean` files for imports (414 import only Mathlib), axioms, `definition_names`, `enable_nanoda`, local definitions and line counts; `rg` across those files for any reference to Formal Conjectures; a whitespace-normalized text comparison of the hole definitions in the Rokhlin and `L = RL = BPL` challenges against their solution modules; `git log` on Formal Conjectures for when each covered problem was first added; and a scan of the files touched by all 704 open pull requests. The quasi-RH pull request builds with `lake --wfail`.*

*What I did not run: the build, Comparator, or `nanoda`, for anything in the release. Every claim here that a proof is correct is OpenAI's claim, not mine. The coverage and stateability columns are judgment calls made with `rg` and a reading of each result's one-paragraph description, not by compiling Lean. One batch of 62 rated "a few short local definitions" generously, mostly for complexity-theory results. I did not check whether the 44 pre-existing Formal Conjectures statements match OpenAI's challenge statements. That comparison is the obvious next step.*

*Written with Claude. Six parallel Claude subagents did the coverage and stateability classification across the 372 families. Claude wrote the scripts, did the clone, checked the specific numbers above against the files, and drafted this post. Don't trust it, and don't trust me. The table is linked, and every number in this post can be rerun from a laptop.*
