---
title: "In Decreasing Order of Digestion"
description: "Three results dropped today: unforced Euler, unforced Navier–Stokes, and the Generalized Riemann Hypothesis. The Riemann statement is one line of Mathlib over 17 million lines nobody has read. The Navier–Stokes theorem in Lean begins with a hypothesis, and the data that discharges it is a hash."
date: 2026-11-20
draft: false
tags: ["ai", "mathematics", "verification", "lean", "navier-stokes", "riemann-hypothesis"]
---

# In Decreasing Order of Digestion

Three things were posted this morning. Levent Alpöge, Tristan Buckmaster and Matei Coiculescu posted finite-time blowup for the unforced 3D Euler equations. Alpöge and Buckmaster posted finite-time blowup for the unforced 3D Navier–Stokes equations, which would disprove statements (A) and (B) of the Clay problem, the ones [I said in September](https://korbonits.com/blog/2026-09-08-one-of-the-following-four-statements/) were untouched. And Alpöge, who works at Anthropic, posted a four-page [note](https://alpo.ge/remark.pdf) that begins "I believe I have, working with Claude, discovered a proof of the Riemann Hypothesis," with a footnote linking fourteen pieces of a tarball.

Terence Tao [listed all three](https://terrytao.wordpress.com/2026/11/20/unforced-euler-singularity-unforced-navier-stokes-singularity-and-the-generalized-riemann-hypothesis/) "in decreasing order of how 'digested' the results currently are." That ordering is the right one, and it is not the order of the headlines.

None of these is verified. I have not built any of them. What I did is what I have done all year: read the statement files, because that is the part a person can read, and see who wrote the question.

The three answers are different, and the famous one has the best answer.

## Riemann: one line, seventeen million lines

The note claims RH, and GRH for Dirichlet L-functions, and says the Extended Riemann Hypothesis is expected "with more time." There is no paper. Alpöge says a readable one does not exist yet and that he declined to post an unreadable one:

> It being obviously inappropriate to attempt to post some massive unreadable travesty of "slop", I decided to only present Lean.

A footnote adds that he tried anyway, for fun, "and hit 250,000+ pages."

So the artifact is the Lean. This is the whole of `Main.lean`:

```lean
theorem riemannHypothesis : RiemannHypothesis := rhOfAll

theorem riemannZeta_eq_zero_re : RiemannZetaZerosOnLine := rhOfAll

theorem LFunction_ne_zero_of_half_lt_re : DirichletLNonvanishingRightOfHalf := allChi

theorem LFunction_eq_zero_re : DirichletLZerosOnLine := claimAll.2.1

theorem LFunction_eq_zero_re_of_isPrimitive : PrimitiveDirichletLZerosOnLine := critAll
```

The first line is the one that matters. `RiemannHypothesis` is not defined anywhere in the submission. It is Mathlib's:

```lean
∀ (s : ℂ), riemannZeta s = 0 → (¬∃ n : ℕ, s = -2 * (n + 1)) → s ≠ 1 → s.re = 1 / 2
```

Formal Conjectures has stated the Millennium problem as exactly `theorem riemannHypothesis : RiemannHypothesis := by sorry` since before any of this. In [October](https://korbonits.com/blog/2026-10-19-ten-of-four-hundred-sixteen/) I counted ten of 416 OpenAI challenges written purely in Mathlib's vocabulary and called the quasi-Riemann hypothesis the cleanest. This is that case with 7/8 replaced by 1/2. Nothing in the submission can change what the statement means. For the hardest problem on the list, the question-writing problem does not exist.

The other four statements are the author's, four short definitions in `Statements.lean`, built on Mathlib's `DirichletCharacter.LFunction`. They are short enough to read and I think they say GRH. They are not the same sentence as Formal Conjectures' `generalized_riemann_hypothesis`, which excludes trivial zeros by a set rather than restricting to the critical strip. Getting from one to the other is a functional-equation argument that someone should write down in Lean. As far as I can see nobody has.

Under those five lines is the rest. The numbers here are from an independent mirror, [`Deicyde/riemann-hypothesis`](https://github.com/Deicyde/riemann-hypothesis), which sprang up within hours with a Comparator setup the submission did not ship: 175,187 Lean files, 17,374,215 lines, 818,755 theorems and lemmas, no comments. Their text scan finds no `axiom`, no `sorry`, no `native_decide`. Their notes also say, correctly, "A text search is not a proof."

The verification notes are the best thing I read today, and they are careful in a way worth copying:

- The project pins Lean 4.33.1. The notes list which of this year's kernel soundness fixes that release includes and which it does not, and conclude that a pass by the 4.33.1 kernel alone "counts as limited evidence here." So the proof is built on 4.33.1 and judged by 4.36.0.
- Compiling the submission runs its code. So the challenge is exported from a clean clone that never compiled any of it, and the build runs in a container with no network.
- Of the five extra kernels `lake comparator --paranoid` offers, three cannot read this project for reasons unrelated to correctness. That leaves Lean's kernel, its hardened build, and nanoda.
- Andrew Sutherland, on the Lean Zulip, reports a solution export of 61.5 GB.

As of this evening their results section reads: "The build is in progress."

In September I wrote that what you get from a 616,000-line proof is one bit. This is 28 times larger, and the bit has not come back yet. If it comes back green, the situation is one mathematics has not been in before: a true sentence of Mathlib, and no human who can say why. Tao notes that this time there is at least "a human author to communicate with." The author's own note says Claude "far outpaced me near the finish line," and of parts of his timeline that he is "paraphrasing a 'slop' summary, which might make zero sense."

He does not read that as a loss, or not only as one. The sentence from the note I keep going back to:

> It is natural to feel that this is an end, in my case a loss of a principal guiding light, but what I saw during the course of progress so amazed me that I believe it is one dawn across science of something far greater.

This is someone who writes that he planned his student years around this problem. Everything else in this post is about what can be checked. That sentence can't be, and it is the one I would least want to be wrong.

## Navier–Stokes: the theorem starts with `h`

This one has a [79-page paper](https://github.com/tristanbuckmaster/navier-stokes/) and about 600,000 lines of Lean, so it looks like September. It is not like September, in two ways.

First, the question. In September OpenAI proved DeepMind's statement, character for character. Here the statement file is the authors' own: `Challenge.lean`, 110 lines, with its own `BreakdownOnSpace` and `BreakdownOnTorus`. I searched the statement files for any reference to Formal Conjectures and found none. Formal Conjectures does state (A) and (B). Nothing in this repository proves their negations as DeepMind wrote them.

Second, and more important, here is the theorem:

```lean
theorem breakdown_on_space (h : PeriodicSimilarityProfileExists) : BreakdownOnSpace := by
  sorry
```

It takes a hypothesis. *If* a smooth, nonzero, discretely self-similar profile exists, *then* there is blowup. Every headline theorem in the package has this shape: from a profile, from a weak profile, from a "certificate," from "cell system" data satisfying fifteen conditions. The repository's own `NOTICE` says it twice: "No certificate is given in this package." "No such data are given in this package."

The profile is the whole problem. Whether such a solution exists is an open problem in the literature the paper cites. The paper gets it from a certificate, an approximate solution found numerically and then checked against explicit inequalities. Where that is, from the abstract:

> The certificate will be released soon; its hash is `3f7a335b…`

And where the check is, from the README:

> That the certificate satisfies the conditions of Appendix B is verified by a rigorous computation, involving interval arithmetic (128 bits of precision, with the Arb library through FLINT), which is not part of the formalization and is not in this repository either.

So Lean, if it builds, certifies an implication. The antecedent is a file nobody outside has seen, checked by a program nobody outside has seen. I am not saying the antecedent is false. Computer-assisted proofs with interval arithmetic are an established method, and Buckmaster's line is fair: "once the numerical profile is in hand, the proof follows a well-trodden computer-assisted path." But the sentence "formalized in Lean" is carrying less than it did in September, and the difference is exactly the part that makes the theorem true.

Three smaller things, all stated by the authors. The Lean "was not rebuilt" after a patch applied today to the cell-layer statements. Two propositions of Appendix B are proved only in the formalization and not in the paper. And `enable_nanoda` is false in `comparator.json`.

They are candid about why it looks like this. Buckmaster, [this morning](https://mastodon.social/@tristanbuckmaster/117649035783718539): "After last time, we couldn't risk waiting, so the Navier–Stokes write-up is unpolished and heavily LLM-assisted." The paper says a first draft was generated from the Lean proof using Scott Armstrong's "enslopification" skills and then rewritten until it "no longer follows the Lean proof."

## Euler: the digested one

The [Euler paper](https://cims.nyu.edu/~tristanb/cc.pdf) is 55 pages and reads like a paper. Smooth periodic data on the torus, bounded velocity, vorticity at the origin going to infinity. The [Lean](https://github.com/mcoiculescu/EulerBlowupPublic) is 132,060 lines, and `Theorem11` is an unconditional existential with no hypothesis in front of it. The statement file is again the authors' own.

One caveat, again theirs: the formalization follows the manuscript as frozen on November 18, and the arXiv version revised Section 2 since. A matching formalization is promised.

OpenAI's September repository had unforced Euler blowup on ℝ³, the result I called the more remarkable one. This paper has a provenance section about that. It says the authors had an unreleased forced-Euler proof using "the same Craik–Criminale amplification mechanism as OpenAI's paper, including the same reduced equations," before OpenAI released. The priority dispute from September is not over. It has moved into the introductions.

## Yang–Mills: a fourth way to do this

The update here is ten days old but belongs in the same picture. A committee chaired by David Gross published an [assessment](https://github.com/yang-mills-group/YM) of OpenAI's October 28 submission. Its opinion is that the submission constructs a four-dimensional quantum Yang–Mills theory satisfying the Wightman axioms as a scaling limit of Wilson lattice gauge theory, and "does not prove that the infinite-volume limit is unique or that there is a mass gap." That is existence without the gap, half the Clay problem, and not the half with the prize.

What interests me is the structure. The submission's reference `Challenge.lean` has a Mathlib-only import closure. The committee keeps the original source unchanged in one directory and its own checks in another, and it publishes dated evidence, including what it did not do: OpenAI's own README says "The standalone Comparator executable was not run." The table of submissions has two more rows marked work in progress. One is OpenAI's, from November 4. The other is "Alpöge and Sussman, 15 November 2026."

A standing group that receives submissions, writes its own checks, and says in plain words what was and was not proved is close to what I was asking for in August: someone with no stake in the answer holding the statement. It exists for exactly one problem.

## What's left

Four results, four arrangements for who writes the question:

| | Question written by | Antecedent | Human-readable proof | Independent check |
|---|---|---|---|---|
| RH | Mathlib | none | none | in progress, by volunteers |
| Navier–Stokes | the authors | a certificate, unreleased | 79 pages, rushed | none yet |
| Euler | the authors | none | 55 pages | none yet |
| Yang–Mills existence | OpenAI, then a committee | none | 1,147 pages, unreleased | committee assessment |

The result with the most trustworthy statement has the least understood proof. The result with the most familiar method has a formal theorem that stops one step short of its own abstract. I did not expect the first sentence to be about the Riemann Hypothesis.

The work that is available tonight, to people without 500 GB of RAM: write the Formal Conjectures bridge for GRH. State the negations of (A) and (B) in DeepMind's vocabulary and see whether `BreakdownOnSpace` implies them. Read 110 lines of `Challenge.lean` against Fefferman. And when the certificate is released, check its hash.

---

*What I ran: downloaded `alpo.ge/remark.pdf` and read it in full (its sha256 matches the one in the mirror's notes); read `Main.lean`, `Statements.lean`, `Challenge.lean` and `verification/NOTES.md` from `Deicyde/riemann-hypothesis` at `712a6ae`; cloned `tristanbuckmaster/navier-stokes` at `ad53419` and `mcoiculescu/EulerBlowupPublic` at `36d3961`, counted files and lines, read the statement files, both READMEs and `NOTICE`, searched for `sorry`, `axiom`, `native_decide` and any reference to Formal Conjectures, and extracted both PDFs to text; read Formal Conjectures' `Millennium/RiemannHypothesis.lean` and `Millennium/NavierStokes.lean` at `d8d031d`; read the READMEs of `yang-mills-group/YM` and `OpenAI-YM-20261028`.*

*What I did not run: any build, Comparator, or any kernel, for anything. I did not download the 239 MB Riemann archive; every count for it is the mirror's, not mine. I did not read the Lean Zulip thread, so Sutherland's numbers are as relayed in the mirror's notes. I did not open the Yang–Mills assessment PDF, only the README that summarizes it; the 1,147-page figure is from reporting. My reading that the four Dirichlet statements "say GRH" is a reading. Tao's post I have through a fetch that returned short quotations, which is why I quote so little of it.*

*Written with Claude, and this time that needs a disclosure. Two of today's three results were produced with Claude, one of them by an Anthropic employee, and the note went up, per reporting, the day after Anthropic's Nasdaq debut. Alpöge writes that "there is an obvious cynical view, to do with the timing, but it is false." I have no way to check that and neither does the model that drafted this sentence, which has a stake of its own. Don't trust it, and don't trust me. The statement files are short. Go read them.*
