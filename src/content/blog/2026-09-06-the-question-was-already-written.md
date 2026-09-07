---
title: "The Question Was Already Written"
description: "Anthropic formalized Fermat's Last Theorem in Lean in eleven days. It is the frontier I said in May was untouched, and the first result in this series where nobody had to trust the statement. The problem that replaced it is that 13 million lines is more than anyone can read."
date: 2026-09-06
draft: false
tags: ["ai", "mathematics", "verification", "lean", "ricci-flow"]
---

# The Question Was Already Written

First, the thing every headline this week got wrong. Anthropic did not solve Fermat's Last Theorem. Andrew Wiles solved it in 1994, with Richard Taylor closing the gap, and it has been solved ever since. What happened on September 4 is that a machine wrote a proof of it in Lean, and the Lean kernel checked the proof, and now there exists a file you can compile that ends with a theorem saying `a ^ n + b ^ n ≠ c ^ n`.[^anthropic]

That is a smaller claim than the headlines made, and for the argument I've been running since May it is a much larger one.

In [Who Verifies the Verifier](https://korbonits.com/blog/2026-05-28-who-verifies-the-verifier/) I drew a line between two frontiers. On one side: generate proofs natively in Lean, against statements that somebody has already formalized, and let the compiler check them as you go. DeepMind was there in May, and it was working, and it was cheap. On the other side: take mathematics that humans actually wrote, at research depth, and autoformalize it. I said that second frontier was untouched, and I said why — you cannot point an agent at a theorem that has no formal statement to point it at.

That was three months ago, and it is no longer true.

## The receipt

I could not run this one, and I will come back to why. But the artifact is public, Apache-2.0, pushed to `anthropics/fermats-last-theorem` on September 4, and you can read it without building it.[^repo]

The statement, in `Theorems/Thm_fermat_last_theorem.lean`:

```lean
theorem fermat_last_theorem (n : ℕ) (hn : 3 ≤ n) (a b c : ℕ)
    (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) : a ^ n + b ^ n ≠ c ^ n
```

The default build target is a file called `FinalCheck.lean`, and here is the whole of the part that matters:

```lean
/-- info: 'fermat_last_theorem' depends on axioms: [propext, Classical.choice, Quot.sound] -/
#guard_msgs in
#print axioms fermat_last_theorem
```

I have spent four posts telling you to run `#print axioms` yourself. Here it is wired into the build with `#guard_msgs`, which means the build *fails* if the answer is anything other than those three axioms. You cannot compile this repository and get a proof that leans on a `sorry`, or an added `axiom`, or `native_decide`, because the compile is the check. A year ago an announcement like this came with a PDF and an invitation to take it seriously. This one does not compile unless the axiom list is what they say it is.

Then two more layers. `leanprover/comparator` at `v4.33.0` checked the build against a challenge file written using only Mathlib, confirming the proved statement and every constant it mentions are identical to the challenge — verdict, per the README, `Your solution is okay!`. And `nanoda`, an independent Lean kernel written in Rust, accepted an export of the environment: `Checked 1052234 declarations with no errors`. In August I noted that I had skipped the second-kernel re-check because it was slow. Anthropic ran it. They also patched nanoda four ways to make it finish, and they say none of the four adds, removes or weakens a typing rule. I have not read the patches. Of everything in this post it is the part I would worry about least, but it is worth knowing.

The size: 60,475 modules, 29,511 theorems, 1,450 definition modules. Anthropic puts the generated Lean at 13 million lines, which it describes as more than five times all of Mathlib, and says about 7% of the non-boilerplate lines were failed attempts. (I checked the denominator: `Mathlib/` at commit `fc98d420c2` is 870,767 lines of Lean, so on a lines-of-code basis 13 million is nearer fifteen times Mathlib than five. Whatever metric gives 5x, it isn't this one, and the direction of the error is not in Anthropic's favour.) Eleven days. Roughly six billion output tokens from an internal model they compare to Fable 5.1. Dozens of agents, coordinated through a platform called Prove2Me that holds the theorem dependencies as a directed acyclic graph. Anthropic says the fully autonomous approach did not work and that the DAG is what made it work, which seems like the load-bearing detail in the whole announcement.

## Nobody had to trust the statement

Five weeks ago, in [Who Writes the Question](https://korbonits.com/blog/2026-08-01-who-writes-the-question/), I built OpenAI's ten proofs, ran the axiom check on all 38 theorems, watched every one of them pass, and then said the trust had not gone anywhere — it had been squeezed into 1,700 lines of definitions that OpenAI wrote, for problems OpenAI answered, against a registry that still said OPEN. The compiler held. The vacancy was upstream of it. What was missing, I wrote, was "roughly a hundred lines of Lean per problem, written by somebody with no stake in the answer, before the answer exists."

The statement of Fermat's Last Theorem is the same six symbols Fermat wrote in a margin in 1637. It has a formalization in Mathlib, `FermatLastTheorem`, written by people with no stake in this, years before Anthropic existed as a company with a model. `FinalCheck.lean` derives Mathlib's version from Anthropic's in one line. Comparator confirms the constants are stock. The statement uses Lean's built-in naturals and `+`, `≤`, `≠`; its single Mathlib ingredient is `^` on ℕ, which is Lean's own exponentiation.

So this is the control case. Same scale of machine-generated proof as August, same three axioms, same kernel — and zero of the doubt I spent August on, because nobody in this story wrote the question. The question was written 380 years ago and pinned in a library 30,000 people use for other things.

That is a real answer to the thing I said was missing, and it is worth being precise about what supplied it. Not the model. A decade of unglamorous infrastructure: Mathlib, Kevin Buzzard's FLT project at Imperial, `flt-regular`. `ATTRIBUTION.md` in the repo is 50 KB long and lists 106 files containing material adapted from the first two. Eleven days of compute sat on top of roughly ten years of that work, and without it there would have been nothing to check the output against.

## The check left consumer hardware

Here is a sentence I wrote on August 1 that I have to withdraw:

> Today the credibility of ten results rests on a compile and a manifest hash. [...] checkable by a stranger on consumer hardware in under an hour, with no institutional standing required and nobody's weekend consumed.

That was true of 550,000 lines. At 13 million it isn't.

From the repository's own instructions: the build needs about 5 GB of memory per parallel job, a few modules need up to 36 GB, and Anthropic's own run took 5 hours 32 minutes at 96 jobs with a peak of 153 GB of memory and about 220 GB of transient C files on top of 67 GB under `.lake/`. Comparator takes roughly 15 hours, nearly all of it a single-core kernel replay, peaking at 230 GB — they advise allowing 300 GB. The nanoda export is 37.8 GB and wants about 90 GB of memory to write.

I have a laptop, so I can't check this, and neither can most of the people I would want checking it, including the nine mathematicians who spent a weekend on the unit-distance disproof in May.

I want to be careful here, because this is not the old problem wearing a new coat. The verification is still *mechanical*: it is a compile, not a judgment, and anyone with a 300 GB machine gets the same answer as anyone else with a 300 GB machine, which is not remotely true of nine experts reading a PDF. Renting that machine costs less than a plane ticket. But "no institutional standing required" was the part I liked best, and it is now "no institutional standing required, plus a quarter-terabyte of RAM for a day." The set of people who can independently confirm the largest formal result yet produced is small again. It is small for a better reason than it used to be, since this is a constraint that yields to money rather than to reputation, but it is small.

## What no tool can check

The repository says this about itself:

> What no tool can check is that each intermediate theorem means what its name suggests; that is for the reader to judge.

And, on the sources:

> The Lean sources were produced by AI agents building on human-written open-source Lean, with Lean as the arbiter, and are written to be checked rather than read: names are machine-generated, labels such as `P2M` or hexadecimal suffixes are pipeline labels rather than mathematics, and where a name and a statement disagree the statement is what was proved.

Read `PROOF-PATH.md` and you find out exactly what this costs. The file has a closing section titled "Exact strength of the named steps," and it is a list of things that are not what you think they are. `FreyPackage.Mazur_Frey` proves irreducibility of `E[p]` for Frey curves; it does not prove Mazur's theorems on rational isogenies or torsion of general curves. Langlands–Tunnell appears in the octahedral case only, for surjective ρ̄₃ with cyclotomic determinant. The two modularity lifting theorems hold under level conditions at p = 3 and p ∈ {3, 5}. Ribet's level lowering is proved for the Frey representation at squarefree conductor-supported levels, as a congruence of traces — not for a general modular mod-p representation. My favorite line in the whole document is a parenthetical in step 5 noting that the level-lowering proof re-derives modularity itself, "so the `IsModular` hypothesis above is formally unused." Somewhere in there, an agent proved Wiles's theorem twice because it was cheaper than looking it up.

So the top-level statement is exactly right, checked three ways, while the 29,511 theorems underneath it are named by a pipeline and have been read by essentially nobody. You get Fermat's Last Theorem about as securely as anything in mathematics, and you get very little else. You cannot cite `FreyPackage.Mazur_Frey` in your paper as Mazur's theorem, because it isn't. You cannot import this and build on it. Mathlib is 871,000 lines that prove millions of things, every one of them citable by name. This is 13 million lines that prove one thing.

The regress I have been chasing since May — proofs, then statements, then the definitions under the statements — did not move again; it ran out of places to go. There is no remaining spot in this artifact where a wrong result could hide. What is left instead is illegibility, which is a different problem and possibly a worse one, since doubt at least tells you where to look.

Buzzard wrote his own post about this the day it landed, titled, with more grace than I would have managed, [*FLT: Anthropic has beaten me to it*](https://xenaproject.wordpress.com/2026/09/04/flt-anthropic-has-beaten-me-to-it/). Two things in it stuck with me.

The first is what his review actually consisted of. In his words: "I manually inspected every line of the code base which (according to Claude) was not a mathematical definition or proof of a theorem, and verified that none of it was doing anything malicious." The human review that took place on the largest formal proof ever produced was a *security* audit — is anything in here attacking the build — and the partition into "mathematics" and "not mathematics" that made it tractable was supplied by the model being audited. None of which is a criticism of Buzzard, who did the sensible thing and then said plainly what he had done. It is just a fairly exact description of where this leaves us: nobody has read the mathematics, because reading it is not something a person can do.

The second is the consequence: **none of these 13 million lines can enter Mathlib as things stand.** Mathlib will not currently accept AI reviews, reviewers are reluctant to review machine-generated code — Buzzard notes, fairly, that most AI-generated PRs are poor quality — and the queue is already enormous: I counted 3,073 open pull requests against `leanprover-community/mathlib4` this morning, 2,538 of them not drafts. Human review capacity is, Buzzard says, the massive bottleneck, and that is before anyone proposes adding thirteen million lines to it. So the largest formal proof ever produced is, as a contribution to the shared mathematical library, worth zero lines. He also notes that it completes Freek Wiedijk's list of 100 theorems, a benchmark that has stood for twenty years.

## Meanwhile, by hand

I have spent the last month on the other side of this.

Since August I have been writing a [`leanblueprint` for Ricci flow](https://github.com/korbonits/ricci-flow-blueprint), aimed at Hamilton's 1982 theorem first and Perelman's spherical space form theorem — the Poincaré conjecture as the case Γ = 1 — as the terminal node. It is 7,304 lines of Lean, 278 declarations, 74 blueprint nodes. There is no `sorry` anywhere in it. The two things that aren't proved are marked `proof_wanted`, which elaborates the statement and type-checks it while admitting no proof exists: short-time existence for the flow, blocked on quasilinear parabolic systems that Mathlib does not have, and `hamilton_1982` itself, which is years away.

Eleven days versus a month of evenings is not a flattering comparison and I am not going to pretend otherwise. But look at what the month of evenings produced. When I started, Riemannian curvature did not exist in Mathlib — manifolds yes, connections yes, but no Riemann tensor, no Ricci, no scalar curvature, and zero occurrences of the string `sectional`. Some of that vocabulary arrived while I was working: Mathlib got the Levi-Civita connection itself in [#36845](https://github.com/leanprover-community/mathlib4/pull/36845), which is grunweg's work, not mine, and the blueprint builds on it. What I have sent upstream is smaller. Four merged PRs in `Topology/Connected`, about path-connectedness of products and pi types, which are the kind of thing you end up needing and then discover nobody wrote down. And one that is still open: [#42815](https://github.com/leanprover-community/mathlib4/pull/42815), twenty-nine lines proving that the Lie bracket acts as a derivation on functions, lifted out of the blueprint because it belongs in the library rather than in my repo.

It has been open for twenty-one days. Its labels are `t-analysis`, `new-contributor`, and `LLM-generated`, all three applied automatically by Mathlib's bot. The last one is accurate. Twenty-nine lines, honestly marked, sitting in the queue I described two sections ago. I am not complaining about this; the reviewers owe me nothing and the queue is 3,073 deep for reasons that have nothing to do with me. But it is the same bottleneck seen from the other end, and it suggests how the next few years actually go: not a dramatic policy fight over thirteen million lines, just everything moving at the speed people can read.

The machine is extraordinary at filling in a proof when someone else has written the question, and the writing of questions is still done by people, in public, before the answers exist. Anthropic got eleven days because Buzzard and Mathlib spent ten years writing the questions. Buzzard's grant still commits him to the same two things it did last week: pull requests to Mathlib adding fundamental objects from modern number theory, and a dynamic document letting humans explore the modern proof — the Khare–Taylor route, not the 1995 exposition Anthropic followed. He is right that somebody has to, and this week is the demonstration rather than the argument: you can now have a completely sound artifact that nobody is able to build on.

One correction to my own record while I'm here. I ended the August post saying I was going to go formalize Erdős #146 and #180 into `formal-conjectures`, because they were clean transcriptions I could write without having read anyone's answer. I didn't. Will Blair added 146, 180 and 183 on August 7, with links to `openai/ten-proofs`. So the queue does not wait for me, which is good, and I did not do the unglamorous work I had just spent a post telling you to value.

So the next milestone in the blueprint is the contracted second Bianchi identity — `tr_g(∂ₜ Ric) = ΔR`, the missing half of the scalar curvature evolution. It closes a node that has been open since I started. It is worth roughly one line of a press release, in about a year, if anyone writes one.

---

*What I ran: `gh api` against the repository to read `README.md`, `PROOF-PATH.md`, `FinalCheck.lean`, `formalization.yaml` and the file listing; `git log` and `rg` against my own blueprint for every number in the last section. Every quotation from the repository is from those files as of today.*

*What I did not run, in the interest of not overclaiming: the build, comparator, or nanoda. I don't have the hardware, which is a third of this post. So every claim here about the proof being correct is a claim about what Anthropic reports, checked for internal consistency and against a `#guard_msgs` I can read but not execute — not a claim I verified, which is a distinction I have insisted on for four months and am not going to quietly drop the first time it costs me something. I also have not read the four nanoda patches, and I have read essentially none of the 29,511 theorems, which is the post's own subject.*

*Written with Claude, which fetched the repository files, inspected my blueprint, and argued with me about the ending. Same disclosure as [last time](https://korbonits.com/blog/2026-08-01-who-writes-the-question/), same instruction: don't trust it, and don't trust me. The difference this month is that I can't tell you to go run it yourself. Somebody with 300 GB should.*

[^anthropic]: Anthropic, [*Formalizing Fermat's Last Theorem*](https://www.anthropic.com/research/formalizing-fermats-last-theorem) (September 4, 2026). The work is led by Tianyi Peng. Buzzard's quotations here are from his own post, [*FLT: Anthropic has beaten me to it*](https://xenaproject.wordpress.com/2026/09/04/flt-anthropic-has-beaten-me-to-it/) (September 4, 2026), except the endorsement in the second paragraph of this one, which is from Anthropic's.

[^repo]: [github.com/anthropics/fermats-last-theorem](https://github.com/anthropics/fermats-last-theorem), Apache-2.0, created 2026-09-04, 232 MB in git plus a 390 MB `html/` folder that renders all 29,511 theorems as browsable offline pages. The repository describes itself as a "Research artifact. Not maintained and not accepting contributions." Its `formalization.yaml` records `review: status: "self-assessed"`, `reviewers: []`, and `author_endorsement: "not-contacted"` against Wiles, Taylor–Wiles and Darmon–Diamond–Taylor.
