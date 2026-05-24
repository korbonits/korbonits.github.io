---
title: "The Verification Problem"
description: "An AI disproved one of Erdős's favorite conjectures. The interesting part isn't the proof — it's who read it, and what happens when nobody can."
date: 2026-05-23
draft: false
tags: ["ai", "mathematics", "verification"]
---

In October, OpenAI said its model had solved ten open Erdős problems. It hadn't.[^openai-october] The model had surfaced existing solutions buried in the literature and presented them as fresh conquests, and Thomas Bloom — who runs [erdosproblems.com](https://erdosproblems.com) and would know — said so publicly and without much patience for the framing.[^bloom-pushback]

Last week OpenAI announced that an internal model had disproved Erdős's planar unit-distance conjecture.[^openai-may] This time the announcement came with a companion paper written by nine mathematicians, and one of the nine was Thomas Bloom.[^remarks]

From the outside, the two announcements are indistinguishable: *AI solves famous math problem*, same company, same Erdős. What separates them is not that the model got smarter between October and May. What separates them is that the second result was verified and the first was not. This post is an attempt to say what verified means, why it was expensive, and what happens to mathematics when the thing producing proofs gets cheaper than the thing checking them. We need to automate the verification.

## What the model actually did

The unit-distance problem, posed by Erdős in 1946, asks how many pairs among $n$ points in the plane can be exactly distance one apart. A $\sqrt{n} \times \sqrt{n}$ grid, read as Gaussian integers, gives about $n^{1 + c/\log\log n}$ pairs — barely faster than linear. For eighty years the belief, Erdős's own, was that you couldn't do meaningfully better: the conjectured ceiling was $n^{1 + o(1)}$. Bloom had it on a "Top 10 Erdős Problems" list one month before the result landed, written partly to push back on the idea that unsolved Erdős problems were trivia nobody had bothered to attempt. This was not trivia. Many working combinatorial geometers had thought about this problem.

The model disproved it. It produced an infinite family of configurations achieving $n^{1+\delta}$ for a fixed $\delta > 0$ — a polynomial improvement, a different order of magnitude than anyone expected — later pinned by Will Sawin at $\delta = 0.014$.[^remarks] The proof does not come from better geometry. It imports algebraic number theory: infinite class field towers, Golod–Shafarevich theory, the machinery of factorization in field extensions, turned onto an elementary question about dots on a page.

Here is the part the headlines flattened. The construction is, in Bloom's own assessment, a natural generalization of Erdős's original grid — natural with eighty years of hindsight, and genuinely hard to find without it. Erdős built his grid from the Gaussian integers, a single fixed field. The model's move was to replace the fixed field with a *tower* of fields of growing degree. Sawin and Jacob Tsimerman, both in the companion paper, explain why that inversion is so unintuitive: the natural instinct is to take a sequence of larger and larger point sets inside one field, and if you do the calculation that way, the field you choose doesn't seem to matter — there's no signal telling you to vary it. You have to vary the wrong-seeming knob. Tsimerman actually tried this problem, with exactly the increasing-degree idea, and didn't get there; he calls increasing degree "a very scary dynamic that often doesn't work out."[^remarks]

So: a real disproof of a real conjecture, using deep tools, that several of the experts who checked it believe they *could* have found and did not.

## The texture of the reasoning

OpenAI released a rewritten version of the model's chain of thought, and it's the most interesting document in the pile, because it's the one thing that's hard to fake and hard to retrieve.[^cot]

The model takes the best known construction and starts trying to generalize it, and then spends pages failing in specific, recognizable ways. It tries the Boolean hypercube — subset sums of unit vectors — and notes it gives only $n \log n$, hypercube-scale. It tries powers of the rational point (3+4i)/5 on the unit circle, works through clearing denominators, and lands back at "exactly the divisor-function picture, only linear." It tries higher-degree algebraic numbers, then Dirichlet units, and kills the units idea with a clean observation about CM fields: the relative unit rank is zero, so norm-one relative units are essentially just roots of unity. Dead end after dead end, each rejected for a stated mathematical reason, not abandoned at random.

And then, on the page Arul Shankar flagged by name in his remarks,[^remarks] it turns. The paragraph opens "Suppose optimistically that K is a high-degree CM field in which a fixed rational prime splits completely into principal prime ideals" — and the model immediately sees the prize, calls the resulting construction "frightening," sketches the point count, and in the same breath stress-tests it against the known $n^{4/3}$ incidence barrier: if the prime were 2 or 5 the construction would *exceed* that barrier, so such a family can't exist in the naive form, but for larger fixed primes the exponent sits below $1/3$ while still being a fixed power — which already refutes Erdős. Then it names the real obstruction in one sentence: do these fields exist? Everything after is the answer to that question.

Whatever else is true, that is not a database lookup. Shankar's read is that the model has "some combination of good intuition, willingness to try approaches considered long-shot by the community, and a predisposition to attempt constructions."[^remarks] A striking majority of the chain of thought is spent trying to *break* the conjecture rather than prove it — which is precisely the move human consensus discouraged.

## Why verification was the expensive part

The proof was produced in one shot. According to the paper's own statement on AI use, the model's output was first sent to an *AI grading pipeline*, which reported high confidence the solution was correct. Only after that did human researchers at OpenAI start reading. After internal cleanup, a draft went to external mathematicians — number theorists among them — who confirmed correctness and, in the process, simplified and strengthened the argument. The published remarks are a human-digested version, compressed from a 125-page artifact down to a few pages of actual mathematics plus a dossier of reflections.

Count the verification layers: a machine grader, then OpenAI's internal humans, then nine external experts. The result's credibility does not rest on the model's confidence, or even on the AI grader's confidence. It rests on Bloom, Alon, Gowers, Litt, Sawin, Shankar, Tsimerman, Wang, and Wood having read it and put their names on it. That is what "verified" meant here. It meant a significant fraction of the people on Earth qualified to check this particular argument spent a weekend checking it.

The digested argument, Tsimerman notes, was short enough that the participants were willing to invest the time — and "had the final digested argument been a bit longer, the process may have been trickier, as participants may have been less willing to invest time in checking details."[^remarks] I.e., the verification happened because the proof compressed to a few pages *and* because the problem was famous enough to summon nine experts. Both conditions are contingent. Neither scales.

This is the asymmetry that I think matters most, and it's the one I think about continuously. When you build serving infrastructure, the thing you learn in your bones is that the expensive step is rarely the one everyone's watching. Here the proof is the cheap step — cheap enough that the same lab over-claimed seven months ago by generating something that *looked* like proofs. The trust is the expensive step, and right now trust is earned by routing the artifact through scarce human attention. That's a batch process with a human in the critical path, and the human doesn't get faster.

## The part the field isn't ready for

Melanie Wood, in her remarks, makes two points that together form the uncomfortable core of all this.

The first: this result is survivorship bias made into a press release. It "does not show us all the times AI has claimed to have a proof of something and been wrong." Most people don't have the October context — or the dozens of quieter wrong claims working mathematicians have already fielded in private — and without it they'll draw the wrong conclusion about where the technology is. The second, and sharper: "in many cases, it will be easier for AI to convince humans it has a proof than to come up with a correct mathematical argument, and I believe that we as mathematicians are not sufficiently prepared for this."[^remarks]

That sentence is the dark twin of the verification story. If trust is produced by expert reading, and the supply of expert reading is fixed, and the supply of plausible-looking proofs is about to go vertical, then the binding constraint isn't proof generation. It's the verification bandwidth — and worse, the gap between *convincing* and *correct* is exactly the gap a generative system is optimized to close from the wrong side. A model that's better at persuasion than proof is not a hypothetical; it's the failure mode that produced October.

This is the point where formal verification stops being a nice-to-have and becomes the only thing that obviously scales. A proof checked in Lean doesn't care whether nine famous people had a free weekend. The verification cost goes to roughly zero once the formalization exists, and the trust no longer routes through scarce human attention. If you believe Wood's forecast — and the October data point is evidence for it — then machine-checkable proof isn't an academic curiosity. It's the infrastructure the next hundred of these results will require in order to mean anything. I sketched what that infrastructure would look like three weeks earlier in [Proofs and Essays Are Paths](/blog/2026-04-30-proofs-and-essays-are-paths): LLM outputs as extensions to a knowledge graph, with a prover holding the model accountable.

I'd love to end the essay there, on a tidy thesis. But the honest version has a catch, and Tsimerman names it in passing: "it will be interesting to see how formalization progresses alongside AI."[^remarks] Nobody has formalized this proof. Formalizing 125 pages of class field theory in Lean is itself a major undertaking, and the autoformalization tooling that would make it routine does not exist yet. So the bottleneck I'm describing is real *and* currently unsolved. The verification layer that saved this result was nine humans, and the verification layer that would let us trust results at volume hasn't been built. We are, at the moment, relying on a process that everyone involved can see won't hold.

## What it was really for

Wood points out that if you had assembled these same nine experts a month ago and asked them to find a counterexample — and they'd put in the hours they later spent reading the model's — they probably would have found it.[^remarks] The mathematics was within reach of the people who eventually checked it. What was missing wasn't capability. It was that nobody was looking, because everybody believed Erdős, and there was no reason to convene the experts or point them at this problem rather than the thousand others.

So maybe the model's real contribution wasn't an idea no human could have had. Several of the humans think they could have had it. The contribution was solving an *attention-allocation* problem: it told the world's relevant experts, with enough specificity and credibility to be worth acting on, exactly where to point their scarce time. Daniel Litt, in his remarks, worries this implies our incentives toward specialization have quietly cost us good science — that there's low-hanging fruit going unpicked because the person who could pick it is in the wrong silo and never thought to look.[^remarks]

That's a humbler story than "AI does mathematics," and a more interesting one. It suggests the near-term shape of the thing isn't a machine that replaces the mathematician but one that reallocates mathematical attention — and that the binding constraint on the whole enterprise, the one nobody has solved, is verification. We just watched the most impressive case go right because nine people read carefully. The open question is what we do the next thousand times, when there aren't nine people, or nine weekends, to spare.

[^openai-october]: OpenAI's October announcement has since been taken down; see [*OpenAI claims AI breakthrough but gets math facts wrong instead*](https://www.technology.org/2025/10/20/openai-claims-ai-breakthrough-but-gets-math-facts-wrong-instead/) (Technology.org, October 20, 2025) for contemporaneous coverage.

[^openai-may]: OpenAI, [*Model disproves discrete-geometry conjecture*](https://openai.com/index/model-disproves-discrete-geometry-conjecture/).

[^remarks]: Noga Alon, Thomas F. Bloom, W. T. Gowers, Daniel Litt, Will Sawin, Arul Shankar, Jacob Tsimerman, Victor Wang, and Melanie Matchett Wood, [*Remarks on the Disproof of the Unit Distance Conjecture*](https://cdn.openai.com/pdf/74c24085-19b0-4534-9c90-465b8e29ad73/unit-distance-remarks.pdf) (May 19, 2026).

[^cot]: [*Rewritten Chain of Thought for the Solution to the Unit Distance Problem*](https://cdn.openai.com/pdf/1625eff6-5ac1-40d8-b1db-5d5cf925de8b/unit-distance-cot.pdf) (OpenAI, May 20, 2026).

[^bloom-pushback]: Thomas Bloom, [responding on X](https://x.com/thomasfbloom/status/1979254235075059732).
