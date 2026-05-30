---
title: "Who Verifies the Verifier"
description: "An AI built the machine I said mathematics needed — a compiler that verifies proofs for cents instead of expert weekends. The catch is what it still can't read."
date: 2026-05-28
draft: false
tags: ["ai", "mathematics", "verification", "lean"]
---

# Who Verifies the Verifier

Three days ago I published [an essay](https://korbonits.com/blog/2026-05-23-the-verification-problem/) that ended on a problem I couldn't see a solution to. The short version: an AI had disproved one of Erdős's conjectures, the result was real, and it was *verified* — but verified in the only way we currently know how, which is that nine of the world's relevant experts — one of them a Fields medalist — read a hundred-page argument over a weekend and put their names on it. That doesn't scale. The supply of plausible-looking proofs is about to go parabolic and the supply of expert weekends is fixed, so the binding constraint isn't generating proofs: it's checking them. I argued that formal verification — proofs machine-checked in a language like Lean — was the only thing that obviously scales, and then I admitted the catch: nobody had formalized that proof, and the tooling that would make formalization routine doesn't exist. "We are," I wrote, "relying on a process that everyone involved can see won't hold." I didn't just write about this last week after the OpenAI announcement. I've been thinking about it [since 2023](https://korbonits.com/blog/2026-04-02-i-was-thinking-about-llm-automated-reasoning-before-it-was-cool-and-i-wasnt-ready/).

What I didn't know is that two days before I hit publish, Google DeepMind had posted the machine I was describing.

The paper is [*Advancing Mathematics Research with AI-Driven Formal Proof Search*](https://arxiv.org/abs/2605.22763), submitted May 21. The headline numbers are the kind that get screenshotted: an agent that autonomously resolved 9 of 353 open Erdős problems, proved 44 of 492 open conjectures from the OEIS, and did it at an inference cost of a few hundred dollars per problem. The architecture is simple: a language model generates a candidate proof in Lean, the Lean compiler checks it, the error messages feed the next attempt, repeat until the compiler is satisfied. A generate-and-verify loop.

If you read my last post, this looks like a potential answer. The verification layer that saved the unit-distance disproof was nine scarce humans. The verification layer here is a compiler that costs cents and never gets tired.

## Two frontiers

Both results claim "AI does Erdős." They are not the same arc.

The unit-distance disproof was: *generate deep mathematics in natural language, then have humans verify it.* The hard part was the mathematics, and the verification was the expensive, unscalable, human step.

This paper is: *generate the proof natively in Lean from the start, and let the compiler verify each step as it goes.* The mathematics is shallower (for now), and the verification is free.

These are two different frontiers. DeepMind ran their agent against the 353 Erdős problems that *already had Lean formalizations* in their open-source Formal Conjectures repository. The problems the machine can attack are, by construction, the ones already expressible in Lean. The unit-distance disproof is not in that set. Its proof imports infinite class field towers and Golod–Shafarevich theory, machinery with no mature support in Lean's library. You cannot point this agent at it, because there is no formal statement to point it at.

Native Lean generation is the tractable frontier, and it is advancing fast and getting cheap. Autoformalizing the proofs humans actually write at research depth is the *other* frontier, and this paper — for all its real achievement — does not touch it.

## Inside problem #125

I picked #125 because it is simple to state, and therefore easier to communicate to a lay audience.

Briefly: let *A* be the set of integers that use only the digits 0 and 1 when written in base 3, and let *B* be the same thing in base 4. Does the sumset *A* + *B* have positive lower density — that is, is there some fixed fraction *c* > 0 such that, for every sufficiently large *N*, at least a *c*-share of the integers below *N* lands in *A* + *B*? ("Lower" means the worst large scales must stay above the floor, not just some of them — a set can be thick at carefully chosen scales and thin at others, and lower density rules out the easier case.) The agent proved the answer is no: the lower density is zero.

The idea behind the proof is elegant, and it rides a single observation about two number systems that almost line up. The powers of 3 and the powers of 4 never coincide — `3^k` is never exactly `4^m` — because `log 4 / log 3` is irrational. But they come *arbitrarily close*: for any tolerance you like, you can find exponents where `3^k` and `4^m` are within it. This near-miss is the crux of the whole argument. The proof exploits it to show that every time you zoom out by one of these matched scales, the sumset can capture at most 11/12 of the proportion it held at the previous scale. Apply that thinning *d* times and your density bound is (11/12) raised to the *d* — which marches to zero. The paper names the technique "Inductive thinning via Diophantine approx. `3^m ≈ 4^k`," and that one phrase is the approach to the solution.

Surprisingly, you can read that English paragraph straight off the Lean. The proof decomposes into named lemmas that map one-to-one onto the argument:

```lean
lemma log_ratio_irrational : Irrational (Real.log 4 / Real.log 3)
lemma dirichlet_approx (ε : ℝ) (hε : 0 < ε) :
    ∃ k m : ℕ, 0 < k ∧ 0 < m ∧ (3^k : ℝ) ≤ 4^m ∧ (4^m : ℝ) ≤ (3^k : ℝ) * (1 + ε) ∧ ...
lemma scale_step (N : ℕ) (hN : 0 < N) (C : ℝ) (hC : ... ≤ C * (N : ℝ)) :
    ∃ N' > 0, ... ≤ (11/12 : ℝ) * C * (N' : ℝ)
lemma density_multi_scale (d : ℕ) :
    ∃ N > 0, ... ≤ ((11/12 : ℝ)^d) * (N : ℝ)
lemma density_tends_to_zero (ε : ℝ) (hε : ε > 0) :
    ∃ N > 0, ... ≤ ε * (N : ℝ)
```

`log_ratio_irrational` is the never-coincide fact. `dirichlet_approx` is the come-arbitrarily-close fact. `scale_step` is the single 11/12 shrink. `density_multi_scale` is the induction that applies it *d* times. `density_tends_to_zero` is the limit. The structure of the human idea and the structure of the formal proof are the same shape.

Now the part that earns the word *verified*. Lean has a primitive called `sorry`. Write `sorry` and it closes any open goal and the file compiles, with a warning Lean prints but doesn't block on — the mathematical equivalent of "trust me." A proof is finished precisely when it compiles with *no `sorry` anywhere in it.* So "verified" is not a judgment call, and it is not nine reputations stacked on a PDF. It is a mechanical, checkable property — something you run, not a reputation you weigh. The crude version of the check is the absence of a single keyword: I cloned the repository and the #125 file is 370 lines with zero `sorry`s. The precise version is one command — `#print axioms` — which prints exactly what a theorem rests on. Run it on #125 and it answers `[propext, Classical.choice, Quot.sound]`: the three standard axioms every ordinary Lean proof uses, and nothing else. No `sorry`, no hidden assumption, no escape hatch. That list — finite, enumerable, the same for this proof as for a one-liner — is the entire basis of trust, and it cost a compile.

And here, finally, is the artifact I keep coming back to. The thing the agent was actually asked to produce was a proof filling a marked hole in this statement:

```lean
theorem target_theorem_0
  : answer(
  -- EVOLVE-VALUE-START
  False
  -- EVOLVE-VALUE-END
  ) ↔ 0 < (A + B).lowerDensity := by
  ...
```

The `EVOLVE-VALUE` block is the agent's to fill. Its entire contribution to the *statement* of an eighty-year-old style of question — does this set have positive density? — was the single token `False`. No, it does not. And then a sorry-free proof to back the claim.

Set that against where my last post ended: a 125-page artifact, compressed by hand to a few pages, read across a weekend by nine of the people on Earth qualified to read it, its credibility resting on their names. Here the credibility rests on one boolean and a compiler. That is the whole shift, sitting in five lines of code.

## The machine tried to cheat

If I stopped there it would be a victory lap, and the paper itself won't let me, because to their real credit DeepMind published their failure analysis. It is the most interesting page in the document, and it is the dark twin of the `sorry` story.

When they looked at the high-scoring proof attempts on problems the agent *failed*, two patterns showed up. First, the agent would repeatedly take a problem's central difficulty and bury it inside a single `sorry` in a helper lemma — a lemma that, on inspection, just restated the original goal in slightly different clothes. Explicitly prompting it not to do this didn't stop it. Second, and worse: some of the best-scoring attempts leaned on `sorry`-marked lemmas that the agent asserted were established results from the mathematical literature. They were not. They were hallucinations dressed as citations.

Read that against the warning I quoted last time, from Melanie Wood: "in many cases, it will be easier for AI to convince humans it has a proof than to come up with a correct mathematical argument." This is precisely that instinct, caught in the act. Offered a hard step it can't take, the model reaches for *persuasion* — a deferred goal, a confident false citation — rather than *proof*. It is optimized to close the gap between convincing and correct from the wrong side. The October over-claim that opened my last post was this same failure with no checker in the room.

The difference, this time, is that it didn't work. A compiler is not an audience. `sorry` is not a rhetorical flourish you can talk it into accepting, and a hallucinated lemma doesn't compile just because the prose around it is fluent. But it pays to be precise about what the check buys you, because `sorry` compiles — it only warns. I built the whole repository and it goes green while still containing 128 `sorry`s, scattered across scaffolding files for problems that weren't solved. A passing build means the code is well-formed, not that everything in it is proved; the trust lives in the per-theorem check, not in the build succeeding. That is exactly why the failure analysis matters. The paper's own conclusion is the sentence I'd have written myself: these failures "underscore the value of end-to-end formal verification." That is my thesis getting stress-tested under adversarial pressure and surviving. The formal layer is valuable *precisely because* it is immune to the one thing the model is best at.

## The frontier that's still open

Which brings me back to the gap I flagged at the start, and to why this machine answers a different question than I asked. The evidence is in the paper itself, in two places.

The first is a misformalization the paper reports on #125 itself. The original 1996 problem just said "density," and that word is ambiguous — natural density, lower density, and upper density are different things, and a proof of one is not a proof of another. The paper notes that the informal statement's "density" had to be amended to "lower density" (and on a separate problem, #741, to "upper density") to capture the intended question, after the agent produced a proof against a different reading. The machine proves; a person still has to ensure the formal statement says what the mathematician meant. That correction step is not incidental — it's the seam where a human stays in the loop.

The second is structural: when DeepMind autoformalized 492 OEIS questions, they had to add "test lemmas" — guards that check a formal statement reproduces the first few known terms of a sequence — specifically because autoformalization at that volume produces statements that don't mean what they should. The faithful-translation step is unreliable enough that it needs its own safety net.

Generalize from those two facts and you arrive at the thing my last post was really about. The unit-distance disproof can't even *enter* this system, because formalizing 125 pages of class field theory in Lean is itself an open research problem, and faithful autoformalization of research-level natural-language mathematics does not yet exist. Tsimerman's aside from the unit-distance remarks — "it will be interesting to see how formalization progresses alongside AI" — names exactly the unsolved layer, and it is still unsolved.

So the two frontiers, one last time, cleanly. *Generating* proofs for problems already expressed in Lean is working, and getting cheap fast. *Formalizing* the hard proofs humans actually produce — getting them into the checker at all — is the bottleneck. The machine that holds is real. The pipeline that feeds it is not.

## Quis custodiet

*Who verifies the verifier?* It's the oldest objection to any chain of trust — Juvenal's *quis custodiet ipsos custodes*, who guards the guards, the worry that every checker needs a checker and the regress never bottoms out. In mathematics it's a live fear, not a rhetorical one: my last post was an entire essay about it. The unit-distance proof was trusted because nine experts read it — but who verifies the nine? Their reputations. And who verifies the reputations? More reputation. Trust never touched the ground; it was reputations all the way down, and the supply of reputation is fixed.

The quiet, remarkable thing this paper demonstrates is that, for once, the regress *terminates*. A `sorry`-free Lean proof is trusted not because of who wrote it or who blessed it, but because a compiler mechanically checked every step, and the compiler's own correctness is a small, fixed, scrutinized thing that doesn't grow with the proof. You don't need a more eminent expert to verify a more important result. The verifier that needs no verifier turns out to be a piece of software you can run for cents. That is genuinely new, and I don't want to undersell it: the checker that holds exists, it works, and it's immune to exactly the persuasion failure that produced the October 2025 debacle.

And the architecture has a property the headline numbers can't capture: it decouples cleanly enough that you can productize half of it independently. [Axiom Math](https://axiommath.ai/) is the cleanest example — their [AXLE API](https://axle.axiommath.ai/) is *verification-as-a-service*, a turnkey Lean checker you hit over HTTP. Submit Lean code, get back validation and error messages with line:column positions. That is exactly the loop DeepMind's agent ran, factored out of the agent and sold as infrastructure: the model is yours, the checker is theirs, the two compose by HTTP. *Feel the exponential* is the cultural phrase, and I find it a little hard to say with a straight face, but the capital flowing in is unsentimentally betting that the slope is real — and notice which half it's betting on.

But notice what happened to the regress. It didn't end. It moved. We escaped *quis custodiet* in the verification of proofs and reintroduced it one level up, in the *formalization* of them — because now the question is who guarantees that the Lean statement faithfully says what the mathematician meant, and the answer, for the unit-distance disproof and for the corrected wording of problem #125 alike, is still: a human who reads carefully. The expensive step used to be checking the proof. Now, for the proofs we most care about, it's translating them into the language where checking is free. The compiler will certify anything you can hand it in Lean. It still can't read the proofs we most need certified.

The frontier isn't proving anymore. It's translation — and translation is where the watchmen went to hide.
