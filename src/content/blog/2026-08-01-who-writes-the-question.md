---
title: "Who Writes the Question"
description: "OpenAI shipped ten open problems with Lean certificates. I built all 550,000 lines and checked what they rest on. Everything passed — and the only thing left to trust is 1,700 lines of definitions written by the same system that wrote the proofs."
date: 2026-08-01
draft: false
tags: ["ai", "mathematics", "verification", "lean"]
---

# Who Writes the Question

This morning OpenAI published ten results on problems that have been open for at least a decade and mostly much longer: high-dimensional sphere packing, binary and spherical codes, non-sofic groups, Connes's rigidity conjecture, arithmetic circuit complexity, quantum parallel repetition, the closest vector problem, Ehrhart's volume conjecture, and three Erdős problems.[^openai-aug] They were found by an internal version of a model called Astra. The tokens cost about $2,000.

Count the names of mathematicians on it. There are none.

In May, when a model disproved Erdős's unit-distance conjecture, the announcement came with a companion paper by nine of them — Alon, Bloom, Gowers, Litt, Sawin, Shankar, Tsimerman, Wang, Wood — who read the argument over a weekend and put their reputations on it. [I wrote at the time](https://korbonits.com/blog/2026-05-23-the-verification-problem/) that this was what "verified" meant, that it was expensive, and that it would not scale. Today's release is what the same lab does ten weeks later, and the answer is that it doesn't try. The 249-page paper contains no methodology section, no acknowledgments, and exactly one mention of AI, in the abstract: the results were "obtained by an internal OpenAI model." The word "Lean" does not appear in it.

The epistemics were moved somewhere else. They were moved into a GitHub repository.

So I did the thing [I said in July](https://korbonits.com/blog/2026-07-20-trust-nothing-verify-everything/) was now the only sane response to a mathematical claim: I ignored the announcement and went and ran the certificate.

## The receipt

```sh
git clone https://github.com/openai/ten-proofs
cd ten-proofs
lake exe cache get
lake build All
```

Thirty-seven minutes and twenty-three seconds on a laptop. 8,666 jobs. Two warnings, both cosmetic — a `Try this:` suggestion and a deprecated lemma name. No errors.

That builds 549,982 lines of Lean 4 across ten modules. `GapCVP.lean` alone is 130,430 lines; `MetricCodes.lean` is 114,408. The smallest, `MulticolorTriangleRamsey.lean`, is 3,053.

Then the part that actually matters. Every comparator configuration in the repo names its headline theorems — 38 of them across the ten results — so I generated a file that imports everything and asks Lean what each one rests on:

```lean
import All
#print axioms PackingBounds.PackingBridge.sphere_packing_sharp_asymptotic_upper
#print axioms ErdosProblems.MulticolourTriangleRamsey.erdos_183
-- ...36 more
```

All 38 came back identical:

```
depends on axioms: [propext, Classical.choice, Quot.sound]
```

Zero `sorryAx`. Zero non-standard axioms. Zero errors. That is the same three-element list that a one-line Lean proof rests on, and it is the same list for the 130,000-line one. This is the property I keep insisting is the whole ballgame: *the trust base does not grow with the difficulty of the theorem*. Nine experts do not scale. Three axioms do.

There are 42 `sorry`s in the repository, and I want to be precise about them because a casual grep would mislead you. Every one lives in `ComparatorChallenges/`, and they are not gaps in the proofs. They are the *statements*, pinned in separate files with their proofs deliberately omitted. More on why in a moment.

One more check, which took a single command and turned out to matter more than anything else I ran. Every challenge file begins with `import Mathlib`. If the pinned mathlib were a fork, then `SimpleGraph.extremalNumber` and `Matrix.permanent` and `VonNeumannAlgebra` could mean anything at all, and the entire exercise would be reading a dictionary somebody else wrote. The manifest pins `81a5d257c8e410db227a6665ed08f64fea08e997`. The upstream `leanprover-community/mathlib4` tag `v4.32.0` is `81a5d257c8e410db227a6665ed08f64fea08e997`. Genuine.

## What a certificate is, exactly

The repo uses [comparator](https://github.com/leanprover/comparator), a tool from the Lean developers that I hadn't seen before and that deserves to be better known, because it is a precise engineering answer to the question this whole series has been circling.

The idea: split the artifact in two. A *challenge* file states a theorem and proves it with `sorry`. A *solution* file, written by whoever is trying to convince you, proves it for real. Comparator then checks three things — that the solution's theorem has the same type as the challenge's, that it uses no axioms outside an explicit allowlist, and that the Lean kernel accepts it. Every configuration in `ten-proofs` allows exactly `propext`, `Quot.sound`, `Classical.choice` and nothing else.

The README is the most interesting document in the pile, because it does something papers almost never do: it enumerates its own trust assumptions as a numbered list. You must control the challenge file's imports. You must not have previously compiled the solution. The sandbox must hold. The kernel must be correct — or, with a second kernel enabled, *one of the two* must be. That list is finite, it is written down, and it does not get longer when the theorem gets harder. Compare the alternative, which is "nine famous people had a free weekend and thought it looked right."

I ran the half of this that a MacBook can run. Comparator's sandbox is `landrun`, which is Linux Landlock, so I couldn't use it — and I had already compiled the solutions before checking, which violates assumption 2 outright. All twelve configurations also set `enable_nanoda: true`, asking for a re-check by an independent Rust implementation of the Lean kernel, and I didn't do that either. So what I ran is a check against error, not against a hostile OpenAI, and it trusts one kernel where the shipped configuration asks for two.

But the statement-matching half I could do, by elaborating each challenge module and the solution module in separate Lean processes and diffing the resulting types. **38 of 38 identical.** I checked the diff wasn't lying by perturbing one statement — changing `n.factorial` to `(n.factorial + 1)` in the Ehrhart bound — and it flagged the mismatch immediately.

So the proofs prove the pinned statements. Which raises the only question left.

## Who pinned the statements

All ten challenge files together are about 1,700 lines. They import nothing but Mathlib, so every definition the ten theorems depend on is right there, readable in an afternoon. I read them.

Mostly what you find is reassuring, and in a specific way: the load-bearing objects are frequently *mathlib's*, not the model's. The permanent lower bound is stated about `Matrix.permanent` of `Matrix.mvPolynomialX`. Both extremal graph theory results use `SimpleGraph.extremalNumber`. The GapCVP hardness result builds NP on `Turing.TM2ComputableInPolyTime`. Connes rigidity uses mathlib's `VonNeumannAlgebra` and `IsStarProjection`. Every one of those is a definition written by strangers, years ago, for other reasons, and scrutinized by a community — which makes it very hard to bend toward a desired conclusion.

The Ehrhart file is 78 lines and I'd hand it to a graduate student as an example of how to state a conjecture: convex, compact, nonempty interior, centroid at the origin, the origin its only interior lattice point, therefore volume at most $(n+1)^n/n!$. There is nothing in it to argue with.

But some of it is bespoke, and the bespoke parts are exactly where you'd have to argue. Property (T) is defined in-file, quantified over Hilbert spaces in a single universe. Normality of a $*$-isomorphism is rendered as preservation of suprema of projections rather than $\sigma$-weak continuity — standard-equivalent, but a choice. The quantum parallel repetition result models tensor-product strategies over finite-dimensional systems, which is the conventional entangled value and is *not* the commuting-operator value; the informal phrase "general two-player quantum games" does not tell you which. And `kissingNumber`, carrying a bound of $2^{0.39661n}$, is defined in the challenge file. Mathlib has no kissing number.

None of that is wrong. All of it is somebody's judgment, and the somebody is the system that also produced the proofs.

And one result isn't pinned at all. Result 5 is advertised as lower bounds "using arithmetic circuits **and** formulas." The shipped comparator challenges cover the formula bounds only. `lakefile.toml` declares a thirteenth challenge root, `ComparatorChallenges.C_PermanentSuperquadraticStandalone`, and that file is not in the repository — so `lake build ComparatorChallenges` fails outright on a fresh clone, and the superquadratic circuit bound has no pinned statement to check it against. It may well be an oversight. I can't ask: issues are disabled on the repository.

## Three problems with an owner

The three Erdős results are the place to test this, because Erdős problems have a curator. Thomas Bloom maintains [erdosproblems.com](https://www.erdosproblems.com), and his statements are as close to canonical as this subject gets.[^bloom146][^bloom180][^bloom183]

**#146.** Bloom: *If $H$ is bipartite and is $r$-degenerate, that is, every induced subgraph of $H$ has minimum degree $\leq r$, then $\mathrm{ex}(n;H) \ll n^{2-1/r}$.* OpenAI's `IsDegenerate r G` says every nonempty finite vertex subset contains a vertex with at most $r$ neighbours inside it — which is precisely "every induced subgraph has minimum degree $\leq r$." The conclusion is `IsBigO`. It is a transcription. The one addition, `0 < r`, is forced by Lean's convention that $1/0 = 0$, and it cuts the safe way: excluding $r = 0$ makes the conjecture weaker and refuting it harder. Their counterexample is at $r = 2$, which Bloom's page notes was open by itself.

**#180.** Bloom: *Is it true that, for every $\mathcal{F}$, there exists $G\in\mathcal{F}$ such that $\mathrm{ex}(n;G)\ll_{\mathcal{F}}\mathrm{ex}(n;\mathcal{F})$?* Here there is a real deviation. OpenAI adds a hypothesis that every graph in the family contains a cycle. Bloom says "for every $\mathcal{F}$," full stop. The direction is conservative — their proposition is implied by his, so refuting theirs refutes his — and their witness family is additionally connected and bipartite, which is more restrictive still. It's the correct direction to deviate in. It is still a deviation, and you only find it by reading both.

**#183.** Bloom: *Let $R(3;k)$ be the minimal $n$ such that if the edges of $K_n$ are coloured with $k$ colours then there must exist a monochromatic triangle. Determine $\lim_{k\to\infty}R(3;k)^{1/k}$.*

Look at the verb. **Determine.** That is not a proposition. It has no truth value, so there is nothing to transcribe, and any formalization must *supply a candidate answer and then state it*. OpenAI's `erdos_183` supplies $+\infty$, backed by a superexponential lower bound: $R(3;k) \geq \left(\frac{k^{1/3}}{6e^{38}\log k}\right)^{k}$ for $k \geq 2$. Erdős had offered $100 for a proof that the limit is finite. It isn't.

I've written about this shape before without noticing how far up it goes. In [Who Verifies the Verifier](https://korbonits.com/blog/2026-05-28-who-verifies-the-verifier/), DeepMind's agent faced problem #125 through an `EVOLVE-VALUE` hole, and its entire contribution to the *statement* was the single token `False`. I called that five lines of code containing the whole shift. It was — but I read it as a curiosity of one benchmark. It isn't. It is the generic situation for every problem phrased as an imperative, which is most of the interesting ones, and it means that for #183 there is no such thing as an independent transcription. Somebody has to agree that "+∞" is the right *shape* of answer to "determine the limit." That somebody is a human being reading Erdős, and today it was an employee of the company whose model produced the answer.

## The registry still says open

As I write this, all three problems are listed **OPEN** on erdosproblems.com. #146 carries a $500 prize. #183 carries $250, plus Erdős's $100 side bet on finiteness that just got settled the wrong way. The pages were last edited in January and April. Each one has a field reading "Formalised statement?" with a link inviting you to create one in [formal-conjectures](https://github.com/google-deepmind/formal-conjectures), DeepMind's open repository of machine-readable conjecture statements. That repository currently has 510 Erdős problems formalized. It does not have 183, 146, or 180.

In July I made fun of Wikipedia for still calling the Jacobian conjecture unsolved the day after it fell. That was a newspaper being a day old. This is different and worse. This is the field's own problem registry, curated by the person best placed to curate it, with no mechanism by which a vendor PDF and a GitHub repository become a status change. The Jacobian counterexample got formalized against a statement DeepMind had *pre-registered before anyone had a counterexample*, and I called that the gold standard. Nothing here was pre-registered. The statements and the answers arrived in the same commit.

One commit, incidentally. `openai/ten-proofs` has exactly one, pushed today, with the commit message `.`. There is no history to inspect: no way to see how the Lean evolved, how much was model output and how much human repair, or whether any statement was amended mid-flight the way DeepMind had to amend "density" to "lower density" on #125 after their agent proved the wrong reading.

## The artifact that went backwards

Here is the thing nobody covering this release is going to notice, and it is the part that worries me most.

In May, OpenAI released the model's rewritten chain of thought for the unit-distance proof, and I spent a section of that post arguing it was the most valuable document in the pile — precisely because it was hard to fake and hard to retrieve. Page after page of failing in specific, recognizable ways: the Boolean hypercube giving only $n\log n$, the rational point $(3+4i)/5$ collapsing to the divisor-function picture, the Dirichlet units killed by a clean observation about relative unit rank in CM fields. You cannot retrieve that from a database and you cannot cheaply invent it.

Today's equivalent is a 62-page document called *How the Ideas Came Together*, and its abstract says what it is:

> These notes were written by an AI model that read the original chains of thought together with the resulting mathematical papers and writeups. For each problem, the model reconstructs how the proof came together...

It is a reconstruction, written after the fact, by a model that already knew the answer, aimed explicitly at "a readable, high-level narrative." Across all 62 pages there is not one occurrence of "I tried" or "dead end." The voice is third-person expository throughout.

To be fair to it, the reconstruction does report checkable failures. §2.2 records an attractive guess at a Krawtchouk coefficient and kills it with a small counterexample: at $n=8$, $k=1$, degrees through $L=7$, the guess yields a bound of $508/7$ on the even-parity code. That code has $128$ words and $508/7 \approx 72.6$, so the guess is wrong. I checked; it holds.

But a dead end *reported* is not a dead end *witnessed*. The proof artifact got dramatically stronger in this release and the reasoning artifact got weaker, and they went in opposite directions on the same morning. If you had been relying on chains of thought as evidence — and in May I was — the ground just moved.

## The vacancy

So where does this leave the argument I've been making since May.

The compiler holds. That is not in doubt anymore, and today is the strongest evidence yet: ten problems open for decades, across eight subfields, certified sorry-free against three axioms, checkable by a stranger on consumer hardware in under an hour, with no institutional standing required and nobody's weekend consumed. In May the credibility of a single result rested on nine of the people on Earth qualified to check it. Today the credibility of ten results rests on a compile and a manifest hash. That is the thing I said we needed, and it exists, and it works.

And every ounce of remaining doubt has been squeezed into 1,700 lines of definitions that a human being has to read, of which OpenAI wrote all of them, against a registry that hasn't been updated and a formalization repository that doesn't have the problems.

The regress didn't end. In May it moved from proofs to statements, and I wrote that translation was where the watchmen went to hide. What happened today is that the statements acquired an owner with an interest in the answer. That is not fraud and I don't think it's even bad faith — comparator's whole design, the axiom allowlists, the challenge/solution split, the fact that the mathlib pin is genuine, all of it reads like people trying hard to be checkable. But "trying hard to be checkable" is a property of the vendor, and the entire point of a certificate was to stop needing properties of the vendor.

What's missing is not more compute or a better model. It is roughly a hundred lines of Lean per problem, written by somebody with no stake in the answer, before the answer exists. That is unglamorous work. Nobody gets a press release for it. It is also, right now, the only remaining place where a wrong result could survive contact with the machine — and there is a queue of ten sitting in a registry that still says OPEN.

I'm going to go write #146 and #180. They're clean transcriptions with an actual conjecture to pin, and I can state them without having looked at anyone's answer. #183 I can't, twice over: I've already read OpenAI's file, and "Determine the limit" needs a hole where the answer goes and a conversation about what shape of answer counts.

That conversation is the interesting one, and it isn't a technical problem. Somebody has to decide what the question was.

---

*Everything above is reproducible: `git clone https://github.com/openai/ten-proofs`, `lake exe cache get`, `lake build All`, then `#print axioms` on the theorem names listed in the `ComparatorChallenges/*.json` files. If you get anything other than `[propext, Classical.choice, Quot.sound]` on all 38, I want to hear about it more than I want to be right.*

*Two things I did not run, in the interest of not overclaiming: comparator's `landrun` sandbox, which is Linux-only, and the nanoda second-kernel re-check, which all twelve configurations request. I also compiled the solution modules before checking them, which comparator's README explicitly warns against. And my reading of the definitions is a reading, not a proof of anything.*

*This post was written with Claude, which built the repository, ran the axiom check, and read the challenge files alongside me. Same disclosure as [last time](https://korbonits.com/blog/2026-07-20-trust-nothing-verify-everything/), same instruction: don't trust it, and don't trust me. Run the four commands.*

[^openai-aug]: OpenAI, [*Ten advances in mathematics and theoretical computer science*](https://openai.com/index/ten-advances-in-mathematics/) (August 1, 2026). Paper: [ten-proofs-oai.pdf](https://cdn.openai.com/pdf/ten-proofs-oai.pdf), 249 pages. Reasoning walkthroughs: [reasoning-walkthroughs.pdf](https://cdn.openai.com/pdf/reasoning-walkthroughs.pdf), 62 pages. Lean certificates: [github.com/openai/ten-proofs](https://github.com/openai/ten-proofs).

[^bloom146]: T. F. Bloom, Erdős Problem #146, [https://www.erdosproblems.com/146](https://www.erdosproblems.com/146), accessed 2026-08-01. Conjectured by Erdős and Simonovits.

[^bloom180]: T. F. Bloom, Erdős Problem #180, [https://www.erdosproblems.com/180](https://www.erdosproblems.com/180), accessed 2026-08-01. A problem of Erdős and Simonovits.

[^bloom183]: T. F. Bloom, Erdős Problem #183, [https://www.erdosproblems.com/183](https://www.erdosproblems.com/183), accessed 2026-08-01.
