---
title: "Trust Nothing, Verify Everything: reproducing the Jacobian counterexample from my couch"
date: 2026-07-20
draft: false
description: "The Jacobian conjecture fell yesterday, during the World Cup final, via a tweet. Wikipedia still says it's open. Here's how I checked it myself in two cells of sympy — and why the fact that I could is the real story."
tags: ["math", "verification", "ai", "jacobian-conjecture", "war-stories"]
---

# Trust Nothing, Verify Everything: reproducing the Jacobian counterexample from my couch

Yesterday, while most of the world was watching the World Cup final, the Jacobian conjecture — open since Keller stated it in 1939, Problem 16 on Smale's list of problems for *this century* — was disproved. Not in the *Annals*. Not on arXiv. On X, in a tweet by [Levent Alpöge](https://x.com/__alpoge__/status/2079028340955197566), a number theorist who credited "my close friend fable for working during the world cup final." Fable is Anthropic's new Claude model.

As I write this, the [Wikipedia page](https://en.wikipedia.org/wiki/Jacobian_conjecture) still calls it "a famous unsolved problem." There is no paper. There is no referee report. There is a tweet containing three polynomials.

Ten years ago, the correct response to this situation was: wait. Wait for the arXiv post, wait for the experts, wait for the seminar where someone finds the gap — because the Jacobian conjecture is *notorious* for wrong proofs; the Wikipedia page says so in its second paragraph. The conjecture has eaten careers.

But this isn't a proof. It's a counterexample. And that changes everything about who gets to know it's true, and when.

## The claim

The Jacobian conjecture says: if a polynomial map $F: \mathbb{C}^n \to \mathbb{C}^n$ has a Jacobian determinant that's a nonzero *constant*, then $F$ has a polynomial inverse. Locally, a nonvanishing Jacobian always gives you an inverse — that's the inverse function theorem from multivariable calculus. The conjecture claimed that polynomial rigidity plus a globally constant Jacobian upgrades local invertibility to global. It's one of those problems you can state to a calculus student and then fail to solve for 87 years.

Alpöge's tweet exhibits a specific map $F: \mathbb{C}^3 \to \mathbb{C}^3$:

$$
F(x, y, z) = \begin{pmatrix}
(1+xy)^3\, z + y^2 (1+xy)(4+3xy) \\
y + 3x(1+xy)^2\, z + 3xy^2(4+3xy) \\
2x - 3x^2 y - x^3 z
\end{pmatrix}
$$

For this to kill the conjecture, exactly two things must be true:

1. $\det JF$ is a nonzero constant.
2. $F$ is not invertible.

Both are *finite, mechanical checks*. No class field theory, no referee, no trust. So I checked.

## Cell one: the determinant

```python
import sympy as sp

x, y, z = sp.symbols('x y z')

F = sp.Matrix([
    (1 + x*y)**3 * z + y**2 * (1 + x*y) * (4 + 3*x*y),
    y + 3*x*(1 + x*y)**2 * z + 3*x*y**2 * (4 + 3*x*y),
    2*x - 3*x**2*y - x**3*z,
])

print(sp.simplify(F.jacobian([x, y, z]).det()))
```

```
-2
```

Not numerically-close-to-negative-two. Symbolically, exactly, $-2$. Every monomial in that determinant cancels except the constant. Condition one: verified, by my laptop, in under a second.

## Cell two: the collision

The cheapest way to show a map has no inverse — polynomial or otherwise — is to find two points with the same image. A non-injective map isn't invertible, full stop.

Here's a hint at where to look (this structure is *why* the example works, and it's the part worth staring at): the third coordinate factors as $f_3 = x(2 - 3xy - x^2z)$. So the fiber over the plane $w_3 = 0$ contains two entirely different branches — the plane $x = 0$, and the surface $z = (2-3xy)/x^2$. Two branches, one target. Collisions live where they land on each other:

```python
p, q = {x: 0, y: 6, z: -142}, {x: 1, y: 0, z: 2}
print(F.subs(p).T, F.subs(q).T)
```

```
Matrix([[2, 6, 0]]) Matrix([[2, 6, 0]])
```

$F(0, 6, -142) = F(1, 0, 2) = (2, 6, 0)$. Two distinct points, one image. (If you're worried one collision is a fluke of some degenerate locus: $F(0,7,-193) = F(2,1,-1) = (3,7,0)$ too. The branch structure gives you a collision for every point on a surface's worth of choices.)

That's it. That's the whole verification. The Jacobian conjecture is false.

Two footnotes that fall out for free. The collision points are *integers*, so by a 1983 theorem of Connell and van den Dries the conjecture dies over every field of characteristic zero, not just $\mathbb{C}$. And padding the map with identity coordinates kills every dimension $n \geq 3$. What survives is exactly Keller's original two-variable problem — $n = 2$ remains open, and is suddenly the most interesting question in the area.

## What I didn't need

Here is the part I actually want to write about.

To know — not believe, *know* — that an 87-year-old conjecture is false, I did not need to trust:

- **Alpöge.** He seems trustworthy! Doesn't matter. Didn't need him.
- **Anthropic**, whose model found the map and who have every commercial incentive to promote the result.
- **Wikipedia**, which is wrong right now, in the charming way a newspaper from yesterday is wrong.
- **Peer review**, which will eventually bless this in a journal, sometime next year, to an audience that stopped caring in July.
- **The AI that helped me write this post.** More on that below.

I needed a tweet, sympy, and ninety seconds. The entire trust chain collapsed into two artifacts anyone can rerun. Kevin Buzzard has been [making this argument for nine years](https://xenaproject.wordpress.com/2026/07/20/human-mathematicians-are-being-outcounterexampled/) — his post yesterday, which you should read, chronicles an astonishing two-month wave: Erdős' unit distance conjecture in May, a 60-year-old Grothendieck question about group schemes two weeks ago, and now this. His verification standard is Lean, and rightly so — Paul Lezeau has [already formalized this counterexample](https://github.com/google-deepmind/formal-conjectures/pull/4474) against DeepMind's pre-registered statement of the conjecture, which is the gold-standard version of what I did on my couch. Buzzard checked the Grothendieck disproof in five minutes flat, and his protocol is worth internalizing: trusted definitions, correct statement, compiles. He now refuses to read AI-generated informal mathematics *at all* — and after this weekend I think he's right, for a reason that has nothing to do with AI skepticism.

Informal mathematics always ran on credit. The author's reputation was collateral; refereeing worked because writing a plausible 12-page paper was expensive, so the mere existence of one carried signal. AI took the cost of generating plausible mathematics to zero, and the collateral went with it. What's left is a simple repricing: **if your claim comes with a certificate I can run, I'll know it's true in minutes. If it doesn't, it's now spam** — even if it's correct. The Jacobian counterexample is the purest possible specimen because the certificate fits in a tweet.

The disorienting part isn't that a machine did the math. It's that *authority didn't participate at any point*. Result found by a model, announced on a social network, verified by strangers running code, formalized against a conjecture statement that DeepMind had registered before any counterexample existed — and the traditional institutions of mathematical trust (journals, referees, even the encyclopedia) haven't yet noticed it happened. Trust didn't disappear. It migrated — out of people and institutions, into artifacts.

## Disclosure, which is also the point

This post was co-written with Claude — specifically Fable, the same model credited in Alpöge's tweet, which is either a conflict of interest or a controlled experiment depending on how you hold it. So hold it the right way: **don't trust it.** Don't trust me either. Every load-bearing claim above is one of the two code cells, and the cells don't care who wrote them. Run them yourself. That's not a rhetorical flourish — it's the entire thesis of the post, applied reflexively. The first piece of AI-assisted writing I've published is about how you no longer have to trust the author, and I'd be embarrassed to have it any other way.

## What's next

There's a bigger post coming. The short version: this wave of counterexamples only lands where the *statements* have already been formalized — Buzzard's five-minute check and Lezeau's instant formalization were both downstream of years of unglamorous definitional infrastructure. Whole fields (including the one I contribute to in Mathlib, differential geometry) currently can't be machine-disproven because they can't yet be machine-*stated*. Who builds that statement layer, and for which fields, is quietly determining where the next 87-year-old conjecture falls. That's the real story, and it deserves more than a same-day post.

For now: the Jacobian conjecture is dead, it died during a soccer match, and you don't have to take anyone's word for it — which, if you think about it, is the most optimistic sentence about the epistemics of mathematics written in a long time.

---

*The two cells above are the complete verification. Python 3, `pip install sympy`, no other dependencies. If you get anything other than `-2` and a matching pair, I want to hear about it more than I want to be right.*
