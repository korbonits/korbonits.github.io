---
title: "The Philosophy of Machine Learning, or: What Comes After Hegel?"
date: 2026-04-09
draft: false
tags: ["machine-learning", "ai", "philosophy", "llm", "personal"]
description: "A notebook entry that maps AI paradigms onto the history of Western philosophy — from scholasticism to Hegel — and asks what comes next when the current moment exhausts itself."
---

*A notebook entry from November 2023, a Pilot Custom 823, and the suspicion that the history of ideas has something to tell us about where AI is heading.*

-----

![Notebook entry, November 12 2023 — Kant, the newborn, automated reasoning](/images/notebook-philosophy-6828.jpg)

I was reading Kant while my daughter was a newborn.

Not because I’m a particularly disciplined person — I was doing it on Audible, mostly, at 3am, in the particular delirium that new parents will recognize. But I had this stubborn conviction that she should encounter the best of human thought first, before anything else. So: the *[Critique of Pure Reason](https://en.wikipedia.org/wiki/Critique_of_Pure_Reason)*, read aloud into my ears while she slept on my chest, neither of us fully conscious.

I don’t know how much of it I retained. But something lodged. And a few months later, sitting with a Pilot Custom 823 and a notebook, I found myself drawing a line between Kant and LLMs that I couldn’t quite shake.

## The Kantian LLM

Here’s the analogy that started it.

Kant’s great insight in the *Critique of Pure Reason* is that human knowledge isn’t just accumulated experience — there are structures we bring *to* experience that make it intelligible. Space, time, causality: these aren’t things we learn from the world, they’re the categories through which we perceive the world in the first place. A priori knowledge. The scaffolding, not the content.

LLMs, I kept thinking, are missing exactly this. They are almost entirely a posteriori — trained on the accumulated record of human experience, they are extraordinarily good at pattern-matching within that distribution. But they lack the a priori scaffolding. They don’t have a built-in structure for logical deduction. They can’t reliably perform [modus ponens](https://en.wikipedia.org/wiki/Modus_ponens). They confabulate because conditional probability, applied well enough, produces fluent text without producing *correct* reasoning. The next token looks right even when the chain of inference is broken.

This is why I’d been excited about Automated Reasoning as a complement to LLMs — not because AR is new, but because it provides exactly what LLMs lack: a formal, a priori rule structure that doesn’t bend to statistical pressure. SMT solvers, theorem provers, first-order logic systems: these are structures you bring *to* a problem, not structures you derive from training data. The decomposition maps cleanly. A priori: the AR rules. A posteriori: everything the LLM learned from text. Together, maybe, something more coherent than either alone.

*(I wrote this in November 2023. AR + LLMs is now a real research area. I’m choosing to feel ahead of the curve rather than behind. More on that in a [companion post](/blog/2026-04-02-i-was-thinking-about-llm-automated-reasoning-before-it-was-cool-and-i-wasn't-ready).)*

![Notebook entry — the Kantian decomposition, AR rules vs. training data, what comes after Kant](/images/notebook-philosophy-6829.jpg)

## Expert Systems as Scholasticism

But Kant was just the start of the analogy. Once I started pulling the thread, it kept going.

**[Expert systems](https://en.wikipedia.org/wiki/Expert_system)** — the dominant AI paradigm of the 1970s and 80s — feel, in retrospect, like Thomas Aquinas and the heights of Byzantine scholasticism. Enormously sophisticated, internally consistent, built on hand-crafted rules derived from human expertise. There’s something almost cathedral-like about a well-designed expert system: intricate, load-bearing, built to last. And also: brittle outside its domain, dependent on the authority of its rule-makers, ultimately limited by what could be explicitly articulated.

Scholasticism produced extraordinary thinkers. It also ran out of road.

![Notebook entry — expert systems as scholasticism, NLP as hermeneutics, classical ML as Humean](/images/notebook-philosophy-6830.jpg)

**Classical NLP** — the era of grammars, parse trees, [WordNet](https://wordnet.princeton.edu), hand-engineered features — feels like hermeneutics. Schleiermacher, maybe, or Herder. The core project is interpretation: what does this text mean, given these linguistic rules, in this context? There’s a real theory of language underneath it, a belief that meaning is recoverable through careful analysis of structure. It’s sophisticated. It’s also pre-statistical, pre-empirical in the modern sense — it believes in meaning as something to be decoded, not as something to be approximated.

**Classical machine learning** — Bayesian models, SVMs, carefully engineered feature spaces — feels Baroque. Or maybe Humean. Hume is actually the better fit. Hume looked at causality — one of Kant’s supposedly a priori categories — and said: I don’t think we actually perceive cause and effect. We perceive constant conjunction. We observe that B follows A, repeatedly, and we *infer* causality. We can’t be certain of it. The IID assumption in machine learning is Humean to the core: we observe draws from a distribution and we model their regularities. We don’t claim to understand why the world is the way it is. We claim to predict it.

## The Hegelian Moment

Which brings us to where we are now. And here the analogy gets interesting.

What comes after Kant, in the history of philosophy, is Hegel. And Hegel is strange and difficult and widely misunderstood, but at the center of his thought is something genuinely important: reality is not static, not a fixed set of categories applied to a fixed world. It unfolds. It develops. It is, in some deep sense, a process — a dialectical movement in which the categories themselves evolve as history proceeds, in which mind and world develop together in a way that cannot be understood from any fixed vantage point.

Look at what’s actually happening in AI and tell me that doesn’t feel Hegelian.

The models are trained on our collective knowledge — the entire written record of human thought, up to now. They are then used to generate new knowledge, which enters the world, which becomes training data for the next generation of models. The data and the models co-evolve. The systems are deployed into the world and the world changes in response, and those changes feed back into the systems. There is an inexorable unfolding here, a dialectical movement between human knowledge and machine synthesis of that knowledge, that is unprecedented in scale and genuinely difficult to conceive of in any other terms.

Add retrieval: now the models have access to proprietary knowledge, to live information, to the accumulated records of specific institutions and individuals. The Hegelian [Geist](https://en.wikipedia.org/wiki/Geist) — the world-spirit, the collective mind moving through history — is perhaps the least crazy metaphor available for what’s actually happening.

Perhaps it is even more complicated than that.

## Continuing the Analogy Further

If this periodization has any traction, it raises an obvious question: what comes after Hegel?

In the history of philosophy, the post-Hegelian moment is fractured and multiple. You get Marx, who takes the dialectic and grounds it in material conditions rather than Geist. You get Kierkegaard, who rejects the system entirely in favor of the individual and the leap of faith. You get Nietzsche, who declares the death of the structures that made the system possible and asks what comes after. You get Heidegger’s *[Being and Time](https://en.wikipedia.org/wiki/Being_and_Time)* — a return to the most fundamental questions about existence, prior to any system. You get Sartre’s *[Being and Nothingness](https://en.wikipedia.org/wiki/Being_and_Nothingness)* — radical freedom, groundlessness, the absence of essential structure. You get, eventually, postmodernism: Foucault, Derrida, the suspicion that the grand narratives were always interested constructions.

And you get Žižek, who is basically Hegel but aware that Hegel has been dissolved, which is a strange place to be.

I don’t know which of these the next era of AI will rhyme with. But the question feels generative. If the current moment is Hegelian — collective, synthetic, dialectically unfolding — then the next moment might look like:

**Marx**: a grounding of AI’s development in material conditions, power structures, labor, production. Who controls the training data? Who owns the models? What are the actual economic forces driving the unfolding?

**Kierkegaard**: a reaction toward the individual, the particular, the irreducibly personal — AI that is not collective synthesis but something more like a genuine interlocutor for the singular human being. Which is interestingly where a lot of the most compelling applied AI work is actually going.

**Heidegger**: a return to foundations. What does it actually mean to understand? What is the relationship between the symbol and the thing? What have we been assuming, and should we stop assuming it?

**Postmodernism**: the collapse of the grand scaling narrative. The recognition that the assumptions underlying the current paradigm are historically contingent, not eternal truths. The transformer isn’t the end of history. It’s a moment in a longer story.

I’ll confess I find the Heideggerian reading most interesting, and not only because of my prior convictions about what LLMs are and aren’t doing. The question of what genuine understanding requires — whether it requires embodiment, stakes, something at risk — feels like the right question to be asking. The current systems are extraordinary. I don’t think they understand anything, in the sense that matters.

## Why Bother With the Analogy?

Fair question. Philosophy of mind analogies to ML are a genre with a high noise-to-signal ratio. Why add to it?

Here’s my answer: the history of ideas is a map of the territory we’re actually traversing. We have been here before — not in the sense that AI has existed before, but in the sense that humanity has previously built systems of thought that seemed to explain everything, hit their limits, and gave way to something that could only be understood in retrospect. The scholastics were not stupid. The hermeneuticists were not wrong. Hume was one of the sharpest minds in the history of philosophy. Each paradigm captured something real and missed something real, and the shape of what it missed turned out to matter enormously.

If the current paradigm is Hegelian — and I think it is, in important ways — then the shape of what it might miss is something we can think about now, rather than waiting for the paradigm to exhaust itself and then reconstructing the lesson in hindsight.

What comes after Hegel is not a prediction. It’s a set of questions worth taking seriously before we need the answers.

## The ∞ at the Bottom of the Page

I ended the notebook entry with an infinity symbol. I don’t remember exactly what I meant by it. Probably something about the self-referential quality of the whole thing — models trained on human thought, humans thinking about models, the loop. Or maybe just that it was late and I was being dramatic.

Either way, the questions feel open. I’d suggest starting with Kant and Hegel — not as historical curiosities but as live frameworks. The decomposition between a priori structure and a posteriori experience is still the right decomposition for thinking about what LLMs lack. The dialectical unfolding of knowledge and systems and world is still the right frame for thinking about what’s happening at scale.

Good questions. Could be generative for some scientific *gedanken*.

![Notebook entry — Hegelian AI, gedanken, the infinity symbol](/images/notebook-philosophy-6831.jpg)

∞

-----

*This post is part of an ongoing series on ML, philosophy of mind, and what it means to build systems that think. Or don’t.*
 
