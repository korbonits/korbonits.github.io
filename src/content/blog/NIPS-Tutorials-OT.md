---
title:  "A Primer on Optimal Transport #NIPS2017"
date:   2017-12-04 08:00:00
draft: false
description: "Lecture notes from Marco Cuturi and Justin Solomon's NIPS 2017 tutorial on optimal transport — Wasserstein distances, Sinkhorn's algorithm, and applications in generative models and ML."
tags: ["machine-learning", "optimal-transport", "math", "nips", "notes"]
---

*I wrote these notes in December 2017 after attending my first NeurIPS and never published them. Eight years later, I’m hitting publish with no edits — just this note as context.
Reading them back is strange. The five takeaways from that week — Bayesian deep learning, fairness and bias, the need for theory, deep RL, GANs — all became defining research arcs of the decade that followed. The Rahimi & Recht “alchemy” talk, which was controversial at the time, looks prophetic now. And the throwaway concern about “generating realistic fake videos with geopolitical consequences” landed harder than I think anyone in that room expected.*

*The corporate circus section is its own kind of time capsule: an Intel Flo Rida concert, Nvidia handing out $3k GPUs to the audience, and an invite-only Tesla party with Elon Musk and Andrej Karpathy. A different era.*

*What strikes me most, though, is how optimistic and open everything felt. The field was moving fast but still felt legible — you could go to one conference and get your arms around the major themes. That’s long gone. Anyway — notes from the before-times, published from the future.*

![NIPS](/assets/nips.png)

## [A Primer on Optimal Transport](https://nips.cc/Conferences/2017/Schedule?showEvent=8736)
### [Marco Cuturi](http://marcocuturi.net/) | [Justin M Solomon](http://people.csail.mit.edu/jsolomon/)

What is optimal transport? The natural geometry for probability measures — it puts a distance on the space of distributions, enabling comparisons of "bags of features." Given statistical models $p$ and $p'$, OT gives us a notion of divergence to measure how well we're doing.

**Outline:** intro → algorithms → apps ($W$ as a loss) → apps ($W$ for estimation). Slides at [optimaltransport.github.io](https://optimaltransport.github.io).

### 1. The Monge Problem

Monge (1781): move a pile of dirt $\mu$ into a hole $\nu$ (with shovels).

- $\mu(x)$: height of pile at $x$
- $y = T(x)$: destination for point $x$ — a map from $\mu$ to $\nu$
- $d(x, T(x))$: distance traveled
- $\mu(x) \cdot d(x, T(x))$: work done at $x$

$T$ must satisfy the **pushforward constraint** $T_\#\mu = \nu$, meaning $\mu(A_1) + \mu(A_2) = \nu(B)$ for preimages mapping into $B$.

The Monge problem: find the map $T$ that minimizes total work:

$$\min_{T \,:\, T_\#\mu = \nu} \int d(x,\, T(x))\, d\mu(x)$$

**Caveat:** an optimal Monge map $T^*$ does not always exist (e.g., when $\mu$ is a Dirac mass and $\nu$ is not).

### 1a. The Kantorovich Relaxation

Kantorovich's insight: instead of a deterministic map, allow **measure couplings** (joint distributions) $\gamma \in \Pi(\mu, \nu)$ — i.e., $\gamma(x, y) \geq 0$ with marginals $\mu$ and $\nu$. This is just a linear program.

**Primal (Kantorovich):**

$$\min_{\gamma \in \Pi(\mu,\nu)} \int d(x,y)\, d\gamma(x,y)$$

**Dual (potential functions):**

$$\max_{f,\, g} \int f\, d\mu + \int g\, d\nu \quad \text{s.t.} \quad f(x) + g(y) \leq d(x,y)$$

The dual is elegant: $f$ and $g$ are the potential functions, and the dual tells you which points are most "expensive" to transport.

**Proposition:** for well-behaved cost functions, if $\mu$ has a density then an optimal Monge map $T^*$ between $\mu$ and $\nu$ exists.

### 1b. $p$-Wasserstein Distance

The $p$-Wasserstein distance between probability measures $\mu$ and $\nu$:

$$W_p(\mu, \nu) = \left( \inf_{\gamma \in \Pi(\mu,\nu)} \int d(x,y)^p\, d\gamma(x,y) \right)^{1/p}$$

This is a true metric on the space of probability measures. The geometry it induces is very different from information-theoretic metrics like KL divergence. McCann (1995) showed it gives rise to **displacement interpolation** — geodesics in the space of measures. (Solomon '15 has nice applications.)

### 2. How to Compute OT

Four cases:
1. discrete → discrete
2. discrete → continuous
3. continuous → continuous
4. continuous → discrete *(open)*

Cases 2–3 are largely "up for grabs." Easy special cases:

- **Univariate:** compute CDFs and quantile functions. $W_p$ has a closed form: $W_p(\mu,\nu)^p = \int_0^1 |F_\mu^{-1}(t) - F_\nu^{-1}(t)|^p\, dt$
- **Gaussians:** closed form, $T$ is linear.
- **Dirac masses:** $W_p(\delta_x, \delta_y) = d(x,y)$ — Wasserstein distance between point masses equals the ground distance.
- **Equal number of points:** reduces to the Monge problem (an assignment problem).

**Complexity of the LP:** $O(n^3 \log n)$ via min-cost flow. Ouch.

### Entropic Regularization and Sinkhorn

Optimal solutions $P^*$ to the LP are vertices of a polytope — unstable, non-unique, and non-differentiable. We want something faster, scalable, and differentiable.

**Entropic regularization** (Shannon entropy $H(\gamma) = -\sum_{ij} \gamma_{ij} \log \gamma_{ij}$):

$$\min_{\gamma \in \Pi(\mu,\nu)} \langle C, \gamma \rangle - \varepsilon\, H(\gamma)$$

As $\varepsilon \to \infty$, the solution approaches the independent coupling $\mu \otimes \nu$; as $\varepsilon \to 0$, it recovers the Monge solution. The regularization makes the problem strictly convex and smooth. [Wilson '62]

This can be solved with simple Lagrangians — leading to **Sinkhorn's algorithm**: alternately rescale rows and columns of the kernel matrix $K_{ij} = e^{-C_{ij}/\varepsilon}$ to match the marginals $\mu$ and $\nu$.

Sinkhorn = block coordinate ascent on the dual. [Altschuler et al. '17]

- **Convergence:** linear $O(nm)$ in general; $O(n \log n)$ on gridded spaces using convolutions.
- Sinkhorn interpolates between $W$ (hard OT) and MMD (kernel two-sample test).

**Sample complexity caveat** [Hashimoto '16, Bonneel '16, Shalit '16]: error in $W$ decreases very slowly in $n$ — bad sample complexity. The Wasserstein LP is not well-suited for high-dimensional data directly.

### 3. Applications

**Retrieval:** [Kusner '15] Word Mover's Distance — document similarity via OT over word embeddings.

**Barycenters:** averaging measures under $W$ vs. $L^2$ gives very different results. The **Wasserstein barycenter** of distributions $\{\mu_k\}$ with weights $\{\lambda_k\}$:

$$\bar{\mu} = \arg\min_{\mu} \sum_k \lambda_k W_2(\mu, \mu_k)^2$$

Averaging histograms is an LP; or use primal descent on regularized $W$ [Cuturi '14]. Application: brain imaging, finding smooth interpolations between distributions.

**Wasserstein Posterior (WASP):** aggregate distributed posteriors using Wasserstein barycenters [Srivastava '15].

**Wasserstein Propagation** [Solomon '14]: semi-supervised learning on graphs — propagate label distributions via OT. Could fix label noise or handle missing data.

**Dictionary learning / topic models** [Rolet '16]: represent documents as mixtures of dictionary elements under a Wasserstein loss.

**Wasserstein PCA:** generalized principal geodesics in the space of measures (negative curvature space — worth investigating further).

**Distributionally robust optimization** [Esfahani '17]: learning with Wasserstein ambiguity — robust to perturbations of the training distribution (minimax formulation).

**Domain adaptation:**
1. Estimate transport map $T$ from source to target domain
2. Transport labeled source samples to target domain
3. Train classifier on transported samples

**Generative models:** density fitting via maximum likelihood is just minimizing $\text{KL}(p_\text{data} \| p_\theta)$. Instead, use a low-dimensional latent space with pushforward $f_\theta : \mathcal{Z} \to \mathcal{X}$:

$$\min_\theta\, W(f_{\theta\,\#}\,\mu_z,\, \nu_\text{data})$$

This is the Wasserstein GAN formulation [Arjovsky et al. '17] — use $W$ as the loss between data and model rather than JS divergence.
