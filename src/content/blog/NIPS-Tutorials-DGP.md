---
title:  "Deep Probabilistic Modelling with Gaussian Processes #NIPS2017"
date:   2017-12-04 12:00:00
draft: false
description: "Lecture notes from Neil Lawrence's NIPS 2017 tutorial on deep probabilistic modelling with Gaussian processes — from GPs to deep GPs and variational inference."
tags: ["machine-learning", "gaussian-processes", "probabilistic-ml", "nips", "notes"]
---

*I wrote these notes in December 2017 after attending my first NeurIPS and never published them. Eight years later, I'm hitting publish with no edits — just this note as context.
Reading them back is strange. The five takeaways from that week — Bayesian deep learning, fairness and bias, the need for theory, deep RL, GANs — all became defining research arcs of the decade that followed. The Rahimi & Recht "alchemy" talk, which was controversial at the time, looks prophetic now. And the throwaway concern about "generating realistic fake videos with geopolitical consequences" landed harder than I think anyone in that room expected.*

*The corporate circus section is its own kind of time capsule: an Intel Flo Rida concert, Nvidia handing out $3k GPUs to the audience, and an invite-only Tesla party with Elon Musk and Andrej Karpathy. A different era.*

*What strikes me most, though, is how optimistic and open everything felt. The field was moving fast but still felt legible — you could go to one conference and get your arms around the major themes. That's long gone. Anyway — notes from the before-times, published from the future.*

![NIPS](/assets/nips.png)

## Deep Probabilistic Modelling with Gaussian Processes
### [Neil D. Lawrence](http://inverseprobability.com/)

ML = data + models → prediction. But predictions alone aren't enough — we need to make decisions. To combine data and model we need: (1) a prediction function, and (2) an objective function. Sources of **uncertainty**: scarcity of training data, mismatch of prediction functions, uncertainty in the objective/cost function.

Following MacKay (1992) and Neal (1994): take a probabilistic approach.

### 1. Neural Networks as Probabilistic Models

A neural network computes:

$$f(\mathbf{x}) = \mathbf{W}_2^\top \phi(\mathbf{W}_1, \mathbf{x})$$

where $\phi$ is a nonlinear activation. This is linear in the parameters $\mathbf{W}_2$ but nonlinear in the inputs — **adaptive basis functions**. $\mathbf{W}_1$ are fixed for a given analysis; in ML we optimize both $\mathbf{W}_1$ and $\mathbf{W}_2$.

**Probabilistic inference:**
- $\mathbf{y}$ ← data
- $p(\mathbf{y}^*, \mathbf{y})$ ← model (joint distribution over world)
- $p(\mathbf{y}^* \mid \mathbf{y})$ ← prediction (posterior)

The goal: $p(\mathbf{y}^* \mid \mathbf{y}, \mathbf{X}, \mathbf{x}^*)$ — the predictive distribution at a new point $\mathbf{x}^*$.

The likelihood of a data point: $p(\mathbf{y} \mid \mathbf{x}, \mathbf{W})$. Under iid noise (the iid assumption is about the *noise*, not the underlying function):

$$p(\mathbf{y} \mid \mathbf{X}, \mathbf{W}) = \prod_i p(y_i \mid \mathbf{x}_i, \mathbf{W})$$

Commonly Gaussian likelihood; MLE for supervised learning. With priors over latents, you get unsupervised learning.

**Graphical models** represent joint distributions through conditional dependencies (e.g., Markov chains). Performing inference is easy to write down but computationally challenging — high-dimensional integrals.

### 2. From Neural Networks to Gaussian Processes

Fix $\mathbf{W}_1$. Place a Gaussian prior over $\mathbf{W}_2$:

$$\mathbf{W}_2 \sim \mathcal{N}(0, \sigma^2 \mathbf{I})$$

Since sums and scalings of Gaussians are Gaussian, marginalizing out $\mathbf{W}_2$ gives:

$$p(\mathbf{y}) = \int p(\mathbf{W}_2)\, p(\mathbf{y} \mid \mathbf{W}_2)\, d\mathbf{W}_2$$

$\mathbf{y}$ is distributed with zero mean and covariance $\mathbf{K} = \Phi \Phi^\top / H$ where $\Phi$ is the design matrix of activations. **A neural network with a Gaussian prior over its output weights is already a Gaussian process** — but a degenerate one.

**Degeneracy:** the rank of $\mathbf{K}$ is at most $H$ (the number of hidden units). As $n \to \infty$, the covariance matrix is not full rank: $|\mathbf{K}| = 0$. The model can't respond to new data as it comes in — it's parametric.

**Neal's insight (1994):** take $H \to \infty$. Sample infinitely many hidden units in the kernel function. The prior doesn't need to be Gaussian. Scale output variance down as $H$ increases. You get a **non-degenerate Gaussian process**.

### 3. Gaussian Processes

A GP is a distribution over functions: any finite collection of function values $\{f(\mathbf{x}_1), \ldots, f(\mathbf{x}_n)\}$ is jointly Gaussian. Fully specified by:

- **Mean function:** $m(\mathbf{x}) = \mathbb{E}[f(\mathbf{x})]$
- **Covariance (kernel) function:** $k(\mathbf{x}_i, \mathbf{x}_j) = \mathbb{E}[(f(\mathbf{x}_i) - m(\mathbf{x}_i))(f(\mathbf{x}_j) - m(\mathbf{x}_j))]$

The kernel matrix: $K_{ij} = k(\mathbf{x}_i, \mathbf{x}_j)$.

**Posterior:** given observations $\mathbf{y} = f(\mathbf{X}) + \boldsymbol{\varepsilon}$, $\boldsymbol{\varepsilon} \sim \mathcal{N}(0, \sigma^2 \mathbf{I})$, the posterior at test point $\mathbf{x}^*$ is:

$$p(f^* \mid \mathbf{X}, \mathbf{y}, \mathbf{x}^*) = \mathcal{N}(\mu^*, \sigma^{*2})$$

$$\mu^* = \mathbf{k}_*^\top (\mathbf{K} + \sigma^2 \mathbf{I})^{-1} \mathbf{y}$$

$$\sigma^{*2} = k(\mathbf{x}^*, \mathbf{x}^*) - \mathbf{k}_*^\top (\mathbf{K} + \sigma^2 \mathbf{I})^{-1} \mathbf{k}_*$$

GPs let you analytically compute the posterior mean and variance at all points. The **exponentiated quadratic (RBF) kernel** gives infinite smoothness — not always desirable (Brownian motion is also a GP, with very different smoothness properties).

**Sparse GPs:** full GP inference is $O(n^3)$ in time and $O(n^2)$ in storage (due to the matrix inversion $(\mathbf{K} + \sigma^2 \mathbf{I})^{-1}$). In practice, use a sparse GP with $m \ll n$ **inducing variables** to get a low-rank approximation of the full covariance.

### 4. Deep Neural Networks and Bottleneck Layers

A matrix between two 1000-unit layers has $10^6$ parameters — prone to overfitting. One fix: parametrize $\mathbf{W}$ via its SVD to create **bottleneck layers**. Stacking neural networks gives a composite function.

If you want to eliminate NN parameters entirely: replace each layer with a GP and integrate them out. Taking each layer to infinitely many units gives a **vector-valued GP**. Bottleneck layers are critical in this construction.

### 5. Deep Gaussian Processes

A deep GP is a composition of GPs:

$$g(\mathbf{x}) = f_L(f_{L-1}(\cdots f_2(f_1(\mathbf{x}))\cdots))$$

where each $f_\ell$ is a GP. This is equivalent to a Markov chain under the Markov condition:

$$p(\mathbf{y} \mid \mathbf{x}) = p(\mathbf{y} \mid \mathbf{f}_L)\, p(\mathbf{f}_L \mid \mathbf{f}_{L-1})\, p(\mathbf{f}_{L-1} \mid \mathbf{f}_{L-2}) \cdots p(\mathbf{f}_1 \mid \mathbf{x})$$

**Why go deep?**
- GPs give priors over functions
- Derivatives of a GP are a GP (when they exist)
- Some kernels are universal approximators
- Depth enables **abstraction of features** and handles non-Gaussian derivative distributions

**Caveat:** Gaussian derivatives can be problematic — many functions (jump functions, heavy-tailed) don't have Gaussian derivatives. Depth helps encode these via process composition.

**Difficulties:**
- Propagating probability distributions through nonlinearities
- Normalization of the distribution becomes intractable

**Solution:** use a variational approach to stack GP models [Damianou & Lawrence, 2013]. As depth increases, the derivative distribution becomes heavy-tailed [Duvenaud et al., 2014] — which is actually desirable for modeling complex functions.

Deep GPs handle heteroskedasticity well (e.g., Olympic marathon running times where length scales change over time). Can be extended with a **shared latent variable model (LVM)** for multi-output settings.

See: [How deep are deep GPs?](https://scirate.com/arxiv/1711.11280)
