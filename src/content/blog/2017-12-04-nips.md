---
title:  "NeurIPS notes"
date:   2017-12-04 12:00:00
draft: false
description: "Notes from NIPS 2017 — covering geometric deep learning, GAN theory, reinforcement learning, fairness in ML, Bayesian deep learning, and more."
tags: ["machine-learning", "nips", "notes", "deep-learning", "reinforcement-learning"]
---

*I wrote these notes in December 2017 after attending my first NeurIPS and never published them. Eight years later, I'm hitting publish with no edits — just this note as context.
Reading them back is strange. The five takeaways from that week — Bayesian deep learning, fairness and bias, the need for theory, deep RL, GANs — all became defining research arcs of the decade that followed. The Rahimi & Recht "alchemy" talk, which was controversial at the time, looks prophetic now. And the throwaway concern about "generating realistic fake videos with geopolitical consequences" landed harder than I think anyone in that room expected.*

*The corporate circus section is its own kind of time capsule: an Intel Flo Rida concert, Nvidia handing out $3k GPUs to the audience, and an invite-only Tesla party with Elon Musk and Andrej Karpathy. A different era.*

*What strikes me most, though, is how optimistic and open everything felt. The field was moving fast but still felt legible — you could go to one conference and get your arms around the major themes. That's long gone. Anyway — notes from the before-times, published from the future.*

---

## Geometric Deep Learning on Manifolds and Graphs

Text, audio, images — convolutions are integral to their success. But what about social networks, functional networks, regulatory networks, 3D shapes? What geometric structure do CNNs exploit, and how do we leverage it in non-Euclidean domains? [Duvenaud et al. 2015]

Two setups: (1) manifolds, (2) graphs. Key CNN properties to generalize:
- $O(1)$ parameters per filter (independent of input size)
- $O(n)$ complexity per layer
- $O(\log n)$ layers for classification
- Translation invariance, multiscale structure, localized filters

**Graph Laplacian:** for a weighted undirected graph $G = (V, E, W)$, the unnormalized graph Laplacian is:

$$\mathbf{L} = \mathbf{D} - \mathbf{W}$$

where $\mathbf{D}$ is the degree matrix. The **Dirichlet energy** of a signal $f$ on the graph measures smoothness:

$$f^\top \mathbf{L} f = \sum_{(i,j) \in E} W_{ij}(f(i) - f(j))^2$$

The Laplace-Beltrami operator is the continuous limit of the graph Laplacian on Riemannian manifolds — you still get Dirichlet energy.

**Spectral graph convolution:** the graph Fourier basis $\Phi$ consists of the eigenvectors of $\mathbf{L}$. Spectral convolution:

$$g = \Phi\, \mathbf{W}\, \Phi^\top f$$

where $\mathbf{W}$ is a diagonal matrix of learned filter coefficients. Problems: $O(n)$ parameters per layer, $O(n^2)$ for forward/inverse transforms (no FFT on graphs), no spatial localization guarantee, basis-dependent (doesn't generalize across graphs).

**Fix:** parametrize the filter as a polynomial in $\lambda$ (the eigenvalues):

$$\hat{h}(\lambda) = \sum_{k=0}^{K} \theta_k \lambda^k$$

This gives $O(1)$ parameters per layer, $r$-hop spatial support (filters affect only $r$-hop neighbors), and $O(nr)$ complexity — no eigendecomposition needed. [Defferrard et al. 2016, ChebNet]

**GNNs (spatial):** simple graph NN — mean over neighbors:

$$h_v^{(l+1)} = \sigma\left(\mathbf{W}^{(l)} \cdot \text{MEAN}\left(\{h_u^{(l)} : u \in \mathcal{N}(v) \cup \{v\}\}\right)\right)$$

[Kipf & Welling 2016] Can decorate edges and vertices with types. Neural message passing / interaction networks generalize this further.

**Spatial charting methods:** MoNet defines convolution as an integral over Gaussian mixtures in geodesic polar coordinates — handles non-Euclidean domains without spectral methods.

**Applications:** deformable 3D correspondence (FMNet), geometric matrix completion (Netflix-style recommender systems with user/item graphs), particle physics, chemistry.

---

## Optimization Landscape of Tensor Decompositions

Why care about loss surfaces? Model/loss → loss surface → learning dynamics and generalization [Keskar et al. '17, Ding et al. '17].

**Tensor decomposition:** for a symmetric tensor of order $p$ and rank $n$, find components $a_1, \ldots, a_n \in \mathbb{R}^d$ — a natural extension of the eigenvector problem. Connects to learning NNs via method of moments for latent variable models.

**Random overcomplete setting:** $a_i$ Gaussian, $d \ll n \ll d^2$. Maximize a degree-$p$ polynomial over the unit sphere — looking for approximate global maxima.

**Main result:** let $Q = \mathbb{E}[f(\text{random init})] = 3n/d^2$. In the superlevel set $S = \{x : f(x) \geq 1.1Q\}$, with high probability $f$ has exactly $2n$ local maxima: $\pm a_1, \ldots, \pm a_n$. (Exponentially many saddle points, but SGD helps escape them.)

Tool: **Kac-Rice formula** counts stationary points of a random function/field $f : \mathbb{R}^d \to \mathbb{R}$ — combined with random matrix theory [Auffinger et al. '10, spin-glass model].

---

## Robust Optimization for Non-Convex Objectives

Goal: train DNNs robust to noise. Noisy learning → optimization problem over loss functions. Key reduction: $\alpha$-robust optimization reduces to $\alpha$-stochastic optimization (no convexity assumptions needed). Algorithm uses multiplicative weights update. Generalizes beyond statistical learning to adversarial attacks.

---

## Bayesian Optimization Using Gradients

GP regression naturally accommodates derivative observations. Standard acquisition functions: Expected Improvement (EI), Knowledge Gradient (KG).

**Core contribution:** novel acquisition function $d\text{KG}$ — samples at the point that maximizes the derivative of the knowledge gradient factor (via infinitesimal perturbation analysis / envelope theorem). $d\text{KG}$ samples far from already-sampled points; $d\text{EI}$ samples near them. $d\text{KG}$ converges to the global optimum. Using derivatives makes Bayesian optimization significantly faster.

---

## Gradient Descent Can Take Exponential Time to Escape Saddle Points

For convex functions, finding minima is easy in polynomial time. For non-convex? This paper shows noise injection is **necessary** for polynomial-time convergence — deterministic GD can take exponential time to escape saddle points.

---

## Near-Linear Time Optimal Transport

Simple algorithm for approximate OT in near-linear time. The discrete OT problem:

$$\min_{\gamma \in \Pi(r, c)} \langle C, \gamma \rangle$$

With entropic penalty (the "matrix scaling problem" [Cuturi, NIPS '13]):
1. Form $A = \exp(-\eta C)$ (elementwise)
2. Alternately rescale rows/columns to match marginals $r$, $c$ → **Sinkhorn**

**Greenkhorn** (greedy variant): rescale one row or column at a time (the one with largest violation). Same convergence guarantee as Sinkhorn, faster in practice.

---

## Implicit Regularization in Matrix Factorization

Matrix completion, linear NNs, multitask learning. Naive approach (impute with 0) gives bad trivial global minima. GD on the factorization $\mathbf{X} = \mathbf{U}\mathbf{V}^\top$ generalizes better — implicitly minimizes the **nuclear norm** $\|\mathbf{X}\|_*$. [Poster 162]

---

## Linear Convergence of Frank-Wolfe over Trace-Norm Balls

Low-rank matrix optimization (e.g., matrix completion). Tradeoff:
- $O(\log 1/\varepsilon)$ iterations with full SVD: $O(d^3)$ per iteration
- $O(1/\varepsilon)$ iterations with rank-1 SVD
- $O(\log 1/\varepsilon)$ iterations with rank-$k$ SVD ✓

---

## Cyclic Coordinate Descent

When is cyclic CD (CCD) faster than random CD (RCD)? CCD can be $O(n^2)$ *slower* than RCD in the worst case, but can also be faster. Paper characterizes the best deterministic cyclic ordering.

---

## Kate Crawford — The Trouble with Bias

Bias in ML $\neq$ bias in the colloquial sense (judgment based on preconceived notions). Much harder to fix with technical approaches alone.

**Two types of harm:**
- *Allocative:* immediate, quantifiable, transactional (loans, insurance, hiring)
- *Representational:* long-term, diffuse, cultural (stereotyping, denigration, under-representation)

Five types of representational harm: stereotyping, misrecognition, denigration, under-representation, and erasure. Technical interventions (accuracy improvements, blacklists, demographic balancing) don't address representational harms — those require theoretical and structural fixes.

**Classification and power:** classification is always a product of its time. Datasets reflect culture and hierarchy — who's in the news, who's not. ML is developing its own culturally-specific classifications. Some things we probably shouldn't build.

---

## Safe and Nested Subgame Solving for Imperfect-Information Games

Libratus beat every human at no-limit Texas Hold'em (4σ significance, 120k hands). Why is imperfect information hard? The optimal strategy for a subgame can't be determined from that subgame alone — you need to consider the full game.

**Safe subgame solving:** estimate the values the opponent receives (not how they play). Unsafe subgame solving exploits the abstraction; safe subgame solving doesn't. **Nested subgame solving** handles large branching factors by generating a unique subgame response to each action at runtime. First AI to beat humans at no-limit poker.

---

## Information-Theoretic Analysis of Generalization

Generalization error = population risk − empirical risk. This can be bounded using mutual information between the training set $S$ and the output hypothesis $W$:

$$\text{gen error} \leq \sqrt{\frac{2\sigma^2}{n} I(W; S)}$$

Holds for unbounded/uncountable spaces. Control by quantizing the output hypothesis, or regularize ERM → Gibbs algorithm. Noise can help regularize. [Russo & Zou 2016 / Xu & Raginsky 2017]

---

## Poincaré Embeddings for Hierarchical Representations

Underlying latent hierarchies are everywhere (Zipf, 1949). Euclidean embeddings need 100–200 dimensions to capture hierarchy; hyperbolic space needs ~5.

The **Poincaré distance** in the unit ball $\mathbb{B}^d$:

$$d(\mathbf{u}, \mathbf{v}) = \text{arcosh}\left(1 + 2\frac{\|\mathbf{u} - \mathbf{v}\|^2}{(1 - \|\mathbf{u}\|^2)(1 - \|\mathbf{v}\|^2)}\right)$$

Center = general concepts, boundary = specific concepts. Simultaneously captures similarity (distance) and hierarchy (norm). Trained with Riemannian SGD on the transitive closure of WordNet. Recovers hierarchical structure that Euclidean embeddings miss entirely.

---

## GAN Theory (Tuesday GAN session)

**Gradient descent GAN optimization is locally stable** [Poster #99]: The GAN objective is *concave-concave* near equilibrium (not convex-concave), so divergence is possible. But: simultaneous infinitesimal gradient descent on generator and discriminator is locally exponentially stable — proved via nonlinear dynamical systems theory (Jacobian eigenvalues). Gradient-norm regularization provably enhances local stability.

**f-GANs** [Poster #100]: GANs as $f$-divergence minimization. The generator minimizes $D_f(P \| Q)$ for a convex $f$; different choices recover KL, reverse-KL, JS, etc. Information-geometric perspective on what kinds of distributions the model can represent.

**Bayesian GAN:** introduce distributions over generator and discriminator parameters; perform posterior inference via stochastic gradient HMC. Completely avoids mode collapse. Achieves SOTA with <1% of labels (semi-supervised).

**Convergence of GAN learning:** if the discriminator has limited capacity (e.g., only linear functions), GAN is doing generalized moment matching. Generated data weakly converges to the true distribution on compact metric spaces.

---

## Off-Policy Evaluation for Slate Recommendation

Goal: evaluate a new recommendation policy $\pi$ using logged data from policy $\mu$. Standard inverse propensity scoring (IPS) requires exact slate matches — extremely unlikely for large slates.

$$\hat{V}(\pi) = \frac{1}{n}\sum_i \frac{\pi(s_i | x_i)}{\mu(s_i | x_i)} r_i$$

Under a **linearity assumption** on the reward, count partial matches instead. The **pseudo-inverse estimator** is tractable and reliable. Goal: automatic policy evaluation without new experiments.

---

## Reinforcement Learning (selected talks)

**Imagination-Augmented Agents (I2A)** [DeepMind]: combine model-based and model-free RL. Agent learns a world model, rolls it out, encodes the rollouts, and combines with a model-free path. Deeper rollouts = better. Tested on Sokoban.

**ACKTR:** scalable trust region method using Kronecker-factored approximate curvature (K-FAC) for natural gradient updates. More efficient than TRPO.

**Uniform-PAC** [Poster #198]: unifies PAC learning and regret minimization for episodic RL. PAC tells you how many mistakes but not severity; regret can't distinguish many small from few large mistakes. Uniform-PAC bounds mistakes for all $\varepsilon$ simultaneously and gives convergence to optimal performance. UBEV algorithm is minimax optimal up to $\sqrt{H}$.

**Inverse Reward Design:** proxy reward $\neq$ true reward. Key idea: rewarded behavior has high true utility *in the training environments*. Be Bayesian about the reward function; be risk-averse in new environments.

**Safe Interruptibility for Multi-Agent RL** [Poster #204]: a safely interruptible agent learns a policy in the interruptible environment that is optimal in the non-interruptible one. Works for single agents; hard with multiple agents unless they can observe the joint action.

**Options and Regret** [Poster #14]: options (temporally extended actions) can be harmful if poorly designed. Regret analysis characterizes how options affect exploration in MDPs.

**EX² — Exploration with Exemplar Models** [Poster #3]: reward novel states by training classifiers to distinguish new states from past experience. States easy to distinguish = novel. Amortized over a replay buffer to avoid per-state classifiers.

---

## Bayesian Deep Learning (Yee Whye Teh)

Bayesian learning: prior $p(\theta)$, likelihood $p(x \mid \theta)$, posterior $p(\theta \mid x) \propto p(x \mid \theta) p(\theta)$. Strengths: normative, explicit inductive biases, unified uncertainty treatment. Weaknesses: wrong model = wrong posterior, scalability.

**Posterior server:** in distributed training, instead of a parameter server (point estimate), workers each construct a tractable posterior approximation and send it to a server that aggregates them. [Hasenclever et al., JMLR 2017]

**Concrete VAEs / Gumbel-Softmax:** VAEs with discrete latent variables require the reparametrization trick for backprop. The **Concrete distribution** is a continuous relaxation of discrete distributions via the Gumbel distribution:

$$z_k = \frac{\exp((\log \alpha_k + g_k)/\tau)}{\sum_j \exp((\log \alpha_j + g_j)/\tau)}, \quad g_k \sim \text{Gumbel}(0,1)$$

As temperature $\tau \to 0$, samples approach one-hot; as $\tau \to \infty$, samples approach uniform. Differentiable → backprop through discrete variables. [Maddison et al. ICLR 2017, Jang et al. ICLR 2017]

**FIVO (Filtered Variational Objectives):** use particle filters as unbiased estimators of the marginal likelihood in sequential models. Tighter bound than standard ELBO for variational RNNs.

Open questions: being Bayesian in the space of *functions* rather than parameters? Handling uncertainty under model misspecification?

---

## Masked Autoregressive Flow (MAF)

Neural density estimation. Autoregressive models with Gaussian conditionals are normalizing flows. **MAF** stacks autoregressive models:

- **MAF:** fast to evaluate $p(x)$, slow to sample → good for density estimation
- **IAF:** slow to evaluate, fast to sample → good as inference network in VAEs
- **Real NVP:** fast for both, but limited capacity

Diagnostic: plot the internal random numbers the estimator uses. If they're not uniform, the density estimator is failing. Fix: explicitly model the density of those internal random numbers.

---

## Deep Sets (Smola)

ML usually uses fixed-dim vectors or ordered sequences. What if inputs are sets (variable size, order doesn't matter)?

**Key theorem:** a function $f : 2^X \to \mathbb{R}$ is permutation invariant iff it can be written as:

$$f(\mathcal{X}) = \rho\left(\sum_{x \in \mathcal{X}} \phi(x)\right)$$

for some $\rho$ and $\phi$. Universal architecture for set functions. Generalizes much better than LSTMs/GRUs on set tasks. Applications: point cloud classification, galaxy redshift estimation, set expansion, outlier detection.

---

## SELU — Self-Normalizing Networks

$$\text{selu}(x) = \lambda \begin{cases} x & x > 0 \\ \alpha e^x - \alpha & x \leq 0 \end{cases}$$

with $\alpha \approx 1.6733$, $\lambda \approx 1.0507$. These constants are the fixed point of the mean/variance map under the activation — proved via Banach fixed-point theorem (contraction mapping). With normalized weights, the network self-normalizes: vanishing and exploding gradients are impossible. Train deep networks without batch norm.

---

## Causality for Interpretability (Bernhard Schölkopf)

Dependence $\neq$ causation. **Reichenbach's common cause principle:** if $X \perp\!\!\!\perp Y$, there exists $Z$ causally influencing both, and $Z$ screens $X$ and $Y$ from each other.

**Functional causal model (Pearl):** $X_j = f_j(\text{Pa}(X_j), \varepsilon_j)$. Causal Markov condition: $X_j \perp\!\!\!\perp \text{NonDesc}(X_j) \mid \text{Pa}(X_j)$.

For two variables, Bayes gives two factorizations:

$$p(a, t) = p(a \mid t)\, p(t) \quad \text{or} \quad p(a, t) = p(t \mid a)\, p(a)$$

Which is causal? The one where the mechanism $p(\text{effect} \mid \text{cause})$ is independent of the marginal $p(\text{cause})$ — the **independence of input and mechanism** principle. Conjecture: the initial state $s$ and system dynamics $M$ are algorithmically independent → thermodynamic arrow of time. [Janzing, Chaves, Schölkopf 2016]

---

## The Hidden Cost of Calibration

A well-calibrated classifier satisfies $P(Y=1 \mid \hat{p}(x) = v) = v$. But group-wise calibration forces different false positive rates (FPR) and false negative rates (FNR) across demographic groups — which is itself a form of unfairness. Both classifiers must be perfect for group calibration and equal error rates to hold simultaneously. **Group calibration is at odds with fairness** (equal FPR/FNR across groups). See [interpretable.ml](http://interpretable.ml/).
