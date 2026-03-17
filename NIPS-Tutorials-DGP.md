---
layout: post
title:  "Deep Probabilistic Modelling with Gaussian Processes #NIPS2017"
date:   2017-12-04 12:00:00
---

![NIPS](/assets/nips.png)

## Deep Probabilistic Modelling with Gaussian Processes
### [Neil D. Lawrence](http://inverseprobability.com/)

ML? data + models -> prediction

ML is a mainstay of AI because of the importance of prediction. But preds not enough, need to make decisions.

to combine data + model:
1. prediction function (!= inference)
2. objective function

UNCERTAINTY

- scarcity of training data
- mismatch of pred functions
- uncertainties in objective/cost function


neural networks are gr8!

f(x) = (w^2)^T * phi(W_1, x), phi nonlinear

in statistics, it's linear in the parameters, but non-linear in the inputs. 

adaptive basis functions.

activation functions are basis functions.

W1 (weights) are static parameters

stats: interpretability > prediction

In ML we optimize W1 and W2.

This tutorial (MacKay, 1992) (Neal, 1994) theses -- follow their path. Probabilistic approach.

p(y* | y, X, x*)

JOINT MODEL FO THE WORLD --> we compute posterior distributions (bayesian neural networks)

p(y | x, W) is the likelihood of a data point

(normally we assume independence)

iid assumption is about NOISE, not the underlying function.

commonly a gaussian likelihood, do MLE for supervised

considering priors over latents, you can do unsupervised learning

probabilistic inference
y <-- data
p(y*, y) <- model
p(y*|y) <- prediction

GRAPHICAL MODELS:

- represent joint distribution through _conditional dependencies_

e.g., markov chain

performing inferene:
- easy to write in probabilities
- wealth of computational challenges
- high dimensional integrals require lots of computation


linear models:

HOLD W1 as fixed for given analysis.

Gaussian prior for W

sum of gaussians is gaussian

scaling a gaussian --> gaussian

hashtag linear algebra

p(y) = integral (P(x) * P(y|x))dydx

design matrix of activations of data points and hidden units

y is distributed with 0 mean and a covariance (see slides)

neural net is already a type of gaussian process (a degenerate one...)

joint gaussian density: only a certain amount of maths that works. kernels are dangerous.

====

(Ioffe and Szegedy, 2015) batch normalization, gaussian process maths

basis function can be a deep net if you like

Non-degenerate gaussian process

- the process is degenerate
- rank is a most H of the kernel function
- as n--> infinity, cov matrix is not full rank
- |K| = 0

model can't respond to the data as it comes in (cuz its parametric, duh)

"what if you took h --> infinity?" <-- radford's thesis (page 37, big deal!)

sample infinitely many hidden units in kernel fcn instead... prior doesn't need to be gaussian either.

scale output variance down as you increase sample size

obj. in bayesian inference: sample from model for what the world looks like.


data... throw away all functions that don't go through the data (ABC)

dist. over functions

if i've observed f1, what's f2?

key object = covariance function (kernel fcn) K

it's a distribution and a fcn of X

k_ij = k(x_i, x_j)

#linear algebra

posterior mean/covarance functions of the kcovariance

GP --> analytically compute mean/variance at all points

Exponentiated Quadratic Covariance

Alan Turing would've done well in the 1948 marathon(!)

why the hell would we want neural networks???

infinite smoothness not always a good idea...


brownian motion is a gaussian process... stochastic 

CMWB is a GP !

at some point the universe was a GP and it is no longer

---> pushing around too many nonlinearities does this :)


Full Gaussian fitting

in practice, we'll do a sparse gaussian process --> use low-rank approximation of full covariance matrix

SPARS GAUSSIAN PROCESSES
inducing variable fit

cubic complexity (quadratic storage, cubic for matrix inversion)



=====

Deep neural networks

in GP we talked about making hidden layer --> infinity

matrix between two 1000-unit layers is 1 million parameters. overfitting! dropout...

try parametrizing W with its SVD: Bottleneck Layers

stacks of neural networks

if you want to get rid of NN params... replace each network with a gaussian process. integrate all of them out. you get a vector-valued GP.

Take each layer to infinity units. Bottlenecks important in deep GP.

composite stochastic process...

g(x) = f5(f4(f3(f2(f1(x)))) --> composite multivariate function

equiv to markov chain assuming markov condition

p(y|x) = p(y|f5)p(f5|f4)p(f4|f3)...

why deep?

gps give priors over fcns

derivatives of a gp are a gp (if they exist)

for some covariance fcns they're universal approximators.

gaussian derivatives might ring alarm bells! (lots of fcns don't have gaussian derivatives... e.g. a jump function or heavy tail fcn)

you have to ENCODE jumps

process composition.

just graphical models with GPs... (lots of conditional independence assumptions)

difficulty -->

- propagating probabliity dist. through a nonlinearity
- normalizatioin of distribution becomes intractable

check out density networks



======

Deep GPs

- deep ==> abstraction of features
- use variational approach to stack GP models

derivative distribution becomes heavy-tailed as depth increases Duvenaud et al. 2014

How deep are deep GPs? https://scirate.com/arxiv/1711.11280

deals with stochasticity quite well

heteroskedacicity with olympic marathon running times (length scales changing)

shared LVM (latent variable model)

