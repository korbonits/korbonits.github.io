---
layout: post
title:  "A Primer on Optimal Transport #NIPS2017"
date:   2017-12-04 08:00:00
---

![NIPS](/assets/nips.png)

## [A Primer on Optimal Transport](https://nips.cc/Conferences/2017/Schedule?showEvent=8736)
### [Marco Cuturi](http://marcocuturi.net/) | [Justin M Solomon](http://people.csail.mit.edu/jsolomon/)

what is optimal transport?

the natural geometry for probability measures

(comparisons of bags of features) -- puts distance on this space

generative models vs. data, or, comparing statistical models p and p'

(see slides)

notion of divergence to measure how well we're doing.

OUTLINE

intro

algos

apps (W as a loss)

apps (W for estimation)

OT and ML workshop on Saturday

optimaltransport.github.io <-- slides and survey

Monge problem: but pile of dir mu into hole nu (in 1781... shovels)

mu(x), height of pile mu.

y = T(x) <-- destination for point x. (mapping of points in mu to points in nu)

D(x, T(x))  <-- distance

mu(x)*D(x,T(x)) <-- work *

T must map red to blue (sum up preimages) # mu(A1) + mu(A2) = nu(B) (T is the pushforward)

T#mu = nu (pushforward notation)

=====
What T s.t. T#mu = nu  minimize WORK (integral of  mu(x)*D(x,T(x)) <-- work *) 
=====


Kantorovich problem: move troops around.

Naive approach results in too many displacements --> find a cheaper alternative

TABLE: sources are rows, destinations are columns (matrix)

distance matrix (distances between sources and destinations) --> what's a transportation matrix?

all soldiers leave, all barracks filled up (supply = demand?) (so this is just a lagrangian?)

look at slides for eq.s

T not does exist for all probability measures.

MEASURE COUPLINGS (bijection)

nothing fancy. just a linear program

minimize the expectation of th e cost over all possible couplings

primal vs. dual --> elegant formulation! (potential functions)




Prop. for "well behaved" cost functions, if mu has a density then an optimal monge map T* between mu and nu must exist

p-Wasserstein distance!

distance on the space of probability measures

OPtimal transport geometry

distributions on a manifold

McCann '95' displacement interpolation

very different geometry than information theoretic metrics (KL divergence etc.)

SOlomon '15'


what's missing? computation

2010: fast OT solvers

Now: use OT as a loss or fidelity term

taking gradient of W distance



2. How to compute OT

discrete/continuous problems:
1. discrete -> discrete
2. discrete -> continuous
3. continuous -> continuous
(what about continuous -> discrete?)

2-3 "up for grabs"

easy cases, zoo of solvers,

univariate case easy :)

compute CDF and quantile functions. 

gaussians easy too :)

CLOSED FORM :-D

T is linear.



====

 - distance between points == Wasserstein distance between dirac masses of those points! 
 - when # of points same on each side, it simplifies to the Monge problem

dual problem is great because it tells you which points are most expensive

O(n^3 * log(n)) # OUCH (min cost flow function is super cubic)

entropic regularization

differentiability of the W distance



solutions P* are unstable (vertices in polytopes from linear programs) and not unique. makes W distance non-differentiable. want: faster, scalable, differntiable


Entropic regulatization (shannon entropy). as gamma increases, you move away from the Monge problem but the fuzziness helps generalize. gamma = infinity ==> coupling of marginals
[Wilson '62']

solve with simple lagrangians -- wow

Sinkhorn's algorithm -- repeat

[Altschuler '17'] <-- nips paper on this algo.

LINEAR CONVERGENCE :-D

O(nm) 
O(nlogn) on gridded spaces with convolutions!

Sinkhorn = Block coordinate ascent on dual

sinkhorn is inbetween W and MMD



a programmer view

[Hashimoto '16] [Bonneel '16] [Shalit '16]

q: how many samples do you need to achieve a certain error epsilon?
a: error decreases VERY SLOWLY (bad sample complexity) --> wasserstein linear program stuff NOT good for data
========================


3. Applications

retrieval
barycenters
unsupervised learnine
inverse problems
learn parameters... generative functions too


photo's as distributions over color space


*** [Kusner '15'] word mover's distance (similarity between documents) *** <-- #data-team (retrieval?)

averaging measures: L2 vs. W average.

barycenter for multiple distributions! (wasserstein barycenter)

averaging histograms is a LP

or primal descent on reg. W --> just to ML instead of hard LP

[Cuturi'14] <-- one of the speakers!


Optimal transport barycenter --> brain imaging

imagine using this for finding smooth document maximums for high scores ---> high gamma to regularize could be helpful #data-team

===

Wasserstein POsterior WASP

aggregate distributed optimization problems with Wass. barycenters. cool.
[Srivastava '15']


Wasserstein propagation [Solomon '14'] (semi-supervised learning problem) ((could we use to label unlabeled documents? or fix label noise?))

could be used for missing data as well.


Dictionary learning --> topic models![Rolet'16]

Let's compute not just means, but also variances?

Wasserstein PCA

negative curvature space... (hmmm investigate)

Generalized Principal Geodesics (!)

=====

Wasserstein inverse problems

dict, new data point, want to write as combination of dict elements, but weights unknown.

application: volume reconstruction
\

distributionally robust optimization [Esvahani'17]

supervised learning
learning with wasserstein ambiguity

(e.g., data != distribution, how do we learn robustly) robust to dataset perturbations (minimax gain style problem)

domain adaptation/transfer learning

1. estimate transport map
2. transport labeled samples to new domain
3. train classifier on transported labeled samples.

======

learning with a W loss

goal is to find mapping f from images --> labels (e.g. photo of dogsled --> caption with (husky, snow, sled, slope, men))

compare two non-normalized histograms

=====

generative models

density fitting

maximum likelihood estimation

just a KL div of your data to a prob dist.

low dimensional latent space, use a push-forward to get to data space. f: latent space --> data space

deconvolutional nets

f#mu <-- BOOM

goal: find theta such that f_theta # mu fits nu_data

[GPM'14] adversarial problem formulation

use wasserstein distances to define.a loss between data and model
