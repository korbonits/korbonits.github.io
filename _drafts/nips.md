





========

Deep learning on manifolds

text, audio, images ---> convolutions are integral to this success

social networks, functional networks, regulatory networks, 3D networks.

what geometric structure is exploited by CNNs?

how to leverage this structure in non-Euclidean domains?

two setups:
1. manifold
2. graphs

domain struct. vs. data on a domain

fixed. vs. diff. domain

known vs. unknown domain (e.g. need to learn a graph)

Duvenaud et al. 2015



- basics of euclidean CNNs
- graoh theory
- fourier analysis of graphs
- spectral domain methods
- spatial domain methods

applications

key props of cnns:

1. convolutional (trans. invariance) -- related to stationarity
2. scale separation (compositionality)
3. filters localized in spave (deformation stability)
4. O(1) parameters per filter (independent of input image size)
5. O(n) complexity per layer)
6. O(logn) layers in classification tasks

CNNs:

translation invariance
multiscale structure
inductive bias

extend to more general geometries by replacing filtering and pooling by appropriate operators.

spatial vs. spectral

====

Non-euclidean


challenges

extend NN techniques to graph/manifold-structured data
assumption: locally stationary data and manifest hierarchical structures :)

Graph theory:
weighted undirected graph, all weights positive

functions over vertices

Hilbert space with inner product

Graph Laplacian! (unnormalized) -- diff between f and its local average

laplacian = degree matrix - weight matrix

dirichlet energy of f: f^T*laplacian of f

measures the smoothness of f

Riemannian manifolds in one minute! (LOLOLOLOLOLOL)

manifold
tangent plane

remannian metric

scalar fields
vector fields (functions of the tangent bundle)
hilbert spaces

laplace-beltrami operator

continuous limit of graph laplacian

still get dirichlet energy :)

orthogonal bases on graphs: find the smoothest orthogonal basis of fcns {phi_1, ..., phi_n} subset of L_2(V)

solution PHI = laplacian eigenvectors

we get eigenvectors and non-negative eigenvalues




Fourier analysis on euclidean spaces

fourier basis = laplacian eigenfunctions

convolutions

(f * g) = integral(-pi, pi) f(x')g(x - x')dx' 

	shift invariance
	conv. theorem: fourier transform diagonalizes the convolution operation => conv. can be computed in the fourier domain as f^ * g^ 

circulant matrix


spectral convolution

instability over deformation :(

=======


Spectral Domain geometric deep learning models

g = PHI* W * PHI_T * f <--- W is what we want to learn

filters are basis-dependent => does not generalize across graphs
O(n) parameters per layer
O(n^2) computatoion of forward and inverse Fourier transforms (no FFT on graphs)
no guarantee of spatial localization of filters


Vanishing moments

localization in space = smoothness in frequency domain

parametrize filter using smooth spectral transfer function tau(lambda)

==> see Bruna, Zaremba et. al 2015

smooth spectral transfer function... try a polynomial

- O(1) params per layer
- filters have guaranteed r-hops support (affect only neighbors)
- no explicit computation of eigenvalues/vectors => O(nr) computational complexity

Defferrand 2016; etc.

spectral graph CNN :)

graph pooling: produce a sequence of coarsened graphs

binary tree arrangement of vertices

 - poor generalization across non-isometric domains unless kernels are localized
 - spectral kernels are isotropiconly (laplacian limitation)
  undirected graphs as symmetry of laplacian is assumed



GNNs (graph neural networks)

spatial constructions that can eat structure dynamically

input is a graph and vector on each node



set: invariant to permutations, dynamic resizing: how can we make a network that does this?

 - mean operation (example)
 - can replace mean with any commutative normalized mapping (attention, geometric mean, etc.)
 - taking the mean (e.g. BOW embeddings) can be very strong baseline (always try first!)


warmup: processing sets

given set of vectors, pick two matrices.

(see slides if possible)

the right setup gets you a plain multilayer NN with transition matrices.

Simple graph NN: mean over neighbors instead of mean over everyone else.  

[kipf and welling 2016] [defferrard et al. 2016]

because spatial implementation, we can take a graph as an input

we can decorate edges with types
we can decorate vertices with types

Interaction networks/ neural message passing

skip connections
gating or multiplicative interactions
W depends on layer...

what does a gnn look like on a grid?

- depends on the grid... :)
- if g is lattice, you can get isotropic filters
- get only radial kernel
- no notion of up and down on a graph

up/down on grid come from group structure
implicit in the ordering of the coordinate representation of the filters

you can get edge filters with decoration



====

spatial domain charting based methods

missing piece: spatial convolution in non-euclidean case.

patch operators... but no shift invariance

geodesic polar coordinates

set of weighting functions

gaussian weighting functions, too

can define convolution as an integral over gaussian mixtures 

MoNet, ACNN, GCNN


mixture model networks on graphs

ChebNet (Chebyshev, makes sense)



===

Applications in Computer graphics and vision

deformable 3d correspondence (intrinsic geometry would help -- invariance will come automatically)

correspondence 1: local feature learning

siamese nets.

correspondence 2: labelling

geodesic distance error

correspondence as classification problem

soft correspondence error

pointwise vs. structured prediction

Correspondence III: FMNet intrinsic structured prediction  (best SOTA yet)


Applications in matrix completion/recommender systems

Netflix challenge

[Kalofolias et al. 2014] geometric matrix completion

use just user graph, or just movie graph

how to learn filters on mutiple graphs?

multi-graph fourier transform


matrix completion with recurrent multi-graph cnn

#138 poster , tuesday

sRGCNN (omg the acronyms)

Apps in particle physics and chemistry






===================

10:40 Tuesday Optimization track

On the optimization landscape of tensor decompositions

3 orals, 7 spotlights

why do we care?

landscape/loss surface of obj. fcns

model/loss --> loss surface --> learning dynamics, training error

overparametrized model means that landscape --> generalization performance [Keskear et al'17, Ding et al/'17']

ReLU, batchnorm, resnet... training faster, better local minimal, design models with easier landscapes

---

Kac-Rice formula counts # of stationary points of a random function

tensor decomp: connects to learning NNs

tensor decomp (see slides)

for a symmetric tensor of rank n, find components a1, ..., a_n (natural extension of eigenvector problem)
\
moment methods for latent variable models

connects to learning of NNs 

random overcomplete tensor decomp

a_i gaussuan, d << n << d^2. (d is dimensionality of tensor e.g., Reals^(d x d x d x d))

maximize polynomial over unit sphere. looking for approximate global maxima

algo Question: can we find global maxima using gradient descent or vairants?
landscape q: any other local maxima?

main results: Q=: expected function value of a random initialization = 3n/d^2

Superlevel set S = {x: f(x) >= 1.1Q}

Therom: with high probability, in the superlevel se s, f has exactly 2n local maxima: +/- a_1, ..., a_n

open Q: Q.

NB: exponential saddle points but SGD helps

Kac-Rice formula (!)  (cf random fields and geometry)

random function makes it hard to pin down stationary points but we can still count with KR formula

random function/field f: R^d --> R

KR fomula counts # of maxima

[Auffinger et al'10] spin-glass model

random matrix theory <-- learn more about this

poss 1: no spurious local maixma

poss3: grad asc goes above 1.1Q quickly 

COnclusions:  Kac-Rice formula + random matrix theory

analyzing other problems?

other math techniques for understandin landscape

-------

Robust optimization for non-convex objectives

goal: train DNN's robust to noise

what we dan do well: train DNNs

we can also train DNNs on distributions over noises

what we want: learning under noise

noisy learning --> optimization problem over loss functions

obtain a robust ensemble NN solution

approx. robust optimization reduces to approx. stochastic optimization

no convexity assumptions (DNN training non-convex)

SGD works well in practice, but we're unlikely to have guarantees

algo: using multiplicative weights update

main theorem: oracle enables robsust optimization <-- look up, also look up regret terms

exampple: multinomial logistic loss

beyond stat learning

conclusions:

1. cast noisy learning problem in terms of optimization
2. construct concrete algorithm to rubost classification
3. reduction: apha robust optimization --> alpha oracle
4. abstracted convexity into an oracle

what's next: adversarial attacks; other applications



============


Bayesian optimization using gradients

using derivatives to speed stuff up!

GP regression with gradients -- naturally acoomodates derivative observation

acquisition trading off exploration/exploitation --> query points for GPR

lots of contributions :)

EI --> one aquisition function... KG another <-- look this stuff up (expectef improvement)

can we do better?

core contribution: novel acquisition function for bayesian optimization

dKG factor --> samples at point that maximizes this acquisition function

infinitesimal perturbation analysis

look up ENVELOPE THEOREM

dKG samples far away from already-sampled points; dEI samples near sampled points

dKG converges to global optimum

using derivatives makes Bayesian Opt faster with the right acquisition function

increasingly practical




=========

GD can take exponential time to escape saddle points

(read this paper)

if convex, easy to find in poly time

nonconvex?

is noise injection necessary for polynomial time convergence? --> this paper says YES! 

=========

near linear times for optimal transport! (woohoo)

simple algo for approx. OT in near-linear time

analyze sinkhorn algo.

converges wrt l1 distance...

GreenkHorn (greedY) gives same gaurantee with faster performance

can cast as linear program in discrete caase

entropic penalty!

"matrix scaling problem"

[Cuturi, NIPS'13]

1. form A = exp(- eta*C)
2. alternately rescale rows/columns tp match r, c

greenkhorm: greddily rescale one row/column to match r/c.

===========

limitations on variance...

finite sum problem



although varence-reduces algorithms are stocastic, they operate over finitely supported distrincutions

Individual Identity is Important 

OBLIVIOUS ALGORITHMS <-- superlinear convergence rates

works for any dimensionality

===========

implicit regularization in matric factorixation

e.g. matrix completion, linear neural networks, multitask learning...

bad trivial global minima don't give us new information (e.g. impute missing entries with 0) took simple least squars problem and made it nonconvex :(

GD on factorization of data generalizes better :) (e.g. svd factorization, do GD on U, not X)

"nuclear norm"

poster 162, uchicago (toyota)

=======

linear convergence of a frank wolfe algo over trace-norm balls

low-rank matrix optimization

ex: matrix completion

# iterations O(log1/epsilon) :) BUT
per uteration full SVF is O(d^3)

O(1/epsilon) .... 1-svd # tradeoff between # of iterations and iteration complexity

O(log1/e) ... k-svd

:hands-up:

==========


acceleration and averaging in stochastic descent dynamics

smooth convex minimization

ineratction of noise and averaging rate (acceleration helps)

mirror descent / continuous time optimization

normal, accelerated, stochastic... 

optimal learning rates!


=============

cyclic coordinate descent

when is ccd faster than random coordinate descent?

cyclic = deterministic order of choice of coordinates

CCD can be O(n^2) slower than RCD

1. can it be faster? yes
2. bounds of improvement
3. characterize best deterministic cyclic order


============

Kate crawford - the trouble with bias

harms of allocation
harms of rep
politics
tackling

1. what is bias, anyway?

this is NOT the same thing as bias in machine learning. judgment based on preconceived notions or predjudices rather than impartiality

much harder to fix with technical approaches

structural inequalities

socio-technical problem

83% of stop-and-frisk were black or hispanic

decades of systemic racial discrimination --> illegal racial profiling

humanities and social sciences have a LOT to offer -- decades of research

2. allocative harms: where we are now (forthcoming paper)

allocation = resources <-- economic view, e.g., mortgages, loans, insurance, etc.

representation = identity

representations of black criminality --> racial sterotype --> prospects in the labor market

( representations of black criminality --> racial sterotype ) <-- problematic outside of labor market context/crim justice

effect of ML in human reps. of identity

Allocation: immediate, readily quantifiable, discrete, transactional
Representation: long term, hard to formalize, diffuse, cultural

3. Representational harms (5 types)
	a. stereotyping (word2vec gender bias paper, google translate swaps genders with gender neutral languages such as Turkish
	b. recognition: respect, dignity, personhood, does the system work for you
	c. denigration: culturally discouraging terms (autocomplete) long history of being used to demean people is part of the issue -- someone needed in the room to note offensiveness
	d. under-representation: lots of white dudes in suits if you search for CEO, first woman is CEO barbie



technical responses - improve accuracy, blacklist, scrub to neutral, demographics or equal representation, awareness

representation harms exceed the scope of technical interventions

how do we represent culture? we need different theoretical fixes --> need to address underlying issues

4. classification -- where do identity representations come from? 

history of classification

	1. classification is always a product of its time
	2. we are in the biggest experiment of classification in human history

ARISTOTLE

reflects social order of the time

religious themes embedded in zoological classification!

enlightenment: taxonomy of the entire universe (john wilkins) 40 categories

empirical vs. linguistic approach to classification

56 genders at facebook
4 years ago, only 2
berkhardt had 11 in 1653 

product of design

basically we're dividing up the world

most representative face: george w. bush (news photographs)

datasets reflect CULTURE AND HIERARCHY (who's powerful)

who's in the news? who's not?

ML developing its own culturally-specific classifications. what if we created an alembert classification for ML?

mental disorders handbook, homosexuality a disorder for 30 years.

dewey decimal system had homosexuality as disorder until 2015.

detecting sexual orientation from faces -- phrenology

representative? cultural features uncontrolled for (e.g. beard)? ETHICS matter, methodology aside

homosexuality still criminal in 78 countries

ML could be deployed and really harm people :(

5. classification and power

criminal prediction based on faces  "free of bias criminality detector" bias encoded

......

end on 3 things: off the top of my head

1. fairness forensics
2. take interdisciplinarity seriously (fairness accountability transparency ehtics FAKE group at MSFT)
	AI now institute --> where disciplines can work together on these problems (comp sci, sosc sci, business, law, engineering)
3. we need to think harder about the ethics of classification

are there some things that we just shouldn't build?

what to do as individuals?

renee camille, 1940's, sabotaged machines used to classify minorities in germany

who is going to benefit
who might be harmed

fairness? that's the question we have to focus on.

=====================

Safe and Nested Subgame Solving for Imperfect-Information Games


imperfect information games (best paper award for poker game :))

Libratus -- texas hold-em

4 sigma significance

every human was beat

why so hard?

because optimal strategy for subgame can't be determined from that subgame alone

find nash equilibtuum

two-player zero-sum games, NE ensures you won't lose in expectation

why care about exploitability?

robust to adversarial adaptation and exploitation by human players

120k hands of poker

coin toss example

p1 flips, p2 guesses

p1 can sell coin or play, EV of subgame = 0.5



you have to consider game as a whole with imperfect information

don't need to care about strategy, just need to know EV

unsafe subgame solving BAD

safe subgame solving GOOD

estimate values opponent receives, not how they're going to play

large spaces--- auctions with large branching factors

bucket -- 201 vs. 200 $ not different, but 200 vs. 250 you lose some info

come up with unique subgame response to action...

nested subgame solving

domain-independent techniques

only ai to beat humans in nolimit poker

makes safe subgame solving effective for the first time

WOW

=======================

A graph-theoretic approach to multitasking

(e.g. driving and talking on your cell phone)

same processors trying to access representations ar the same time

formal analysis: graph theiry 

motivation: what kind of architecture to use with NNs,
formal rel. between avg. degree and signal interference in neural systems


1. TASK (mapping btw vector spaces)
2. inputs
3. outputs
4. neural net

create task graph

graph theory
	- matching
	- induced matching
	- regular graph

interference: sharing input OR sharing output

matching is a necessary condition for multitasking

tradeoff between decay and degree of multitasking

"large degree" == efficient

how to quantify multitasking capacity of parallel architectures?


which graphs have good multitasking properties?

limitations?

multitasking decays with d -->

general case (no regularity assumption) --> capacity should decay when d >> log n

extremal combinatorics

separation when depth > 2

role for discrete math in designing architectures for neural networks

==========================


info_theory analysis of generalization capability of learning algorithms!

population risk (test error), empirical risk(training error)

generalization error = test error - training error

can reflect dependence on training set and output hypothesis -- can be measured with mutual information

upper bound generalization error

holds for uncountable spaces

absolute generalization error is small with high probability

control by quantizing output hypothesis -- or regularize ERM algo --> leads to Gibbs algo.

Noise can help regularize ERM algo.

implications for learning algorithm design... cool!

============================


net-trim: convex pruning of neural networks... with performance guarantee

pre-trained nets

prune and sometimes improve generalization :)

RelU activations

sparsify layer-by-layer

retrain afrter pruning --> almost no discrepancy

Cascade Net-trim improves error

O(s log n) sample complexity as well 

93% zeroed weights (i.e. identical model with 7% weights)


===========================

clustering billions of reads for dna data storage

error rate high :( including insertions, deletions, and substitutions. decode absolutely is :thumbs up:)

clustering is re edit distance. very different than hamming distance.

1 GB --> 1 billion reads

initial strands, cluster centers

ACGT edit distance space

reads = noisy copies

input = noisy copies

hard for large scale k = Omega(n)
O(n^2) time :( :(

new alg!

hashing/embedding scheme for edit distance (wow)

SUPER FAST AND ACCURATE :) #74 poster

===========================

on the complexity of learning neural networks

provable guarantees by GD when learning data generated by small NNs.

what about "natural" NNs?

n gaussian inputs, O~(sqrt(n)) sigmoid units, can you efficiently learn the data generated from this network???

no matter what algorithm you use, you won't succeed in learning this function (statistical query algorithms, at least, that query an oracle)

statistical query SQ algorithms encompasses all known grad desc. variants

need exponential queries to recover function :(

==================

multiplicative weight updates... in congestion games

nash equilibria are fixed points!

if agent follows linear variant then you'll converge to NE :)

exonential MWU can exhibit chaotic dunmaics with n-player congestion games

sensitivity for initial conditions

==========================

Mutual info estimation in mixtures

estimate MI from samples

gene network inference, information bottleneck method, etc.

mix of discrete and cont. distributions --> weirdly you can get MI < 0 even though MI >= 0...

develop algo to estimate MI for discrete-continuous mixtures

(why radon-nikodym derivative required?) what about other simpler derivatives?

===============

missed talk on frame labeling (unsupervised())



==========================

eigen-distortions fo hierarchical representations

how can we quantify the visibility of image distortions

MSE is simplest/commonest -- flawed metric

humans don't make judgment based on pixels, we make it based on rep. of those pixels in our visual system --> find f that is like human's visual system

models of visual physiology

knowledge gets worse with depth

in this talk

- train database to predict human sensitivity to distortions
- use data to optimize filters/weights in physio model

models with 12 and 13 parameters doing as well as neural net with 440k parameters

how well do models generalize?

traditional methods of cross validation are not really what we want for real world. targeted measures for models to make predictions we can test in the real world

===========================


Interpretable and Globally Optimal Prediction for Textual Grounding using Image Concepts

motifation: linking natural language and images Easy for humans

applications in human-computer interaction and robotics

textual grounding a good evaluation task

image i and query phrases q, find bounding box given queries

globally optimal bounding boxes found (!) wow

energy minimization problem

search over all possible bounding boxes... 

"image-concepts" <--- word priors

geometric cues

object detection

semantic sementation

...

interpretability of weights

word-concept relationship --> concetpts come from detection

word-word relationships

===========================

Towards Accurate Binary Convolutional Neural Network

compression

binarized networks , -1 and 1 only --> you can just use logic

wow these talks are short...

========================

deep learning for precipitation nowcasting....



==========================


Poincaré Embeddings for Learning Hierarchical Representations


underlying LATENT HIERARCHY

Zipf, 1949


poincare distance --- simultanuesouly capture similarity and hierarchy in the embedding space from unsupervised data

hyperbolic space

think of it as continuous version of trees

w/ poincare distance fucntion

center is general, edge is specific, similarity is ball around a point; hierarchy is moving from specific to general

riemannian manifold!

riemannian sgd

compute transitive closure of wordnet?

5 dimensions in poincare space

100-200 dimensions needed in euclidean space?

wow. recovers hierarchical information, too




===========


deep hyperspherical learning


phase contains crucial discriminative information (fourier transform of an image) -- magnitude not as much

sphereNet --> network that focuses on angular phase information

spherical phase operators... for convolutions

cosine sphereConv

replaces inner product with hyperspherical convolution

linear, cosine, sigmoid, learnable sphereConv... 

sphereConv insensitive to scaling

use it as new normalization method (sphereNorm), comparable to batch norm.
better than batch norm!

====================


what uncertainties do we need in bayesian deep learning for computer vision?

many different types of uncertainty

epistemic uncertainty - model's lack of knowledge
aleatoric uncertainty - noise in the data

===========

One-Sided Unsupervised Domain Mapping

circular gans?

(pretty cool)

=-==========

Deep Mean-Shift Priors for Image Restoration

ill posed problem

kernel density estimation since p(x) unknown

denoising-autoencoders

compute gradient of prior -- gradient of density estimate -- with denoising autoencoder trained to remove noise

gradient of data term can also be computed

================

Deep Voice 2: Multi-Speaker Neural Text-to-Speech

artificial speech tsynthesis via text-to-speech (TTS)

lots of major challenges

======================

Graph Matching via Multiplicative Update Algorithm

integer quadratic programming problem

NP hard

doubly-stochastic relaxation.... non-negative matrix factorization approach...

====================


Dynamic Routing Between Capsules


CAPSULES

each hidden layer turns into smaller capsules

allows network equivariance

"coincidence detection"

A capsule is a group of neurons whose activity vector represents the instantiation parameters of a specific type of entity such as an object or object part. We use the length of the activity vector to represent the probability that the entity exists and its orientation to represent the instantiation paramters. Active capsules at one level make predictions, via transformation matrices, for the instantiation parameters of higher-level capsules. When multiple predictions agree, a higher level capsule becomes active. We show that a discrimininatively trained, multi-layer capsule system achieves state-of-the-art performance on MNIST and is considerably better than a convolutional net at recognizing highly overlapping digits. To achieve these results we use an iterative routing-by-agreement mechanism: A lower-level capsule prefers to send its output to higher level capsules whose activity vectors have a big scalar product with the prediction coming from the lower-level capsule.


#94
======================

Modulating early visual processing by language

visual QA


===================


the unreasonable effectiveness of structure

Lise Getoor

structure, structure in inputs, structure in outputs

data is multi-modal, multi-relational, spatio-temporal

classic structured prediction problems

"all interesting decisions are structured"

tables: data not independent, doesn't support collective reasoning, not declarative

this talk: patterns, tools, templates for structure

logical rules (for structure)

structured pred. probs:
- entity resolution (when two nodes refer to the same entity)
- link prediction (inferring the existence of edges)
- collective classification (labels of nodes in a graph)

Lise's favorite patterns combine all three :)

statistical relational learning/ai/probabilistic programming

PSL := probabilistic soft logic

psl.linqs.org

unified inference approaches: randomized algorithms (CS), probabilistic graphical models (ML), soft logic (AI)

rounding probabilities --> prob. of rounding up to 1 or down to 0.

3/4 rounding guarantee... 


can also view as template for factor graph....

rules are potential functions (markov random field)

variational inference to help solve (approximation to NP-hard problem)

LOCAL CONSISTENCY RELAXATION -- approximating MAP in MRFs

SOFT LOGIC

Lukasiewics logic

p or q = min(p+q, 1)
p and q = max(p + q - 1, 0)
not p = 1 - p

same optimizations :)

hinge-loss MRFs

scalable map inference

ADMM - alternating direction method of multipliers

FAST :-D

PSL: fast, we can make it faster, we can learn weights, deal with latent variables etc

computational social science

debate stance classification

climate change

social trust models: how do trust links form?

structural balance
social status

latent variable model

how trustworthy is an individual?

knowledge discovery

knowledge graph construction. extractors are noisy. which are the facts you actually want to add???

MLN's - markov logic networks?

knowledge completion via embeddings (work well when you have a lot of data)

PSL helps when data is noisy and sparse

how do we combine structured methods + embeddings?

RESPONSIBLE ML

perils of ignoring structure. privacy, fairness, algorithmic discrimination

encode fairness constraints IN PSL?!?

understanding structure can be key to mitigating effects!



==============


ternGrad : ternery gradients to reduce communication 

typically

1. training data split to n subsets
2. n workers, with a model replica
3. each replica is trained on a data subset
4. synchonisation in parameter server(s)

scalbility:
1. computing time decreases with N
2. communication can be bottleneck
3. this work: quantizing gradient to 3 levels (-1, 0, 1)

stochastic gradients w/out bias

SGD almost truly converges
ternGrad almost-truly converges

stronger gradient bound in TernGrad

closing bound gap

layer-wise ternaring

gradient clipping

integration with manifold optimizers

scaling to large-scale deep learning

ternGrad: randomness and regularization --> decrease randomness in dropout or use smaller weight decay

no new hyperparameters

no accuracy differences with terngrad + gradient clipping

when batch size very large, SGD sucks and gets stuck in local minima, whereas ternGrad noise helps us escape from local minima

performance model: estimate speedup of ternGrad

terngrad gives higher speedup when: 1. using more workers; 2. smaller bandwidth; 3. training DNNs with more fully-connected layers

===========================

train longer, generalie better: how to close the close the generalization gap in large batch training

- model parallelism
- data parallelism

SGD.

Can we increase batch size and improve parallelism?

[Keskar et. al 2017] -- small batches are noisy, which helps

SGD logarithmic with # of iterations in terms of generalization

SGD + momentum + gradient clipping

usually generalize better than adaptive

learning rate sqrt(b) <-- batch size

ghost batch norm, split batch into smaller slices, compute norm

insufficient # of steps with larger batches, fewer weight update iterations

large batches can generalize as good or BETTER than smaller batches if you let them have enough iterations

# iterations fixed for all batch sizes ==> no generalization gap

why weight distances increase logarithmically? [Marinari et al. 1983]

physical property

random walk over random potential

ultra-slow diffusion

distance traversed is logarithmic in time

can we reduce wall-clock time?

why overfitting is good for generalization?

early stopping suggests we should stop training? validation accuracy can keep improving, including for logitic regression

slow convergence to max-margin solution

the implicit bias of gradient descent on ... data !!! 



===================

END TO END DIFFERENTIABLE PROVING

how to prove a theorem in an end-to-end differentiable manner

combining deep and sympbolic reasoning

NNs:
- trained end to end
- strong generalization
- need a ton of data
- not interpretable

first-order logic expert systems
- rules manually defined
- no generalization
- no data needed
- interpretable

how do we get best of both?

nn for proving queries to a knoedge base

proof success differentiable

learn vector representations of symbols

induce interpreatble rules end to end from proof success



fuzzy logic
prob logic programming
inductive logive programming
neural symbolic connectionism

unification

neural unification 

sucess no longer discrete but continuous

embed symbols in vector space and compare similarity using radial basis function kernel

differentiable prover

avoid cycles

we can learn vector representations of unknown predicates :)

induce logical rules using gradient descent

WE CAN LOOK AT THE INDUCED RULES!!!

POSTER #128

========================

GRADIENT DECSNT GAN OPTIMIZATION IS LOCALLY STABLE

when will gan optimization converge to a good solution

intro to gans

how well discriminator tells apart generated from real

find equilibrium point of the game, ie, saddle point of min-max objective

if realizable, does it have good convergence properties???

seems good... has it really converged to gloabl equilibriumn???


is the equilibrium locally exponentially stable?

(minimum requirement -- still an open question -- what they look at in their paper)

proving gan stability is hard

concave at equil.

convave even arbitraility close to equilibrum even for linear generator

NOT THIS: convex0concave.

WHAT WE HAVE: concave concave

divergence very possible :(

pure minimization: train discriminator longer

updating only the generator will cause divergence

update generator and discriminator SIMULTANEOUSLY AND INFINITESIMALLY

hashtag differential equation.

despite concave-concave objective, simultaneous gradient descent gan equilibrium is locally exponentially stable!

noninear dynamical systems theory


jacobian of gan system, if eigenvalues are all negative, then you're good to go???

gradient-norm based regularization

conc.

local stability of gans using nonlinear systems

regularization term provably enhances local stability

poster #99


==========================

f-gans in an information geometric nutshell

game <-- information theory <-- information geometry

convergence in distribution between P and Q with f-Gan model.

what is happening from parametric standpoint? what kinds of distributions can you model?

differential information geometry

#100

======================

unsupervised image 2 image translation networks

summer --> winter (say, for same place)

ill-posed problem
we need additional assumptions

shared latent space

create information bottleneck to couple the mapping function (via weight-sharing)

adversarial training

Coupled gan gramework for unsupervised image2image

(what about for natural language??? )

=========================

the numerics of gans

why are gans hard to train?

rotational geometry of gradient vector field

(good lord lots of jargon)

stabilizes gan training

#101


=======================

dual discriminator GANs

mode-collapsing problem of GAN

3 player game with two discriminators

========================

bayesian gan

mode collapse... gan instability

interventions:

mini-batch discrimination
feature matching
label smoothing

bayesian generalization of the GAN

introduce distributions over the generator and discriminator, and perform posterior inference using stochatic gradient HMC

stochatic gradient HMC
stochatic gradient HMC
stochatic gradient HMC

completely avoids mode collapse issue

can achieve state of the art with less than 1% of the labels! (test using semi-supervised learning... rest of the data unlabled)

========================

approx and convergence properties of GAN learning

what is GAN?

what if discriminator has limited capacity?

cable of only... linear regressions?

when obj converges to its min, does the generated data converge to the true data?

GAN is doing generalized moment matching

yes, generated data weakly converges to the true distribution. compact metric space, and discriminator uses only continuous functions

=-==============

DUALING GANS

instability of alternate gradient updates

wproposed solution: dual form

MAX MAX much more stable

learning is very stable

#103


====-=========

generalizing GANs

model-discriminator co-evolution
- behavioral inference

GANS
- synthesizing photos

turing test 

TURING LEARNING

generalized gans

case study: a robot modeling itself

(discriminator has to be allowed to control movements)


=================

hindsight replay

model-free RL

hierarchical RL

meta RL

=================

ELF for game research 

reinforcement learning: ideal and reality :(

what can we do to accelerate training

extensible lightweight, flexible framework

flexible: many different RL methods
lightweight: good performance, cheap compute
extensible: any c++ game?

actor-critic models in CNN

good perf.

self-training?
===================

imagination agumented agents for deep RL

(deepmind)

intro: model-based RL and imagination

too much trial and error
- still data inefficient :(
- limited generalization

wwant to build agents with internal models of the world
unsupervised learning AND RL
planning capabilities

expensive and brittle (modelbased RL)

hig dimensions, costly and innacurate

combine model-based and model-free

agent learns a model of the world, queries it for information, and learns to interprets the predictions in order to act

encironment model: recurrent model that makes env. prediction at every time step

simuation policy: policy ysed for roloouts

rollout model



monte carlo policy rollout

build imagination agent block by block: rollout encoder

model -based path + model-free path

planner learned by reward maximization and backprop

distillation

I2A is a form of learned policy improvement

tested on Sokoban

deeper rollouts are better 

more imagination is better

active imagination papers

learning to thing --> schmidhuber

representation learning -- env. model only needed for training, not testing.

abstract models in space and time
=========================

ddual path networks

res nets
densely connected networks

resnets are densely connected but with shared connections

good at reusing features

not as good at extracting new features

dense networks have lots of redundancy

dual path arch: benefits from both models

DPN

w/ bottlenecks

=====

a simple neural network model for relational reasoning

visual qa?

relate physical entities

relate non-physical entities (answer reading comp q's)

relate entities that aren't proximal in space or time... very general.

trad. methods are used to things in the scene

relational network 

lots of small pairs of objects and MLP's

CLEVR dataset

========

scalable trust region method for deep RL using kronecker-factored approximation

use natural gradient -- a more efficient optimizer!

trust region policy optimization is efficient and uses natural gradient, but requires expensive updates

ACKTR

K-FAC : scalable approx natural gradient algorithm


efficiency improvements

===========


why attention is all you need

basic building block

RNNs are everywhere

limits parallelism

long range dependencies still hard to learn

attention between encoder and decoder good in NMT

self-attention

makes model expressive

can replace recurrence!!!

the TRANSFORMER

(look at paper for details)

attention is cheap!

thousands of tokens no problem

multi-head attention

better/faster machine translation

free code

Thank you for your ATTENTION

=============

learning combinatorial optimization techniques over graphs

minimum vertex cover

NP-COMPLETE PROBLEM

learn better non-myopic algorithms

structure2vec deep learning for graph representation (RL???)

almost optimal! and faster :)

================

simple and scalable predictive uncertainty quantification

output Y WITH confidence

regression: output mean and variance

what's good?

calibration
higher uncertainty on out of distribution examples

existing bayesian solutions

1l let NN parametrize distrubution but use proper scoring rule as training criterion (log likelihood for example)

2. augment with adversarial training

3. train an ensemble with random initializations
4. combine predictions at test time

ensembles lead to better predictive uncertainty

adverarial training leads to further improvements

predictive entropuy on known and unknown inputs

train: mnist
test: mnist and NOT MNIST (out-of-distribution)

deep ensebles robust

===================

off policy evaluation for slate recommendation

news recommendation -- motivating example

goal: recommmend content that maximizes user engagement

challenge: bandit feedback (no feedback for content not shown) <-- cant' answer counterfactuals

combniatorial contextual bandit

context X --> action s --> reward R (content and rewards stochastic)

policy pi picks context-specific action

V(pi) = E[r(pi(x))]

estimate V(pi)

using logs mu of context, slates, and rewards problem: x from mu, not pi

evaluating slate recommenders is hard!

unbiased apporach: inverse propensity scoring

exact match extremely unlikely to happen

need to collect unreasonably large logs to get decent results

[Wang et al, 2017] <-- need lots of data, can't really do better

linearity assumption

count partial matches instel of modeling phi

pseudo-inverse estimator

very tractable algorithms

pi estimator is a very reliable strategy

goal: automatic policy evaluation

================

robust and efficient transfer learning

(w/ markov decision processes)

variations in the real world are the rule

state, actions vailable to agent, transition model, maps p of arriving to a specific state given an action

S, T, R, pi

learning across related MDPs

randomize your environment

transformed observed mdps into canonical/invariant subspace. (cool)

use latent variable to IDENTIFY mdps and queue policy to pick the right one

hidden parameter MDP

fix hip-mdp by changing transition fcn from bases of GPs to bayesian net (trained with alpha-divergence minimization)


1. initial exploratory episode
2. estimate w_b and refin e the BNN model
3. train a control policy pi_b
4. execute pi_b in supsequent episode

====================

inverse reward design

reward engineering is HARD!

deep RL reduces problem of generating useful behavior to that of designing a good reward function

true reward function
proxy reward function
intended environment

actual environment != intended

THERE BE DRAGONS

proposal: uncertainty plus risk aversion

be bayesian

which part of reward function should be "unspecified"???

reward design for racing

win, high score, low score, placing

test environments: race tracks

******
key idea : rewarded behavior has high true utility IN TTHE TRAINING ENVIRONMENTS
******

reward design process -- how does it work?

 observation: optimizing score leads to winning behavior

 negative side-effects

 inverse reward design -- avoids unknown cells in new environments??? seems unrealistically risk-averse

 challenge: latent rewards

 =========================

 safe interruptibility for multi agent RL

 learned behaviors are hard to predict

 rewards hard to design

 agents lack information due to lack of sensors and designer wanting simplicity

 interruptibility! (supervisor who can change the agent's policy)

 agents may try to avoid interruptions

 NOT TERMINATOR (but more like sher khan snake in jungle book)

 safe interuptibility

 the policy learn in the interruptible environment should  be optimal in the non-interruptible environment

 safe interruptibility should be orthogonal to optimality

 def: interruptions should not prevent exploration (dynamic safe interruptibility)

 an agent using a neural learnign rule can be safely interrupted

 HARD WITH MORE AGENTS

 if agents can observe the joint action, then you're OK

 when they can only observe their own actions then... they can't be safely interrupted

 #204

 ================

 unifying PAC and regret: uniform pac bounds for episodic RL

 performance guarantees useful!

 Uniform-PAC

 missing bridge between existing PAC and regret

 optimality gaps

 PAC bounds instead?

 limitations 

 PAC tells us how many nmistakes but not how severe

 PAC stops improving policy once we reach epsilon

 regret can't distinguish between many small mistakes or few large mistakes

 not directly comparable

 PAC bound ==> highly suboptimal reget bound (but reverse does not hold)

 uniform pac: bounds mistakes for all epsilons simulteously

 convergence to optimal performance ! (pac and regret don't give either)

 UBEV algo: UCB-style algo is great

 minimax up to sqwrt(H) <-- episode length

missing bridge between PAC and regret

 #198




 =========

 INVERSE REINFORCEMENT LEARNING

 human behavior maximizes for the correct reward function (yeah right!)

 ill-posed problem

 degeneracy

 good news: still provides useful tools efor current task

 still: undefinability --> cannot generalize to new tasks

 repeated inverse RL framework

 agent gets demonstrations from multiple tasks

 let experimenter to choose tasks

 let NATURE choose tasks (passive setting) possibly in an adversarial manner

 1. nature chooses a task
 2. learner chooses policy. 
 3. if suboptimal, mistake --> then human demonstrates optimal policy

 goal: make a few mistakes as possible.

 =============

 learning multiple visual domains with residual adapters

 universal visual representations (can we do this for text)

 visual domain decathlon challenge

 better than fine-tuning WOWOWOWOW !!! #39


 ===========

natural value approximators

value function approximation

how to estimate total future reward

intuition: more naturally match the shape of value functions

use past estimates when more reliable than current ones

combine old and new estimates

#6
============

EX^2: exploration with exemplar models for deep RL

gpal: exploration in high-dimensional state spaces

reward very sparse

method: we encourage optimistic exploration by rewarding NOVEL STATES

we determine novelty by classifying states against past experience

intuition: states that are easy to distinguish from past experience ARE novel

training model this way <--> solving density estimation problem

traiing a model for every state we wish to query??? too many

amortized model, so as to avoid training individual classifiers for every state

1. collect experience
2. train exemplar models to distinguish new experiences from old via replay buffer
3. compute ereward bonus
4. update policy
5. repeat until task is solved

poster #3

=========

regret minimization in mdps with options without prior knowledge

OPTIONS -- useful for RL

regret minimization -- important problem

why options?

RL problems often involve solving different subtasks

Taxi problem

when state-action space becomes large, flat RL methods can be inefficient

sparse reward -- hard for algo to exploit efficiently

HIERARCHICAL RL decomposes large problems into smaller ones (like divide and conquer for RL)

options are temporally extended actions (skills) for hierarchical RL

sucessful for atari, minecraft (2017!)

why regret minimization?

main challenge: introducing options can be harmful :(

increasing amount of work on option design/discovery

maximizing learning speed <==> minimizing regret

how do options affect exploration in MDPs?

theoretical analysis

regret analysis leads to option understanding

#14

=========

successor features for transfer in reinforcement learning

exchange of information should take place whenever useful
transfer should be seamlessly integrated within the rl process

goal: general framework for transfer in RL

problem def:

environment is a SET of MDPs (potentially infinite)

each MDP is a task

same structure except for reward function

-- generalized policy imporovement
-- successor features

successor features generalization of successor representation?

(so it's like the features are changing????)

#9

===========

overcoming catastrophic forgetting with incremental moment matching

continual learning <--- fix for catastrophic forgetting

manifold between networks learn from different tasks (in parameter space)

during trainig, use transfer techniques to create this manifold

average parameters of two networks?

find mode of mixture of gaussians of parameters of both networks? of local guassian posteriors...

wtf

drop-transfer


transfer tech. that allow the assumption of gaussian to be reasonable

incremental moment matching as continual learning methods
bayesian perspetcives

drop-transfer as knowledge transfer AND continual learning method


=======


fair clustering

#via fair lets

how to get fair algorithms?

1. design fair algos
implore people to use

3. design fair representations where every algo is fair
apply any algo to data

---


clustering

designate k points to minmize a clustering objective function

annotate data in X . find color-balanced partitions

technical point: solution specified by cluster assignments.

in every balanced cluster, there must be a matching between different-colored points

- first find a matching (these are fairlets)

select one point from each fairlet.

cluster the representatives using traditional methods

assign each point to the cluster of its representative

THEROM: clusters balanced by definition, are approximately optimal

sergeiv@google.com


=========

fitting low-rank tensors in CONSTANT TIME

usually put data into a vector...
matrix for multiple rows...
over ages for people... tensor!

when n=10k, # elements n^# is terascale

decompose with tucker decomposition?

tucker rank - very sensitive hyperparameter

find suitable rank with brute-force search? obvi high rank is lower error but more compression is higher error

measuring error requires cubic time? (just 3 way tensors or all)

given a tensor, take a random subsample

mini tensor --> compute a mini tucker decomp. measure its error, of estimate of original tensor tucker decomp error. 

this preserves error gery well with constant sample size.

for INFINITE tensor, we can find error in constant time (provable gap)

fastest and most accurate method ever!!!!!

#71
=====

LEARNING STATE REPRESENTATIONS

Yael Niv

how do we cross a street?

what environmental features relevant?

time on light
speed/distance of approaching cars

how do we learn a task representation that supports efficient decision making and learning?

brain performs drastic dimensionality reduction, autmatically

most important parts of the task state unobserved?

the same task has different rule in different contexts (jaywalking OK in new york, but not in Washington DC)

learning = generalization

no experiences are exactly alike

bayesian inference with an infinite capacity prior (chinese process prior)

1. observed events causes by latent cassuses
2. prolific latent cause is more likely to cause the next observation
3. number of possible latent causes is unbounded

likelihood: each latent cause tends to produce similar observations

hypothesis: people group experiences based onsimilarity

this determines the bounds od generalization

inference about latent causes determines the boundaries of generalization

real world learning as clustering (with ever-growing set of clusters)

bits of experience impinge on us all the time

we cluster experiences into states

learning hapens within a cluster, not across boundaries

---> this is how humans store events in memory

can we read out these inferred representations from the brain?

challenge for AI: learn to solvve many tasks with little data

the mysterious orbitofrontal cortex

pervasive but subtle role in decision making in the lab

involved in makeing good "life choices" in the real world

most abstract part of reasoning?

hypothesis: Orbitofrontal Cortex codes the current state of the task

especially: inferred states that are not observable (hidden states, or partially observable)

can the current state be decoded from the orbitofrontal cortex?

know true state for each stimulus

train SVM to predict which state each person is in

we can classify this :)

we can classify all the unobservable parts of state, but not the current age state!is this unique to OC?

only area in teh whole brain that encodes all of this information

state representations in OC affect performance

open q: how is this flexible representation created in minutes/seconds???

can we use this understanding of representation learning to our advantage?

can we unlearn fear conditioning in mice?

problem: extinction doesn't work. why?

animals also learn about latent causes

animal may create new state instead of modifying old state.

what if we make extinction more similar to acquisition?

gradual extinction

gradual reverse

no recovery of fear in gradual extinction group!

shallow learning with deep representations

representation learning;prontoparietal network, hippocampus, OC.

reinforcement learning; basal ganglia dopamine

understanding how the brain does this can give us clues

===================

Bayesian deep learning and deep bayesian learning

(pioneering work in bayesian nonparametrics and deep boltzmann machines

yee whye teh <-- SMART DUDE

2017: machne learning/deep learning at peak of hype cycle

fully half of the ideas on teh hype cycle are driven by ML

substance behind hype? yes

copernican revolution

theory-led models

(newton, physics, etc.)

data-led models

(LSTM, wave net, googlenet)

let data speak for itself

ever-increasing flexibility

kernel trick, bayesian nonparametrics, gaussian processes, neural networks

ever-increasing COMLPEXITY

graphical models,  probabilistic programming, tensorflow etc.

interface between bayesian vs. deep learning is coming together

b: inference in some probabilistic model
dl: learing of flexible functions parametrized by NNs

intro to bayesian learning

ideal learner interacting with world

doesn't know state of the world

theta --> state of the world

prior p(theta) <--- marginal distribution over theta

likelihood p(x | theta) <-- process by which observed data is generated, given theta

posterior p(theta | x) <--- totality ofthe agent's knowledge of the world after observing X

predict or act

strengths:
- normative account of best learning given model and data
- explicit expression of all prior knowledge/inductive biases in model
- unified treatment of uncertainties
- common language with statistics and applied sciences

rigidity:
- learning can be wrong if model is wrong
- not all prior knowledge can be encoded as joint distributions
-simple analytic forms for cinditional distributions


scalability

the posterior server

bayesian deep learning

regulatized ml estimator is a posterior mode

MCMC
variational inference

distributed learning (most people think about optimization in this world)

we want a distribution over parameters in bayesian setting. workers sampling from same posterior relaxes need for frequent synchronizations over parameter server

parameter server --> posterior server

each worker constructs tractable approximation of posterior, sends to server

[Hansenclever et al, JMLR 2017]

[Kirkpatrick et al, PNAS 2017] <-- toward AGI: multitask and continual learning

catastropic forgetting :(

elastic weight consolidation

parameters in NNs don't have real meaning

prior and posteriors: it might be better to think in the space of functions instead



the concrete VAEs

how can deep learning ideas improve bayesian learning

use NNs in bayesian models

improve inference and scalability

variational-auto-encoders <-- VAEs

[Kingma and Welling 2014]

DRAW: an RNN for image generation

[Gregor et al. 2015]

reparametrization trick!

VAEs with discrete latent variables

can we do reparametrization trick for discrete latent variables? --> concrete random variables

Gumbel distribution

concrete distribution -- continuous relaxation of discrete distributions. DIFFERENTIABLE

now we can backprop

[Maddison et al ICLR 2017] [jang et al ICLR 2017]

rebar: reinforced concrete

[Tucker et al NIPS 2017]





FIVO: filtered variational objectives

importance weighted auto-encoders

rederivation from importance sampling

fivo: we can use any unbiased estimation of marginal probability... for sequential models, we can use particle filters

variational RNNs

unbiased estimates of conditional probabilities of observations given prior observation

concluding remarks:

brining management of uncertainties into deep learning

bringing flexibility and scalability to bayesian modeling

development opf programming systems

workships

questions to think about:
- being bayesian in the space of functions instead of parameters?
- how to deal with uncertainties under model misspecification?



=============

masked autoregressive flow for density information

neural density information 

unlabeled data -- what's the underlying density?

PixelRNN, WaveNet

exact

but for bayesian inference we can de-noise corrupted images

density estimation for likeihood-free inference

learn posterior from simulations

---

density estimation for monte carlo

proposal distributions

MCMC and sequential MC

--- 

autoregressive models

normalizing flows

designed to be invertible

AR models with guassian conditionals are flows

limitations! (e.g. bimodal conditionals)

masked autoregressive flow

sequence of autoregressive models

masked autoregressive flow inspired by inverse autoregressive flow, another normalizeing flow

different

MARF : fast to calculate, slow to sample -- good for density estimation
IARF: slow to calculate, fast to sample from -- good for use as inference network


real NVP: fast for calculating and sample, but limited capacity as MAF.

MAF (masked autoregressive flow)

conclusions:

1. you can diagnose failure of density estimators looking at internal random numbers the estimators are using, and plotting them
2. we can fix that by EXPLICITLY modeling teh density of the random numbers that we're using internally in the AR models

=======================

deep sets (smola)

motivation:

ML usually uses fixed dim vectors, or ordered sequences

what if inputs are sets?

# elements varies, order does not matter

classification or regression

toy task: find sums of numbers

treating as sequence, permuting doesn't guarantee same result

we want permutation invariance

1. permutation invariant
2. permutation equivariant


given permutation invariant, we can find a function with this property

generalize much better than LSTM or GRU

point cloud classification

redshift estimation in galaxies!

SET EXPANSION (unsupervised task)

concept retrieval #data-team

run LDA and use top words of a tpic to form a set

outperforms all other approaches, even with 3000 sets example (word2vec has billions of words!)

outlier detection

1. we characterized set functions
2. developed universal architecrues based on the results
3. SOTA!

=============================

from bayesian sparsity to Gated Recurrent nets

bridge between multi-loop iterative algorithms <--> multi-scale RNNs

avoid local minima

running example: maximally sparse estimation

what's maximally sparse information?
minimizing a function of 2 terms

second term is a sparsity term (e.g., L_0 term)

applications: solving underdetermined inverse problems

robust regression, outlier removal

NP-Hard

combinatorial # of local minima

---

iterative hard thresholding (IHT)

performance very sensitive to correlation :(

learning a better version of IHT

problem: no possible weights to mitigate strong correlations (w1, w2)

1. use algo that handles correlations well
2. then map to rnn

sparse bayesian learning handles correlations well

consider more complicated RNN architectures

result:

use GRU rnn's.

LSTM can handle SINGLE inner-loop soft-thresholding step

no mechanism for handling multople inner looops

avoid bad local minima

maximize efficiency

revised strategy:

[Koutnik et al 2015] clockwork RNNs

[Chung et al 2015] gated feedback RNNs

gated feedback sparse estimation network

no algo can outperform GF_LSTM


=======================================

turning alchemy of batch norm into electricity

self-normalizing networks

selu(x) = lambda * Piecewise( x if x > 0, alpha....)

READ THIS PAPER

pixed point of two hyperparameters..... NICE. 

(assumed normalized weights)

there exists stable fixed point for unnormalized 

used banach fixed point theirem (contraction mapping theorem)


train networks without external normalization

vanishing and exploding gradients ARE IMPOSSIBLE :)

========================================

batch renormalization

(extension of batch norm)

drawback: takes places differently during training vs. inference

batch REnormalization corrects bias introduced by batch normalization

=======================================

nonlinear random matrix theory for deep learning

why random matrices?

- initial weights are random
  - sure but randomness broken aftera few updates?
  - but randomness remains in overparametrized regimes
- an exact theory of deep learning is likely to be intractable or uninformative 
 - large comlex systems are often well-modeled with random variables
 e.g. statistical physics/thermodynamics

- many important quantities are specific to matrix structure
 - e.g. eigenvectors/values

what kinds of matrices relevant?

activations
hessians
resolvents

can we look at eigenvalues of YY_T or singular values of Y (Y = activation)

method of moments, compute high-dim integrals! look at spectrum...

=======

spherical convolutions and their application in molecular modeling

good for predicting amino acid propensities

predicting change-of-stabilities

==========

translation synchronization via truncated least squares

much better than l1 minimization or using median

==========

self-supervised learning of motion capture

pose estimation
3d mesh

==========

Maximizing Subset Accuracy with Recurrent Neural Networks in Multi-label Classification


Multi-label classification is the task of predicting a set of labels for a given input instance. Classifier chains are a state-of-the-art method for tackling such problems, which essentially converts this problem into a sequential prediction problem, where the labels are first ordered in an arbitrary fashion, and the task is to predict a sequence of binary values for these labels. In this paper, we replace classifier chains with recurrent neural networks, a sequence-to-sequence prediction algorithm which has recently been successfully applied to sequential prediction tasks in many domains. The key advantage of such an approach is that it allows to share parameters across all classifiers in the prediction chain, a key property of multi-target prediction problems. As both, classifier chains and recurrent neural networks depend on a fixed ordering of the labels, which is typically not part of a multi-label problem specification, we also compare different ways of ordering the label set, and give some recommendations on suitable ordering strategies.



===============


The role of causality for interpretability

bernhard scholkopf

dependence vs. causation

common cause principle (Reichenbach)

i. if X and Y are independent, then there exists Z causally influencing both
ii. Z screens X and Y from each other (given Z, and and Y become independent)

FUnctional causal model (Pearl et al)

causal Markov condition - conditions on its parents, X_j is independent of its non-descsendents

what is cause? what is effect? 2 vars

p(a, t) = p(a | t) * p(t) T --> A
        = p(t | a) * p(a) A --> T

via bayes

how to establish?

intervention experiment

hypothetical intervention, if we can think of physical mechanism p(t|a) that is INDEPENDENT of p(a)

maybe p(t|a) is invariant across different countries in a similar climate

factorization

"abstraction challenge for unsupervised learning" -- Y. Bengio

why is modeling p(acoustics) so much worse than modeling p(acoustics|phonemes) p(phonemes)

gedanken

particles scatter at an object

those photons contain information about the object

time evolution is reversible... nevertheless photons only contain info about the object after the scattering?

independence principle shows arrow of time?

s: initial state of a system
M the system dybamics

conjecture: s and M algorithmically independent

thermodynamic arrow of time

paper: [Janzing, Chaves, Scholkopf, 2016]


independence of input and mechanism

causal independence implies anticausal dependence?

transportability <--- barenboim and pearl

Learning Independent Mechanisms

toward learning causal mechanisms

learn general structural causal models (SCMs)

data from multiple tasks

===================

the hidden cost of calibration

Arguably the simplest and most common approach towards classifier interpretability is to ensure that a classifier generates well-calibrated probability estimates. In this talk I consider the (hidden) costs associated with such calibration efforts. I argue that on one hand calibration can be achieved with surprisingly simple means, but on the other hand it brings with it potential limitations in terms of fairness understood as error disparity across population groups.

calibration can be bad for fairness

group-wise calibration -- bad because FPR and FNR differ by groups



-----
http://interpretable.ml/
-----

read papers since I'm missing the last few hours

different FPR and FNR accross different groups, means different mistakes across race or gender or whatever the category is

both classifiers have to be perfect for it not to be a problem of fairness

group calibration is at odds with fairness

