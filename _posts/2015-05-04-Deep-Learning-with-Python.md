---
layout: post
title:  "Deep Learning with Python"
date:   2015-05-04 11:49:39
---

# Getting started and getting from ideas to insights in minutes

The term "deep learning" is laden with misunderstanding. In this post, I will try to get to the heart of what most folks probably want to know, which isn't: *"give me a thorough theoretical background of the last 50 years of relevant AI"*. It's: *"show me how I can quickly learn a few of the basic concepts and apply them to a problem I have so I can do something cool with it"*. There are couple of reasons for this. 

One reason is that I am not an expert in deep learning. Second, there is a lot of great <del>theoretical</del> empirical[^0] material on deep learning floating around already, particularly on researchers' websites. Where applicable, I will reference the literature if you're interested in diving deep into the theoretical/empirical (and figurative) weeds. Third, getting started is relatively painless: you -- *yes, you!* -- can quite literally get started within minutes.

Sound good?

Here are three basic questions to address:

- What is deep learning?
- <del>Can I use Python?</del> What are some existing ecosystems I can leverage?[^1]
- How do I start?

Let's get started.

# What is deep learning?

Yikes. Big question. Still an open question. At least it's more tractable than "What is the meaning of life?", unless of course you're trying to learn the [**Answer to the Ultimate Question of Life, the Universe, and Everything**][42], which you needn't train a deep neural network for since you already know the [answer][42]. Moving on.

# Can I use Python?

Fortunately, the short answer to this question is yes! Yes, you can use Python to do deep learning!

The long answer to this question is... no different, except to say that you'll need Python to act as the glue between different moving parts which need to access code written in other languages, i.e., I don't know of any end-to-end deep learning solutions that are written in native Python. If you do know of something like this, please reach out to me via email.

In fact, there are a number of different ways to do deep learning via Python. Here are a few that immediately come to mind:

- [Caffe][caffe]
- [Torch][Torch]
- [Graphlab-Create][dato]
- [Theano][Theano]
- [PyLearn][PyLearn]
- [Neon][nervanasys]

I'm sure there are others, too. This list is meant to be representative rather than exhaustive. Let's go through some of the advantages and pitfalls of each.

Going to link to separate blog posts for each way of doing deep learning via Python (note that Torch is an exception to this). Just as there's no good reason for a `.py` file to be 1000 lines long, there's no good reason to let this blog post be that long, either. Anyone who has typed `import this` knows that Readability counts!

### Torch

Here's a link to my blog post, [Torch: bleeding edge DNN research]({% post_url 2015-06-28-Torch-bleeding-edge-DNN-research %})!

# How do I start?

OK, so, now you have at least a sketchy notion of what all this deep learning hype means in theory, and that there are some existing tools that you can use to get started. Well, what are you waiting for? Grab a command line, your favorite text editor, and a WiFi connection. Let's go!

All your data are belong to you. Use Python and deep learning for great justice.

[blas]: http://en.wikipedia.org/wiki/Basic_Linear_Algebra_Subprograms
[homebrew]: http://brew.sh/
[scipy]: http://www.scipy.org/
[sklearn]: http://scikit-learn.org/
[osx-caffe]: http://caffe.berkeleyvision.org/install_osx.html
[lenet]: http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf
[AlexNet]: http://www.cs.toronto.edu/~fritz/absps/imagenet.pdf
[QuocNet]: http://arxiv.org/abs/1112.6209
[caffe]: http://caffe.berkeleyvision.org/
[Theano]: http://www.deeplearning.net/software/theano/
[PyLearn]: http://deeplearning.net/software/pylearn2/
[dato]: http://dato.com/
[capsules]: http://www.cs.toronto.edu/~fritz/absps/transauto6.pdf
[cuda]: https://developer.nvidia.com/cuda-zone
[cuDNN]: https://developer.nvidia.com/cuDNN
[42]: http://en.wikipedia.org/wiki/Phrases_from_The_Hitchhiker%27s_Guide_to_the_Galaxy#Answer_to_the_Ultimate_Question_of_Life.2C_the_Universe.2C_and_Everything_.2842.29
[caffe-installation]: http://caffe.berkeleyvision.org/installation.html#prerequisites
[nervanasys]: https://github.com/NervanaSystems/neon
[Torch]: http://torch.ch/

[^0]: While the literature is certainly academic, it is mostly *not* theoretical work in that the progress in this field is -- admittedly, according to many of those achieving the greatest advances -- at present being driven primarily by empirical successes and failures. There are a lot of prominent researchers in the space calling for more theoretically-driven research.

[^1]: When I started writing this blog post, I thought about focusing solely on Pythonic deep learning tools, but after a lot of exploration, such focus is the wrong approach. There are different interesting problems and approaches in deep learning, with different tools to fit. E.g., a researcher may choose a different set of tools than a practitioner.
