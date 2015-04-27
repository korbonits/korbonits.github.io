---
layout: post
title: Deep Learning in one year
tags: "Deep Learning"
date:   2015-03-07 11:49:38
---

## Have your buzzword and eat it, too

> [It's my [birthday], and I'll [learn] if I want to.][its-my-party]

Today is my birthday, and I would like to begin my next ordinal revolution around the sun *just* right by setting a very ambitious goal for myself:

*One year from today* - i.e., March 7th, 2016 - *I would like to have a working understanding of how to design, train, and implement deep learning systems.*

## What do I mean by "deep learning"?

Let's take a (selected) definition from the main Wiki page on [deep learning][deep-learning]:
	
>Various deep learning architectures such as [deep neural networks][deep-neural-networks], [convolutional deep neural networks][convolutional-nets], and [deep belief networks][deep-belief-network] have been applied to fields like [computer vision][computer-vision], [automatic speech recognition][speech-recognition], [natural language processing][nlp], audio recognition and [bioinformatics][bioinformatics] where they have been shown to produce state-of-the-art results on various tasks.

Using the power of the interwebs, my laptop, and perhaps some [EC2 instances][ec2-instances] (if necessary), I would like to blend the theoretical with the practical, and apply different varieties of deep learning architectures to different varieties of problems over the course of the next year. Luckily, there is a lot of literature out there on applying deep convolutional neural networks to computer vision problems such as [AlexNet][AlexNet] and [ImageNet][ImageNet] (I will delve into some of these problems and architectures in subsequent posts). Given my passion for music and science, I'd like to branch out a bit and apply these methods more generally. Music genre classification would be fantastic! 

Doing unsupervised learning with deep neural nets would also be quite fascinating. I really like the notion that it's possible to learn classification/regression problems without labeled data, because:

1. Unsupervised methods are scalable (i.e., they don't require hand-coded labels by humans).
2. Unsupervised methods are less prone to human mislabeling/misclassification.
3. Unsupervised methods can tell us things we don't know about the structure of the data, the representations of the data, or the underlying phenomenae of the data. [QuocNet][QuocNet] is a very well-known example of this.

Last, I'm not just interested in the theoretical underpinnings, though anyone who knows me well knows that I tend to gravitate to wherever there's more mathematics: I am interested in the engineering! I am fascinated by the use of GPUs for doing deep learning. Look forward to some posts about tensors and embarassingly parallel computational methods :-). E.g., here is the canonical form in which the [Ricci flow][ricci-flow] is written: 

$$\partial_t g_{ij}=-2 R_{ij}$$

Wherever possible, I will try to cite my sources and include both math and code, whether inline or as links to math or code living elsewhere (e.g., in a gitHub repo).

{% highlight python linenos %}

## Example from sci-kit learn: http://bit.ly/1GhbMbd
import numpy as np
from sklearn.neural_network import BernoulliRBM
X = np.array([[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 1]])
model = BernoulliRBM(n_components=2)
model.fit(X)
BernoulliRBM(batch_size=10, learning_rate=0.1, n_components=2, n_iter=10,
       random_state=None, verbose=0)

{% endhighlight %}

[its-my-party]: https://youtu.be/XsYJyVEUaC4
[deep-learning]: http://en.wikipedia.org/wiki/Deep_learning
[deep-neural-networks]: http://en.wikipedia.org/wiki/Deep_learning#Deep_neural_networks
[convolutional-nets]: http://en.wikipedia.org/wiki/Convolutional_neural_network
[deep-belief-network]: http://en.wikipedia.org/wiki/Deep_belief_network
[computer-vision]: http://en.wikipedia.org/wiki/Computer_vision
[speech-recognition]: http://en.wikipedia.org/wiki/Speech_recognition
[nlp]: http://en.wikipedia.org/wiki/Natural_language_processing
[bioinformatics]: http://en.wikipedia.org/wiki/Bioinformatics
[ec2-instances]: http://aws.amazon.com/ec2/
[AlexNet]: http://www.cs.toronto.edu/~fritz/absps/imagenet.pdf
[ImageNet]: http://www.image-net.org/
[QuocNet]: http://arxiv.org/abs/1112.6209
[ricci-flow]: http://en.wikipedia.org/wiki/Ricci_flow