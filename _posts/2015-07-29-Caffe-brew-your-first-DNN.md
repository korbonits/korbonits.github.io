---
layout: post
title:  "Caffe: Brew your first DNN"
date:   2015-07-29 11:49:40
---

## Caffe

You can find some background for this post here: [Deep Learning with Python]({% post_url 2015-05-04-Deep-Learning-with-Python %})!

Caffe has its strengths and its weaknesses. For example, there are some outstanding issues regarding using multiple GPUs in parallel during training. According to a wonderful write-up by [Tomasz Malisiewicz][Tomasz Malisiewicz] titled [Deep down the rabbit hole: CVPR 2015 and beyond][rabbit-hole]:

> Caffe is much more popular that Torch, but when talking to some power users of Deep Learning (like [+Andrej Karpathy][karpathy] and other DeepMind scientists), a certain group of experts seems to be migrating from Caffe to Torch.

I read somewhere else that Caffe : Torch :: Applications : Research. If you want to quickly iterate on datasets with the aim of building applications, Caffe gives you a flexible framework with a lot of built-in tools to do so; with Python bindings to boot. Additionally, one of its great features is that you can essentially specify all of the layers and layer parameters of a neural network with a simple config file. I know that there are some out there (you know who you are! :-P) who do not like this feature.

## Prerequisites

Caffe has some prerequisites, which, unless you've already got a CUDA driver installed, will prevent you from getting started in just minutes.

Please go to the Caffe [installation page][caffe-installation] for more details. Be sure to also follow the [OS X Installation][osx-caffe] page very closely if you're on a Mac like I am.

For going against your GPU, you'll need:

- [Homebrew][homebrew] if you don't already have it. 
- [CUDA][cuda] (if you want to use CAFFE in GPU mode, which in itself requires an NVIDIA GPU and an NVIDIA Developer login)
- [cuDNN][cuDNN] (accelerated CUDA, in a nutshell)
- `brew install boost-python` (simply doing `brew install boost` or `brew install boost --with-python` didn't do the trick)

There are plenty of instructions on the caffe site for getting these prerequisites installed. 

Since we're using Python, pay **EXTRA SPECIAL ATTENTION** to the Makefile.config instructions. Particularly, know where your Python and Numpy live.

Mine's a little complicated since I use Homebrew for lots of things, but here's what my config info looks like. This will save you from pulling some hair out:

Add these to your `~/.bash_profile` or just `export` them in your session.

{% highlight bash %}
export PATH=/usr/local/cuda/bin:$PATH
export DYLD_LIBRARY_PATH=/usr/local/cuda/lib:$DYLD_LIBRARY_PATH
{% endhighlight %}

Cool. Now, in your `Makefile.config` file:

```
PYTHON_INCLUDE := /usr/local/Cellar/python/2.7.9/Frameworks/Python.framework/Versions/2.7/include/python2.7 \
		/usr/local/Cellar/numpy/1.9.2/lib/python2.7/site-packages/numpy/core/include

PYTHON_LIB := /usr/local/lib /usr/local/Cellar/python/2.7.9/Frameworks/Python.framework/Versions/2.7/lib
```

Moving on!
In your `<path>/<to>/caffe/` directory:

{% highlight bash %}
$ make all
$ make test
$ make runtest
...
$ make pycaffe
{% endhighlight %}

When all of this is done, start Python in `<path>/<to>/caffe/python/`.

{% highlight python %}
>>> import caffe
{% endhighlight %}

Awesome! Finally, we can let the real fun begin.

## Model Zoo

Caffe, having the great ecosystem that it does, has a special place called the ["Model Zoo"][model-zoo] where various reference models and variations thereof are curated in one place with code and citations. Be sure to check it out for Caffe implementations of some of the most recent cutting-edge research, such as GoogLeNet and models from the CVPR2015 DeepVision workshop, which occurred after this blog post began.

## Deep Dreaming

If you're interested in some light background reading regarding GoogLeNet, the deep learning model that Google Deep Dream uses as its default, you should check out the following arXiv preprint: [Going Deeper with Convolutions][GoogLeNet][^0]. Note that GoogLeNet is an homage to the pioneering work [LeNet](http://yann.lecun.com/exdb/lenet/)[^1], built by Yann LeCun in the 80's for handwritten digit recognition.

In fact, [here][caffe-lenet-tutorial] is a great Caffe tutorial building LeNet by hand and training it on MNIST. But I thought you'd find Google Deep Dream more interesting as a blog post :-).

In [this][deep-dream-blog1] Google Research blog post, Google describes its notion of Inceptionism, and how it visualizes going deeper into neural networks. A couple of weeks later, they published [this][deep-dream-blog2] follow-up blog post, which links you to the DeepDream [repo][deep-dream-repo], conveniently hosted on GitHub and available as an IPython Notebook [here][deep-dream-ipython].

## Deep Dreaming Prerequisites

It's simple enough to follow the IPython Notebook, but here are some instructions:

 - Make sure you have installed the following Python libraries: NumPy, SciPy, PIL, and IPython (just use `pip`).
 - Make sure you have installed Caffe (i.e., that you have read the beginning of this blog post and didn't skip ahead!).
 - Google's [protobuf][protobuf] library (I will go through this since it was a little tricky).

## Installing Protobuf

If you're a nerd like me, you're insane enough to go straight to the source: Google's protobuf GitHub repo, found [here][protobuf-repo].

The installation instructions are fairly simple. Unless you are using a Mac. You can follow their instructions using MacPorts, or you can join me in 2015 and use Homebrew :-) (or the package manager of your choice).

Here's the flow for a Mac user:

{% highlight bash %}
$ brew install autoconf
$ brew install automake
$ brew install libtool
{% endhighlight %}

Nice. Now you can install protobuf. Go to a directory where you'd like to put the protobuf repo.

Then do the following:

{% highlight bash %}
$ git clone https://github.com/google/protobuf.git
$ cd protobuf/
$ ./autogen.sh
$ ./configure
$ make
$ make check
$ make install
{% endhighlight %}

Assuming that works, you're ready to deep dream!

Go to a directory where you'd like to put the Deep Dream IPython notebook and `git clone` it!

{% highlight bash %}
$ git clone https://github.com/google/deepdream.git
$ cd deepdream/
$ ipython notebook
{% endhighlight %}

Click on `dream.ipynb`, et voilà, you're in.

Suffice to say, I won't copy the whole notebook here. That being said, let's take a closer look.

Did you run into an error when you tried to load in GoogLeNet? I did. Just because you've downloaded and installed Caffe doesn't mean that you're ready to brew!

Go to `<path>/<to>/caffe/`, and do the following:

{% highlight bash %}
$ ./scripts/download_model_binary.py models/bvlc_googlenet
{% endhighlight %}

Once that has finished, you should be able to run the block of `dream.ipynb` that loads in a pretrained GoogLeNet. Make sure you set your `<path>/<to>/caffe/models/bvlc_googlenet` properly. Now you're ready to deep dream.

Get down a few lines to where you assign:

{% highlight python %}
img = np.float32(PIL.Image.open("your_own_image.jpg"))
{% endhighlight %}

## Seurat 

I'm picking my favorite work by pointillist George Seurat, his epic chef d'oeuvre, *Un dimanche après-midi à l'Île de la Grande Jatte*.

You don't need to be an art history buff to appreciate this work. Perhaps you're a fan of [Ferris Bueller's Day Off][ferris-bueller]? Who can forget the moment when Cameron sees this work from afar and becomes fixated on its perfection? Here's a clip from their visit to the Art Institute of Chicago (click [here](https://www.youtube.com/watch?v=ubpRcZNJAnE) if you want to see the whole thing):

||

[![Cameron staring at Seurat](/assets/cameron-staring.png)](https://youtu.be/ubpRcZNJAnE?t=1m23s)

||

||

Et Voilà:

![Un dimanche après-midi à l'Île de la Grande Jatte](/assets/seurat.jpg)

||

## Prepare for your mind to be blown

Take a deep breath. This is it. This is the moment. It's so intense.

{% highlight python %}
_=deepdream(net, img)
{% endhighlight %}

||

![Un deep dream dimanche après-midi à l'Île de la Grande Jatte](/assets/seurat_dream.jpeg)

||

BOOM. Holy mackerel. What are all of these strange animals and pagodas and faces popping in and out of mes amis Parisiens?

Let's take a look at what else we can do:

{% highlight python %}
_=deepdream(net,
            img, 
            octave_n=12)
{% endhighlight %}

||

![Un deep dream dimanche après-midi à l'Île de la Grande Jatte](/assets/seurat_dream_12octaves.jpeg)

||

OK, WOW, THIS IS DEFINITELY WEIRDER.

The `octave_n` parameter default is 4. When you run `deepdream(net,img)` for the first time, you will see an output that ranges from `0 0` to `3 9`. The image is redrawn `iter_n` times per octave, so `3 9` here indicates (octave) 4 (interation) 10, since we're counting from 0. Interesting. In music theory, octaves abstractly represent notes whose frequencies are double the previous octave, i.e., you have sung *do re me fa so la ti do* and both *do* notes are exactly one octave apart (and whose frequency ratio between the first and second *do* is 1:2). Here in the notebook you can follow the code to see that the definition of the octave is subject to your own experimentation!

Let's do one more:

{% highlight python %}
_=deepdream(net, 
            img, 
            octave_n=13, 
            iter_n=13, 
            octave_scale=1.5, 
            end='inception_3a/output')
{% endhighlight %}

What is the `end='inception_3a/output'` parameter? Check it out: submit `net.blobs.keys()` and pick a layer, any layer (except the splits, I think).

Here's what we get:

||

![Un deep dream dimanche après-midi à l'Île de la Grande Jatte](/assets/seurat_dream_13o_13i_3a_output.jpeg)

||

How cool is that? Does it seem at all Kandinsky-esque, or is it just me?

## What to do next

You've got the power of Deep Dreaming in your hands. What do you want to do? There are a couple of interesting helper functions in the notebook that I am intentionally not covering so that you can explore them yourself!

One loops over and over, so that the final output of calling `deepdream(net, img)` is the input -- i.e., `img = deepdream(net, img)` -- to another iteration of `deepdream(net, img)`. This leads to a lot of interesting compositional weirdness. Definitely what I think of as *inception*: dreams within dreams.

||

![We need to go deeper](/assets/inception.jpg)

||

The second gives you the ability to use another image as the *objective guide* of your deep dreaming. Put another way, you can kind of think of this helper function as mashing up your deep dreaming with one image that tries to align with the other.

Last, if you're feeling particularly full of free time, try building a tree of such deep dreaming. Maybe you want to train an image on a hierarchy of different images, each one dreamed from a layer below. Moreover, don't just follow my example and deep dream with oil on canvas. Try using natural images instead. Depending on your parameters, your deep dream results could be very far out indeed.

**WARNING:** Deep dreaming photos of people whom you love is definitely nighmare-inducing. Don't say I didn't warn you.

[^0]: Szegedy, Christian, Wei Liu, Yangqing Jia, Pierre Sermanet, Scott Reed, Dragomir Anguelov, Dumitru Erhan, Vincent Vanhoucke, and Andrew Rabinovich. "Going deeper with convolutions." arXiv preprint arXiv:1409.4842 (2014).

[^1]: Y. LeCun, B. Boser, J. S. Denker, D. Henderson, R. E. Howard, W. Hubbard, and L. D. Jackel. Backpropagation applied to handwritten zip code recognition. Neural Comput., 1(4):541–551, December 1989.

[blas]: http://en.wikipedia.org/wiki/Basic_Linear_Algebra_Subprograms
[homebrew]: http://brew.sh/
[osx-caffe]: http://caffe.berkeleyvision.org/install_osx.html
[caffe]: http://caffe.berkeleyvision.org/
[cuda]: https://developer.nvidia.com/cuda-zone
[cuDNN]: https://developer.nvidia.com/cuDNN
[caffe-installation]: http://caffe.berkeleyvision.org/installation.html#prerequisites
[model-zoo]: https://github.com/BVLC/caffe/wiki/Model-Zoo
[Tomasz Malisiewicz]: https://plus.google.com/+TomaszMalisiewicz/posts
[rabbit-hole]: http://www.computervisionblog.com/2015/06/deep-down-rabbit-hole-cvpr-2015-and.html
[karpathy]: https://plus.google.com/100209651993563042175
[deep-dream-blog1]: http://googleresearch.blogspot.com/2015/06/inceptionism-going-deeper-into-neural.html
[deep-dream-blog2]: http://googleresearch.blogspot.com/2015/07/deepdream-code-example-for-visualizing.html
[deep-dream-repo]: https://github.com/google/deepdream
[deep-dream-ipython]: https://github.com/google/deepdream/blob/master/dream.ipynb
[protobuf]: https://developers.google.com/protocol-buffers/
[protobuf-repo]: https://github.com/google/protobuf
[GoogLeNet]: http://arxiv.org/abs/1409.4842
[caffe-lenet-tutorial]: http://caffe.berkeleyvision.org/gathered/examples/mnist.html
[ferris-bueller]: http://www.imdb.com/title/tt0091042/