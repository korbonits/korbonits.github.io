---
layout: post
title:  "Torch: bleeding edge DNN research"
date:   2015-06-28 11:49:40
---

## Torch

You can find some background for this post here: [Deep Learning with Python]({% post_url 2015-05-04-Deep-Learning-with-Python %})!

Torch has its strengths and its weaknesses. According to a wonderful write-up by [Tomasz Malisiewicz][Tomasz Malisiewicz] titled [Deep down the rabbit hole: CVPR 2015 and beyond][rabbit-hole]:

> Caffe is much more popular that Torch, but when talking to some power users of Deep Learning (like [+Andrej Karpathy][karpathy] and other DeepMind scientists), a certain group of experts seems to be migrating from Caffe to Torch.

I read somewhere else that Caffe : Torch :: Applications : Research. If you want to do serious research in deep learning, I would suggest using Torch given the level of current interest in the ecosystem, as well as Torch's flexibility and platform. Facebook AI and Google DeepMind use Torch.

The adept reader may be thinking, "wait a second... I thought this was a series of blog posts about doing deep learning with Python... but Torch is all Lua", and yes, you are right, Torch is not a Python tool (though some parts of Torch have Python bindings): Torch, from a user's perspective, is mostly Lua.

For someone well-acquainted with Python, Lua isn't so different. If doing deep learning is more important to you than what language you use, use Torch. If using Python is more important than using all of the potential horsepower available to you at your fingertips, then don't use Torch, but just know that -- as of this blog post -- it's increasingly the research tool of choice.

## Prerequisites

The documentation for Torch is great. They make installation as easy as a few lines of bash! Here's a link to [Torch][torch].

Let's get started. Assuming you don't mind installing torch into `~/torch`, you can just use the following bash commands to get started.

``` bash
$ curl -sk https://raw.githubusercontent.com/torch/ezinstall/master/install-deps | bash
$ git clone https://github.com/torch/distro.git ~/torch --recursive
$ cd ~/torch; ./install.sh
```

Now Torch, LuaJIT, LuaRocks (package manager akin to `pip`), and some packages (installed via LuaRocks) are installed. Type `th` to use the REPL, and in the REPL, you can type `os.exit()` to quit.

At the end you'll be prompted to add Torch to your PATH environment variable. Type 'yes' to complete everything. Just do a quick `source ~/.bashrc` to update your environment.

Head to Torch's [getting started][getting-started] page for more.

Now we're ready to begin. Fasten your seltbelt: some crazy s*** is about to go down.

## Extending Torch to generate Joycean prose

What the... what does that even mean?

I'll show you.

Andrej Karpathy's very detailed and extremely interesting blog post, *[The Unreasonable Effectiveness of Recurrent Neural Networks][unreasonable-rnns]*, goes through several examples that harness code that he very kindly open-sourced [here][char-rnn], to implement a "multi-layer Recurrent Neural Network (RNN, LSTM, and GRU) for training/sampling from character-level language models."

To do that, `git clone` the repo wherever you like. Then:

```bash
$ luarocks install nngraph 
$ luarocks install optim
$ luarocks install cutorch # for GPU use
$ luarocks install cunn    # for GPU use
```

OK. Awesome. Now that we've got that out of the way, let's get some data:

||

![Big Data](/assets/big_data.png)

||

Lulz. Let's skip the big data for now and just start with something small: the full text of my favorite novel, James Joyce's 1922 masterpiece, Ulysses. Full text available [here][ulysses-gutenberg] via one of the files types of your choosing (I chose .txt for this project). In a text editor, I removed the beginning/end of the file what I considered to be unreflective of Joyce, namely, the Project Gutenberg boilerplate :-). My file begins, famously:

> -- I --
>
>Stately, plump Buck Mulligan came from the stairhead, bearing a bowl of
>lather on which a mirror and a razor lay crossed.

And ends, famously:

> yes I said yes I will Yes.
>
> Trieste-Zurich-Paris 1914-1921

Sweet. Now navigate to your `char-rnn` directory, and move your Ulysses text file to `<path>/<to>/char-rnn/data/ulysses/input.txt` (obviously doing `mkdir <path>/<to>/char-rnn/data/ulysses` first).

If you want to look at some settings, you can type `th train.lua -help`, otherwise, let's start training on Ulysses.

```bash
$ th train.lua -data_dir data/ulysses -gpuid -1 # this goes against your CPU
$ th train.lua -data_dir data/ulysses 		# this goes against your GPU
```

If sucessful, you should see some output like this:

```bash
1897/28850 (epoch 3.288), train_loss = 1.76713313, grad/param norm = 9.8094e-02, time/batch = 0.21s
```

Let's sample from a training checkpoint and see what kind of text we generate.

``` bash
## When sampling, be consistent w.r.t. whether or not you trained/are training with your CPU or GPU.
$ th sample.lua cv/some_checkpoint.t7 -gpuid -1 # if you trained against your CPU
$ th sample.lua cv/some_checkpoint.t7 			# if you trained against your GPU
```

Note that there are some markdown-esque characters in Project Gutenburg files, denoting common fomatting styles such as underline, *italics*, etc.

Here's our first sample:

```bash
$ th sample.lua cv/lm_lstm_epoch1.73_1.9188.t7
```

> nned posted, his bind. He are so had. Mishing not eumal dy saye
> gap, Jesmotition, Hefleston foum his isence, Bloom, the peptlemer and callidant or yame of cersainitien. With Redellosy
> Wisco oum for con. Maldrear sailly of exchochened liaty that in throum munders anutetoH icatiped \_Koumban of falles aroumedupacelly)\_ Jal. Noceping fer scontactrents?
> 
> --Comanen, felliits.
>  Shourd comentlaned
> on or whal onverfoul of wappen in that blinking awdactire of like
> a bancaserable with m. Joy, E! I,, dlodnce good thet? Stubre he owald few of cloum. THy and more of
> the
> varss spewing how. What?
> 
> ut the brom in Bock Murigens, what earte up  vore.
> Herrom Goloonhy
> crarks of the time he burth for me fleeterelfs him.
> 
> --Claper I saum. Learked of thit?
> 
> --On a silthing by smolled in Dra0 Comes beard.
> 
> Bliest \_Ceven te moune, Frambly, sears have the druck, turt, some, Manch Cire u'blow. I house I west and yes? I'res
> babladgow. Jneess of combolast and meeye.\_ Maloraga\_.
> 
> You cipet dought
> who
> ca
> sumper herd claused. Lyformselting tumper. Ithere.
> 
> He your after urot!
> Swort up he siblar cappitites. Quains\_ life to a sude her coucting then
> feose, it wattersing thinsarding oot
> of
> Dostle\_)\_ Who one sporial
> sp. Butnen it the sapined by Gulleruust pursan, Muss? Mome Ponain's. Jesoliy, \_10 Excudis amored he's yel. \_Thobe and pricty.
> 
> I movery's to.
> 
> 
>  moned her have coman 4Dakit them man her are to yeard took to Detrarn yound more, Woundackel. And the bgoinalius Parman's bushove bifferly,
> larging toost)\_ Goine of the nothing any suppencede\_ lictedy groveby)\_
> 
> \_(Werrical mentovatubaly alking flames of conson
> is was diys.
> Hat, they'ke of dest jegcises corsay:
> 
> \_Wemstan's naks 107rearminal gruttell and here to gusrouted or shunonil on that the to in temhord beasing hay Lovely, Mn Purninat.
> 
> U telloss aster. Dewained. Setherades)\_ Hishur hand, Drisim, Hell Twander thack
> \_Dousfuar prosy, doneson. Mound deatingsed, that pibst it on melughands and smul I make to enel in the comuty he and butterelan
> and

I recognize some words in there! Wow, this almost looks like Finnegans Wake[^0] and we're just getting started. Or perhaps some form of primordial English/Anglo-Saxon. Leopold Bloom is in there, "Jesmotition" seems like a play on "Jesuit", and it's occasionally formatting italics properly in markdown with two `_` (though it did not close its parentheses properly... yet).

Here's the best sample, after 50 training epochs. Ulysses is about ~1.5mb in size, so fairly small. Smaller than Karpathy's Shakespeare contactenated training/validation file of ~4.4mb. The lowest validation error that was captured was after 17 epochs (after which it started to slightly overfit... the default parameters are small!).

```bash
$ th sample.lua data/ulysses/cv/lm_lstm_epoch17.33_1.5834.t7 -temperature 1
```

> Power, Kinch, an, his dead reformed, for the churches hand.
> 
> They were namebox: a kitchen and perhage his sight on his canes deep any outwas, life
> stands. Clesser. A fellows her last firing. And beneather to him,
> they give me 39: then he was brilliging bying. A lair unde for paper so
> fresh strangy gallous flashing at the crassies and
> thit about a son of their God's kind. His arm.
> 
> She curaces you much interracied that common of yours. Passenear and he toteher. There
> and in I live for near them it spouched hers.
> 
> Becual left, her wall.
> 
> He is Lounata, the curtor, white hoaryses that gave Coimband, looked by
> a hum, he
> wouldn distraction of Drwaphur, the drinken causing out for everybody holy
> gloriamed and stone.
> 
> Died's patther pleaser, tomberful jung bless that on the door and
> grunting for Pletic laudancy, signorian doing to the would. One a hard
> he avaded him explaid, music hazebrakes vala oberous inquisition,
> and ruges grauts with special pupped letters in which      Buck Poile starts were up to them
> upon his great gizzard exchbumminesses:
> the ebit passed pounds. Insaces. Molly, fallonerly, box to intertails.
> 
> Bloom works. Quick! Pollyman. An a lot it was seeming, mide, says, up and the rare borns at
> Leopolters! Cilleynan's face. Childs hell my milk by their
> doubt in thy last, unhall sit attracted with source
>      The door of Kildan
> and the followed their stowabout over that of three constant
> trousantly Vinisis Henry Doysed and let up to a man with hands in surresses afraid quarts to here over
> someware as cup to a whie yellow accept thicks answer to me.
> 
> Hopping, solanist, cheying and they all differend and wears, widewpackquellen
> cumanstress, greets. Chrails her droken looked musicles reading and reason descorning
> for the by Bloomford, swelling a scrarsuit by breed we mouth,
> the past much turned by Borne.
> 
> 
> Makers hear than, Moormar there, the first porter filsions.
> 
> 
> What player well happened the last. A field stones,
> halling shutualar of anylings, Abbo

Wow, that's stunning. "Brillig" makes me think of Jabberwocky, and there are some humorous gems in there. Notice how real sentence structures are starting to take shape and notice the references to multiple characters, including their nicknames. And the various portmonteaus and *jeu de mots*. Remember that this model trains at the character level and at the start of training didn't know a thing about English or anything else relating to the structure of language or prose. Fascinating.

How did we get this?

One of the command-line arguments used for sampling is called *temperature*, whose flag is `-t`. This can be a float between 0 and 1 (must be strictly > 0). To give some intuition, temperature is in a sense the amount of creative license you are giving to your trained RNN while sampling from it. A temperature of 1 allows for the most creative license. This will give perhaps the most interesting results, but said results may not resemble (depending on the size of your network, the size of your data, and how it fits) something that is as readable as something sampled at a lower temperature. Conversely, lower temperature samples are more conservative: sampling at lower temperatures is more likely to result in samples that behave very nicely, yet they can even be *boring*. In some cases extremely low temperatures will cause self-repeating loops. Here is a very sad example where I have set the temperature = 0.1 (sad because Ulysses is so characteristically original and unrepetitive).

```bash
$ th sample.lua data/ulysses/cv/lm_lstm_epoch17.33_1.5834.t7 -temperature 0.1
```

> nears, the street of the
> constinction of the same of the same of the course of the street of the
> constinical constinion of the constituting the sun and the priest of the
> constinions of the best bearded the stage of the same of the same
> bright the state of the stage of the barrels and the barrels and
> the street of the constituting the same of the stage of the same of the
> constinction of the street of the course of the constituting the stairs
> of the course of the street of the barrels and the
> states of the street of the same of the course of the course of the
> bell discussion of the course of the stage of the street of the
> course of the street of the barrels and the street of the stage of the
> constinction of the course of the same of the course of the constituting the
> second and the same of the course of the same of the course of the
> last of the street of the constable of the constituted the stairs of the
> course of the course of the last setter of the course of the same thing
> and the street of the course of the street of the course of the face of the
> constinction of the course of the same of the stage of the street of the
> constinction of the course of the barrels and the steps of the same of the
> constinions of the course of the constituting the sea and the stage of the
> last stranger of the stage of the street of the street and the street of the
> constinction of the course of the street of the same of the state of the
> constinical sure of the course of the same of the course of the stage of the
> constinction of the same breath of the police of the same the same of the
> construction of the same of the course of the course of the last the
> standance of the course of the street of the same of the darkers
> of the constable of the construction of the same of the street of the
> bellaman with the street of the course of the same of the course of the stairs
> of the course of the street of the course of the beauty of the
> construction of the constituting the same states of the course of

Indeed this isn't quite as fun. Notice how conservative our RNN has become w.r.t. sampling new words and structures. It's not perfectly repetitive, but it sure reads like it. Remember, this is being sampled character-by-character.

## What to do next

You may be inspired to mashup your favorite authors. You may be inspired to train an RNN on Finnegans Wake.[^2] *Klikkaklakkaklaskaklopatzklatschabattacreppycrottygraddaghsemmihsammihnouithappluddyappladdypkonpkot*. You may be inspired to mashup texts in more than one language. You may be inspired to train an RNN on code, or perhaps sheet music. Perhaps you may be inspired to train one on audio or video files.[^1]

Try training on a larger corpus. Try increasing the size of your network layers, as well as the number of layers. To paraphrase [Andrew Ng][ng], if your training error is too high, then add more rocket fuel (data), and if your test error is high, then add more rockets (i.e., increase the size of your deep neural network).

In this post, we have only explored a single deep learning model: a recurrent neural network with a long short term memory. Torch is extremely flexible and can be used for (as far as I know) neural networks topologies represented by arbitrary directed acyclic graphs (DAGs) -- though bi-directional RNNs and other valid DNN architectures seem to violate the DAG requirement -- I need to learn more. If you're a graph theory nerd like I am, that is pretty cool. In fact, that should be its own blog post.

Another thing you can do is do a deep dive into the literature! Here is an excellent technical [primer][graves][^3].



[^0]: To be fair, Finnegans Wake is more readable than this, but when I think of Finnegans Wake in the back of my mind, it is almost no different than this first example. Let's see how successive Epochs bring our RNN closer to Ulysses. Next I'll train an RNN on Finnegans Wake itself. Yikes!

[^1]: Note: I am attempting to train on audio files right now and am seeing mixed results, which I think is due to the character-level training that this code is performing, and how that syncs up against very specific file types such as MIDI (especially multi-channel).

[^2]: You are a brave, brave soul. Finnegans Wake itself looks like the output of an RNN with high temperature.

[^3]: Graves, Alex. Supervised sequence labelling with recurrent neural networks. Vol. 385. Heidelberg: Springer, 2012.

[Tomasz Malisiewicz]: https://plus.google.com/+TomaszMalisiewicz/posts
[rabbit-hole]: http://www.computervisionblog.com/2015/06/deep-down-rabbit-hole-cvpr-2015-and.html
[karpathy]: https://plus.google.com/100209651993563042175
[torch]: http://torch.ch
[getting-started]: http://torch.ch/docs/getting-started.html
[unreasonable-rnns]: http://karpathy.github.io/2015/05/21/rnn-effectiveness/
[char-rnn]: https://github.com/karpathy/char-rnn
[ulysses-gutenberg]: https://www.gutenberg.org/files/4300
[ng]: http://www.andrewng.org/
[graves]: http://www.cs.toronto.edu/~graves/preprint.pdf