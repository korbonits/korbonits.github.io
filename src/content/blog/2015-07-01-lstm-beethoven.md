---
title: "Teaching an LSTM to Play Beethoven"
date: 2015-07-01
description: "In June 2015, I trained Karpathy's char-rnn on a corpus of Beethoven piano sonata MIDI files. Here's what it sounds like — and how I got it to play."
tags: ["deep-learning", "lstm", "music", "torch", "generative"]
draft: false
---

In my [Torch post](/blog/2015-06-28-torch-bleeding-edge-dnn-research), I left a footnote:

> *Note: I am attempting to train on audio files right now and am seeing mixed results, which I think is due to the character-level training that this code is performing, and how that syncs up against very specific file types such as MIDI (especially multi-channel).*

This is that experiment, written up nine years late.

## The Idea

Karpathy's [char-rnn](https://github.com/karpathy/char-rnn) treats any file as a sequence of characters and trains an LSTM to predict the next one. It works on text. It works on code. The question was: does it work on MIDI?

MIDI files are binary, but they're structured binary — a sequence of bytes encoding note events, timing, velocity, and channel information. If you squint, it's not so different from a sequence of characters. The LSTM doesn't know what a note is. It just sees bytes and tries to learn the patterns.

I found a corpus of Beethoven piano sonata MIDI files and pointed char-rnn at them. Same architecture as the Ulysses experiment: a multi-layer LSTM ([LSTM.lua](https://github.com/karpathy/char-rnn/blob/master/model/LSTM.lua)), trained character-by-character (byte-by-byte) on the raw file contents. I was running this on a 2015 MacBook Pro, maxing out the NVIDIA GPU — probably 2–3 layers with 128 hidden units, the largest I could fit.

## The MIDI Prefix Trick

Here's the part that took some figuring out: raw LSTM output is just bytes. To play it in a MIDI player, those bytes need to look like a valid MIDI file. MIDI files have a fixed header structure — a common prefix that tells the player the file format, number of tracks, and timing resolution.

The solution: I opened the Beethoven MIDI files as text and inspected them. Every file in the training corpus shared a common prefix near the beginning — the standard MIDI header bytes. I extracted that prefix and prepended it to the LSTM's generated output before saving as a `.mid` file. The MIDI player then had enough structural context to interpret the rest of the bytes as music — or at least attempt to.

It worked. Imperfectly, but it worked.

## What It Sounds Like

I converted the generated MIDI files to MP3 for easier listening. These were generated in late June / early July 2015.

**Sample 1** (June 28, 2015 — early training):

<audio controls>
  <source src="/assets/lstm-beethoven-1.mp3" type="audio/mpeg">
</audio>

**Sample 2** (June 29, 2015 — same checkpoint, different seed):

<audio controls>
  <source src="/assets/lstm-beethoven-2.mp3" type="audio/mpeg">
</audio>

**Generated MIDI output** (converted to MP3):

<audio controls>
  <source src="/assets/lstm-beethoven-generated.mp3" type="audio/mpeg">
</audio>

It's not Beethoven. It's not really music in any conventional sense. But there's something in there — a vague sense of phrase structure, occasional melodic fragments that feel almost intentional, and a general character that's distinctly piano-ish. The LSTM learned *something* about the byte-level structure of these files.

## Why This Was Early

Google's [Magenta project](https://magenta.tensorflow.org/) launched in mid-2016 with a proper approach to music generation: representing music as a sequence of note events rather than raw bytes, using models that understand musical structure. That's the right way to do it.

This was the wrong way to do it — and it still kind of worked, which is the interesting part. The char-rnn approach is maximally naive: no music theory, no note representation, no understanding of MIDI structure beyond what it could infer from the byte patterns in the training data. The MIDI prefix trick is a hack. And yet the output is recognizably piano-like.

The lesson I took from it: the unreasonable effectiveness of RNNs extends further than you'd expect. Even when the representation is wrong, the model finds something to learn.

## What I'd Do Differently

Use a proper event-based representation — encode notes as (pitch, duration, velocity) tuples rather than raw bytes. This is exactly what Magenta did, and it produces dramatically better results. The byte-level approach conflates musical structure with file format structure, which is why the output sounds like it's trying to be music but keeps getting distracted by MIDI headers.

Also: more data. Beethoven's 32 piano sonatas is a small corpus. The model was almost certainly memorizing chunks rather than learning generalizable structure.

But for a weekend experiment in 2015 with a character-level LSTM and a pile of MIDI files — not bad.
