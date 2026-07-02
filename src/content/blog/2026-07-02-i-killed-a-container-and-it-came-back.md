---
title: "I Killed a Container and It Came Back"
description: "I can design batch, real-time, and streaming inference — the data plane. But I'd never built the control plane that manages it. So I built the smallest one I could, from scratch, to finally understand reconciliation: the one idea that separates a control plane from a deploy script."
date: 2026-07-02
draft: false
tags: ["mlops", "kubernetes", "python", "vibe-coding", "control-plane"]
---

# I Killed a Container and It Came Back

I killed a running container — `docker kill`, gone — and a few seconds later it came back. Nothing restarted it. I hadn't told anything to restart it. A loop I'd written, maybe forty lines of Python, had just noticed the world no longer matched what I'd asked for and quietly put it right.

That small uncanny moment is the essence of what a control plane is, and until this week I'd never actually built one. Frankly, I am still grokking what it is.

## The gap I didn't know I had

I can design production architectures for batch, real-time, and streaming inference. I know how to make predictions flow. But that's all *data plane* — the part that does the work. The thing that *manages* the work — decides what should be running, notices when it isn't, and fixes it — is the control plane, and I'd only ever used other people's. Kubernetes. Spinnaker. I'd read about how they work. Magic, essentially. Black boxes I could use just enough to get something working (barely). I had never written the loop.

Which started to bother me, because I've been designing a control plane for real model serving — the piece that would sit alongside [sheaf](/blog/2026-04-14-sheaf-vllm-for-non-text-foundation-models) and manage the lifecycle of everything it serves. I could draw the boxes. I wasn't sure I understood the thing inside the most important box.

So I built the smallest honest version I could: a control plane that keeps N copies of a Docker container alive to match a number I declare. When you want to *feel* a pattern instead of reading about it, build the tiniest real version and stare at it. And play with it. 

## Take orders, or hold an invariant

Here's the instinct you have to unlearn. The obvious API is imperative: `POST /deploy`, `POST /scale?n=5`. You tell the system to *do things*. Each call is an action, executed once, then forgotten. If a container dies an hour later, nothing brings it back — the action already succeeded.

A control plane doesn't take orders. You hand it *desired state* — "there should be three of these" — and it runs a loop forever: look at what's actually running, compare it to what you asked for, and make up the difference. Observe, diff, act. Again and again.

The distinction sounds academic until you watch it. Because the loop reacts to the *gap* between desired and actual — not to *events* — it didn't matter that nothing was watching the instant I killed the container. On the next pass, three was the goal, two was the truth, and the loop made a third. It's *level-triggered*, not edge-triggered: it can't miss an event because it isn't listening for events. It just keeps asking "does reality match?" and corrects when it doesn't.

One sentence holds it: **a control plane maintains an invariant; it does not execute a command.** Almost everything else is detail.

## The details, briefly

I built it up in a few moves, each teaching exactly one thing:

- a hardcoded loop over Docker — self-healing works;
- an API and a typed SDK, so desired state is *declared*, not baked in — and scaling becomes "re-declare a bigger number." There's no scale endpoint at all;
- desired state moved into Postgres, with the API and the reconciler split into two separate processes that only talk through the database — so state outlives any process. Kill the reconciler, start a fresh one, and it rebuilds the world from the DB;
- a second kind of resource, plus Kubernetes-style *conditions*, so a thing can report "not ready — because it's waiting on that other thing."

The [code is on GitHub](https://github.com/korbonits/tiny-control-plane). It's about four hundred lines across five small files, and none of it is clever. That's the point.

## Where it bit me

The parts I learned most from were the parts that broke.

The Docker CLI and the Python SDK disagreed about where Docker even *was*. OrbStack puts its socket somewhere the SDK's defaults don't look, so the CLI worked and my code didn't, with an error that pointed at nothing in particular. Filed away: "it works in the terminal" and "it works from my code" are not the same claim.

But the best one: for a while my control plane was quietly, confidently wrong. I'd declare an app that needed a network which didn't exist yet, and it would report itself correctly *blocked* — while two containers ran anyway. It took me embarrassingly long to see it. There were **two reconcilers**: an old one I'd left running from an earlier experiment, and the new one, both reading the same database, both acting on it. One was starting containers; the other was reporting them stuck. They were fighting.

That isn't a bug in my code so much as a law I walked straight into: **you can have exactly one thing writing the world.** The moment two reconcilers share a source of truth, they race, and the state thrashes. This is *why* real control planes do the elaborate thing — leader election, a single active controller, a consensus store underneath. I'd always filed "leader election" under ceremony. Now I've felt the precise failure it exists to prevent, in an evening, on my laptop.

## Which plane are you on?

"Data plane" and "control plane" turn out to be two terms from a small vocabulary worth having. The classic split is three:

- the **data plane** does the work, per request — it's what serves your traffic, on the hot path;
- the **control plane** decides what should be running and keeps it that way — the loop I'd just built;
- the **management plane** is how humans and tools drive it: the APIs, CLIs, and dashboards, the `kubectl apply` you type.

The pair that took me longest to pull apart was control and management. My little API and SDK — the part I'd `curl` — is management plane. The reconciler grinding away behind it is control plane. Intent goes in the front; realization happens in the back.

And here's the property that makes the whole split worth the trouble — the one my toy had been demonstrating without my noticing: **the data plane has to survive the control plane.** When I killed the reconciler, the containers kept running. Serving didn't stop; it just stopped *self-healing*. That's not luck, it's the design goal — AWS calls it static stability: in steady state the data plane shouldn't depend on the control plane, only get reconfigured by it. Your models should keep answering even if everything that manages them is down.

## What it actually bought me

Not the code — the reframe. "Understanding Kubernetes" had always meant, to me, remembering enough YAML. It doesn't. It means understanding that the system is continuously holding an invariant against a world that keeps drifting — and that everything above it (the CRDs, the controllers, the reconcile loops, the leader election) is machinery in service of that one idea. Once you've written the loop yourself you see it everywhere: it's Crossplane, it's Argo CD, it's whatever your cloud's control plane is doing while you sleep.

I built this to prepare for something larger — the real serving control plane alongside [sheaf](/blog/2026-04-14-sheaf-vllm-for-non-text-foundation-models). But the payoff showed up right away, and not where I expected: the pattern is small, it's learnable in a night, and you only really get it by building the tiniest version and then killing a container to watch it come back.

The whole thing was an evening of what I've started calling vibe-learning — the sibling of vibe-coding, where the goal isn't the artifact but the understanding, and the AI clears the yak-shaving so your attention stays on the idea. Four hundred lines, one evening, and a pattern I'll never have to take on faith again.
