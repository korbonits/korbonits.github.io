---
title: I Shipped a Feature From My Phone During Lunch
date: 2026-03-25
draft: false
description: I shipped a working stock price dashboard from my phone during a
  lunch break using Claude Code — and learned that execution is getting cheaper
  while judgment stays expensive.
tags:
  - vibe-coding
  - claude
  - astro
  - web-dev
  - ai-tools
---
Today, I built and deployed a working stock price dashboard during a lunch break, entirely from my phone, using Claude Code. No laptop. No IDE. Just a conversation.

I want to talk about what that means — and what it doesn’t.

## The Arc

A few weeks ago I revived my personal blog at korbonits.com after a multi-year hiatus. What started as a migration from Jekyll to Astro turned into something I’m calling my Tier 1 vibe coding curriculum: a sequence of small, shippable projects designed to rebuild web development instincts I’d let atrophy while spending most of my time on ML infrastructure.

The projects in order:

- Blog migration (Jekyll → Astro)
- CMS layer via Decap CMS
- A /now page with a nightly GitHub Actions cron rebuild
- A stock price dashboard

Each one introduced a concept. The CMS project taught config-driven architecture. The /now page taught the build-time vs. runtime tradeoff. The stock dashboard was supposed to be straightforward — and mostly was.

## The Dead End

Before landing on stock prices, I tried to build a Claude token usage dashboard. The idea: hit the Anthropic usage API on a nightly cron, store daily snapshots as JSON, render them as a chart. A post called “What my token usage tells me about how I actually work” was already writing itself.

It didn’t work. The Anthropic usage API tracks API consumption only — not claude.ai usage. Since most of my Claude consumption happens in the chat interface, the data was nearly empty. Dead end.

This is the part vibe coding evangelists underplay: the AI executes your plan faithfully even when your plan is wrong. Claude Code wired up the API call perfectly. The architecture was clean. The data just wasn’t there. That’s a product judgment failure, not a coding failure — and the AI can’t catch it for you.

## The Lunch Break

I pivoted to stock prices. Alpha Vantage has a free tier, decent documentation, and the API is simple enough that the whole thing is client-side JavaScript — no backend, no environment variable headaches, compatible with GitHub Pages.

Here’s the core of what Claude Code produced:

```javascript
async function fetchStockPrice(ticker) {
  const url = `https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol=${ticker}&apikey=${API_KEY}`;
  const response = await fetch(url);
  const data = await response.json();
  const quote = data["Global Quote"];
  return {
    ticker,
    price: parseFloat(quote["05. price"]).toFixed(2),
    change: parseFloat(quote["09. change"]).toFixed(2),
    changePercent: quote["10. change percent"].trim()
  };
}
```

I gave Claude Code a single detailed prompt describing what I wanted — a `/stocks` page, a list of tickers, a clean table with daily change, client-side only. It read my existing Astro project structure, matched my conventions, and produced working code. I reviewed the diff, pushed, and the page was live before my lunch break ended.

From my phone.

## What This Actually Means

I’ve been doing ML engineering long enough to be skeptical of tools that promise to change everything. Most don’t. But something real has shifted here.

The constraint used to be the environment — you needed a laptop, a proper editor, a terminal. The bottleneck was physical access to the right setup. Claude Code running in a mobile browser removes that constraint almost entirely for a certain class of work: small, well-scoped features in a codebase the model can read and understand.

The constraint that remains — and this is important — is judgment. Knowing what to build, how to scope it, when a plan is wrong before you execute it. That doesn’t compress. If anything it becomes more valuable as execution gets cheaper.

The lunch break wasn’t impressive because I wrote code on my phone. It was impressive because I had a clear enough plan that I could hand it off completely and trust the output.

That’s the skill worth developing.

-----

*korbonits.com is my personal blog. I write about ML, software, and books*
