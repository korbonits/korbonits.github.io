---
title: "19 Open Source Pull Requests in One Afternoon"
date: 2026-04-12
draft: false
description: "How I used Claude Code to systematically add Ruff linting and CI to 19 high-starred research repos — and what the week that followed revealed about the difference between correctness and value."
tags:
  - open-source
  - python
  - ruff
  - ci
  - claude
  - ai-tools
  - ml
---

It started with a single question: *what if we added Ruff to this repo?*

The repo in question was [rethink_sft_generalization](https://github.com/Nebularaid2000/rethink_sft_generalization), a paper release from ByteDance/NVIDIA. My first instinct was to look at the existing tooling. `.pre-commit-config.yaml` already had Ruff v0.12.2. The `pyproject.toml` had the lint configuration. Everything was there — it just wasn't wired to CI, and nobody would ever run it.

That gap — tooling declared but not enforced — turned out to be everywhere.

## The Pattern

After working through a handful of repos manually, the pattern became clear enough to systematize. ML research repositories tend to fall into one of three categories:

1. **Nothing** — no pyproject.toml, no linting config, no CI beyond a release pipeline
2. **Half-done** — Ruff declared in dev dependencies or pyproject.toml, but no pre-commit hooks, no lint job in CI
3. **Done** — Ruff configured, enforced, wired to PRs (rare)

The third category is mostly larger, more mature projects with dedicated infra teams. The first two are the vast majority of paper repos — even high-starred ones from top labs.

What made this tractable with Claude Code was that the check-and-fix loop became nearly mechanical:

1. Fetch `pyproject.toml` and `.pre-commit-config.yaml` — does Ruff already exist?
2. Clone, run `uvx ruff check . --statistics` — what's the violation count and how many are auto-fixable?
3. Apply `--fix` with a conservative ignore list
4. Investigate what remains — are they systematic patterns or genuine bugs?
5. Write the config, pre-commit, and CI workflow, open the PR

The interesting work was in step 4. That's where judgment still matters.

## Edge Cases Worth Noting

**jaxtyping annotations.** Google DeepMind's [alphagenome](https://github.com/google-deepmind/alphagenome) and [timesfm](https://github.com/google-research/timesfm) both use jaxtyping, which annotates array shapes inline: `Float[Array, "b n h d"]`. Ruff's F722 and F821 rules flag these as syntax errors and undefined names respectively — they're not. They're a domain-specific annotation language. Adding `F722` and `F821` to the global ignore list was the right call; explaining *why* in the PR description mattered.

**Existing formatters.** Meta's [vjepa2](https://github.com/facebookresearch/vjepa2) uses black at 119 characters. Google's [alphagenome](https://github.com/google-deepmind/alphagenome) uses pyink at 80 characters with 2-space indentation. Running `ruff format` on either would produce a massive diff fighting their existing formatter. The right answer: lint-only pre-commit hooks, lint-only CI step, match their line-length in the Ruff config. Don't colonize their formatting choices.

**Gitignored configs.** Both [VibeVoice](https://github.com/microsoft/VibeVoice) and [InfiniteYou](https://github.com/bytedance/InfiniteYou) had `.pre-commit-config.yaml` in their `.gitignore`. This is a deliberate signal. I didn't force-add the file — I just wired up the CI workflow and noted in the PR description that pre-commit was excluded. Respecting what maintainers have explicitly opted out of is basic PR hygiene.

**Dead code.** Apple's [ml-simplefold](https://github.com/apple/ml-simplefold) had 11 F821 violations (undefined names) and 1 F601 (duplicate dictionary key) — all in code that sits after an early `return` statement and can never execute. I didn't delete the dead code (too invasive for a tooling PR), noted it in the description, and added the rule codes to the ignore list.

**Committed debug statements.** Microsoft's [Magma](https://github.com/microsoft/Magma) had four live `import pdb; pdb.set_trace()` calls across three files — not commented out, not guarded, just sitting there in production paths. I added `# noqa: I001` rather than removing them; that's the maintainers' call. But I flagged it explicitly in the PR body.

## The Numbers

By the end of the afternoon, 19 PRs were open across repos from Apple, ByteDance, Google, Google DeepMind, HuggingFace, Meta, Microsoft, and others. Three more followed the next morning. As of this writing:

| Repo | PR | Stars | Status |
|---|---|---|---|
| opendatalab/MinerU | [#4773](https://github.com/opendatalab/MinerU/pull/4773) | 26k | closed |
| TauricResearch/TradingAgents | [#536](https://github.com/TauricResearch/TradingAgents/pull/536) | 7k | open |
| facebookresearch/lingua | [#100](https://github.com/facebookresearch/lingua/pull/100) | 4.8k | open |
| huggingface/nanoVLM | [#205](https://github.com/huggingface/nanoVLM/pull/205) | 4.8k | open |
| facebookresearch/vjepa2 | [#152](https://github.com/facebookresearch/vjepa2/pull/152) | 3.6k | open |
| google-research/timesfm | [#403](https://github.com/google-research/timesfm/pull/403) | 3k | open |
| bytedance/InfiniteYou | [#50](https://github.com/bytedance/InfiniteYou/pull/50) | 2.7k | open |
| microsoft/MoGe | [#150](https://github.com/microsoft/MoGe/pull/150) | 2.4k | open |
| huggingface/picotron | [#38](https://github.com/huggingface/picotron/pull/38) | 2.1k | open |
| google-deepmind/alphagenome | [#43](https://github.com/google-deepmind/alphagenome/pull/43) | 1.9k | closed |
| microsoft/Magma | [#92](https://github.com/microsoft/Magma/pull/92) | 1.9k | open |
| microsoft/mattergen | [#242](https://github.com/microsoft/mattergen/pull/242) | 1.7k | open |
| bytedance/pasa | [#50](https://github.com/bytedance/pasa/pull/50) | 1.5k | open |
| microsoft/rStar | [#66](https://github.com/microsoft/rStar/pull/66) | 1.4k | open |
| apple/ml-clara | [#10](https://github.com/apple/ml-clara/pull/10) | 1.1k | open |
| apple/ml-simplefold | [#52](https://github.com/apple/ml-simplefold/pull/52) | 960 | open |
| microsoft/VibeVoice | [#338](https://github.com/microsoft/VibeVoice/pull/338) | — | open |
| AMAP-ML/SkillClaw | [#2](https://github.com/AMAP-ML/SkillClaw/pull/2) | — | merged |
| Osilly/Interleaving-Reasoning-Generation | [#7](https://github.com/Osilly/Interleaving-Reasoning-Generation/pull/7) | — | open |
| lmgame-org/GamingAgent | [#80](https://github.com/lmgame-org/GamingAgent/pull/80) | — | open |
| zjunlp/LightMem | [#58](https://github.com/zjunlp/LightMem/pull/58) | — | open |
| yyfz/Pi3 | [#145](https://github.com/yyfz/Pi3/pull/145) | — | open |

That's roughly 55,000 stars worth of repos, across five major AI labs. Nineteen of the twenty-two are still open. MinerU and alphagenome closed theirs without merging; AMAP-ML/SkillClaw merged. The merge rate on cold linting PRs is roughly 5%, even with everything done correctly — which is exactly the thesis at the bottom of the post.

There are also a handful of older open PRs doing similar work: replacing black/isort with Ruff in [allenai/OLMo](https://github.com/allenai/OLMo/pull/909), [facebookresearch/dinov2](https://github.com/facebookresearch/dinov2/pull/595), and [microsoft/markitdown](https://github.com/microsoft/markitdown/pull/1718).

## The Week After

While the linting PRs sat in maintainer queues, a week of real contributions accumulated — and most of those moved.

The one I'm most proud of: [HKUDS/LightRAG#2941](https://github.com/HKUDS/LightRAG/pull/2941). LightRAG (28k stars) had a production bug where graph edge counts drifted above VDB relation counts — 372 orphan edges reported on a live instance. The root cause was a partial failure during entity merging: edges were written to the graph before being upserted into the relationships VDB. If the VDB upsert failed partway through — embedder crash, context-length exceeded with a high-degree hub entity, network timeout — the graph held new edges with no VDB counterpart. A subsequent delete then cleaned up the source entities, orphaning the edges permanently.

The fix: wrap each per-edge VDB upsert in a try/except, collect failures, roll back the corresponding graph writes. Graph and VDB stay in sync: either both have the edge, or neither does. I also added `check_graph_consistency()` and `repair_graph_consistency()` utilities for existing deployments with accumulated drift.

Over in [milvus-io/milvus](https://github.com/milvus-io/milvus/pull/49004) (the core C++/Go service), `BloomFilterSet.PkCandidateExist` was reading `currentStat` and `historyStats` without holding `statsMutex` — while every other read method in the same struct took the lock. A concurrent write produces a data race detectable by `-race`. Two lines added, data race closed. The bot greeted me: *"Welcome to milvus-io/milvus 🎉"*

On the pymilvus client side, three API-fix PRs address friction that bites users quietly: [`filter=` not accepted as an alias for `expr=`](https://github.com/milvus-io/pymilvus/pull/3410) in `AnnSearchRequest`, [`consistency_level` returned as an opaque integer](https://github.com/milvus-io/pymilvus/pull/3409) instead of its string name, and [`uuid.UUID` / `os.PathLike` not inferred as `VARCHAR`](https://github.com/milvus-io/pymilvus/pull/3408) in dtype inference. None of these are flashy. All of them are the kind of thing that makes a client library a joy vs. a grind.

In Feast, four PRs merged — sphinx API docs ([#6271](https://github.com/feast-dev/feast/pull/6271)), dead code removal ([#6266](https://github.com/feast-dev/feast/pull/6266)), a fix for five bugs in the Milvus online store ([#6275](https://github.com/feast-dev/feast/pull/6275)), and support for SQL strings as `entity_df` in the remote offline store ([#6265](https://github.com/feast-dev/feast/pull/6265)). The most hazardous of the Milvus five: `update()` was replacing the entire `_collections` cache with a single-entry dict on every call, corrupting all subsequent lookups for any other collection in the same store.

And then there was [Kronos#238](https://github.com/shiyu-coder/Kronos/pull/238) — three silent bugs in sampling and quantization, found via `ty` during a tooling audit. One of them: calling `sample_from_logits(top_p=0.9, top_k=None)` raised a `TypeError` at runtime because the guard let you enter the block, then immediately hit `top_k > 0` on a `None`. The type checker flagged it in five seconds. The fix was three lines. These bugs were invisible in testing because they only surface when exactly one of the two optional sampling params is passed — which is the most natural way to call the function.

The pattern: linting PRs are mostly still open. The real contributions are mostly moving — four of four Feast PRs merged, two of three pymilvus PRs merged ([#3409](https://github.com/milvus-io/pymilvus/pull/3409) and [#3410](https://github.com/milvus-io/pymilvus/pull/3410)) with one closed without merging ([#3408](https://github.com/milvus-io/pymilvus/pull/3408)), and the milvus core PR still in review.

## What This Isn't

It's worth being clear about what these PRs are not.

They're not clever. Adding Ruff to a repo that's missing it is a five-minute task if you know what you're doing — I just did it nineteen times. The value is in doing it consistently, at scale, with enough judgment to handle the edge cases without breaking things.

They're also not guaranteed to get merged — and that turns out to matter more than I initially gave it credit for. The MinerU PR was closed. I couldn't even comment on it. When I followed up with a discussion ticket to ask why, the answer was clear: they didn't want it. That's a meaningful signal. Linting PRs from strangers are easy to decline, and the maintainers' calculus is reasonable — they didn't ask for the PR, it doesn't fix a bug they care about, and merging it means owning the tooling going forward.

In contrast, my most consistently merged contributions have been to [Feast](https://github.com/feast-dev/feast) and [pymilvus](https://github.com/milvus-io/pymilvus) (more traction there than in [Milvus](https://github.com/milvus-io/milvus) itself) — real contributions fixing actual issues, not infrastructure hygiene. Those get merged because they solve a problem someone already had. There's no asymmetry between what the contributor wants and what the maintainer needs.

That gap — between what's *correct* and what's *wanted* — is where unsolicited linting PRs live. The PRs are correct, the work is done, but "correct" isn't the threshold for a healthy open source contribution. *Wanted* is.

## The Deeper Point

The constraint used to be time. Adding Ruff to nineteen repos in an afternoon would have been a tedious manual grind — clone, check, fix, commit, PR, repeat, nineteen times. Claude Code compressed that into something I could do while thinking about other things. Call it *vibe contributing* — the open source equivalent of vibe coding. You're not grinding through the mechanics; you're directing.

But the judgment layer didn't compress. Knowing that jaxtyping produces F722/F821 false positives, knowing not to fight an existing formatter, knowing when dead code is a flag worth raising vs. an issue to silently suppress — that's all still human. The model executes; I decide what execution means.

This is the same thing I noticed when I [shipped a dashboard from my phone during lunch](/blog/2026-03-25-i-shipped-a-feature-from-my-phone-during-lunch): execution is getting cheaper. Judgment isn't. If anything, the value of good judgment goes up as the cost of acting on it goes down.

The week's data makes the thesis cleaner: the 22 linting PRs are mostly still open. The real contributions — LightRAG, Feast, pymilvus, milvus core — are mostly merging or actively moving. Same tools, same afternoon energy, very different outcomes depending on whether you're solving a problem someone already had.

The McDonald's fry analogy I used in the session that produced these PRs: asking if vLLM needs Ruff is like asking if McDonald's needs help with their french fry process. They don't. The question is finding the repos that do — and having enough taste to know the difference.

---

*Meanwhile, I've been building [Sheaf](https://github.com/korbonits/sheaf) — a unified serving layer for non-text foundation models. Separate post [here](/blog/2026-04-14-sheaf-vllm-for-non-text-foundation-models).*

---

*korbonits.com is my personal blog. I write about ML, software, and books.*
