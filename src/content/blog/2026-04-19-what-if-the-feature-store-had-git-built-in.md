---
title: What If the Feature Store Had Git Built In?
date: 2026-04-19
draft: true
description: A weekend spike exploring whether Dolt — the version-controlled SQL
  database — could replace the ROW_NUMBER dedupe at the heart of every feature
  store’s point-in-time join. Four lines of SQL instead of thirteen, and the gap
  widens with every feature view you add.
tags:
  - mlops
  - feature-stores
  - feast
  - dolt
  - data-versioning
---
A feature store spends most of its lifetime answering one question: *what did this customer look like on the day we trained the model?* Every backend I’ve worked with answers it the same way — an append-only event log, plus a hairball of `ROW_NUMBER() OVER (PARTITION BY entity ORDER BY timestamp DESC)` CTEs that dedupe the log down to the latest-known-before-training-time row per entity.

It works. It’s also a pattern every offline-store backend has to re-implement, and a pattern that, if you get it wrong, silently produces training-serving skew.

Last weekend I asked: *what if the database just did this?*

## The Observation

[Dolt](https://github.com/dolthub/dolt) is a version-controlled SQL database. You commit tables the way you commit code. Every commit is immutable, every row has a full history, and — the bit that matters here — you can query any table at any revision with `AS OF`:

```sql
SELECT * FROM customer_features AS OF 'train_2026_04_01';
```

That’s a point-in-time read. It’s native to the engine. No `ROW_NUMBER`, no `created_ts <=`, no tie-breaker logic.

I stared at this for a while. The claims [Feast](https://github.com/feast-dev/feast) has been making all along — *point-in-time correctness matters, training reproducibility matters, feature-definition lineage matters* — map almost one-to-one onto what a versioned database already does:

- `AS OF` → point-in-time joins
- Tags → pinned training snapshots, bit-for-bit reproducible
- Branches → per-experiment feature pipelines without polluting `main`
- `dolt diff` → `git diff` for feature definitions

If you squint, Feast and Dolt are two projects that have been solving the same problem at different layers without realizing it.

## The Spike

I built a minimal Feast offline-store plugin, `feast-dolt`, and ran the canonical "give me the training features as of tag T" query both ways against a toy dataset: three daily feature snapshots of two customers, with the first snapshot tagged `train_2026_04_01`.

### One feature view

**Dolt:**

```sql
SELECT customer_id, spend_30d, spend_90d
  FROM customer_transactions AS OF 'train_2026_04_01'
 WHERE customer_id IN (1, 2);
```

**Warehouse (the thing Feast ships today):**

```sql
WITH latest AS (
    SELECT customer_id, spend_30d, spend_90d,
           ROW_NUMBER() OVER (PARTITION BY customer_id ORDER BY created_ts DESC) AS rn
      FROM customer_transactions_log
     WHERE created_ts <= '2026-04-01 23:59:59'
)
SELECT customer_id, spend_30d, spend_90d
  FROM latest
 WHERE rn = 1 AND customer_id IN (1, 2);
```

Four lines versus thirteen. Identical results. Fine — that’s a cute trick.

### Three feature views

The one-FV number is easy to dismiss as cherry-picked. Real training queries don’t retrieve one feature view — they join several onto a single entity list. So I extended the spike to three: customer profile, transactions, and support tickets.

Dolt stayed at one join per feature view: each became one `LEFT JOIN customer_<fv> AS OF 'train_2026_04_01'` line. The warehouse version needed a full `ROW_NUMBER` CTE per feature view before it could join anything.

| Feature views | Dolt `AS OF` | Warehouse `ROW_NUMBER` | Gap  |
|:-------------:|:------------:|:----------------------:|:----:|
| 1             | 4 LOC        | 13 LOC                 | +9   |
| 3             | 13 LOC       | 31 LOC                 | +18  |

The gap doubled. Each additional feature view costs `ROW_NUMBER` a fresh 6-line CTE; it costs `AS OF` one extra line. Production retrievals touch five to twenty feature views, so the real gap at real scale is substantially larger than the toy numbers suggest.

## What’s Already Built

Over the weekend: `feast-dolt` as a proper package — config, source, offline store, retrieval job, unit tests. `pull_latest_from_table_or_query` and `pull_all_from_table_or_query` are implemented; `get_historical_features` is stubbed as the primary RFC target.

I filed [an RFC discussion on feast-dev/feast](https://github.com/feast-dev/feast/discussions/6297) rather than opening a PR, because the shape of this plugin — naming, scope, how it relates to the existing registry — deserves community input before I commit to implementations that’ll be painful to change after a PyPI release.

## What This Probably Is Not

A few things it’s important to say out loud:

1. **Not an online store.** Dolt isn’t a low-latency KV. Keep using Redis, DynamoDB, Milvus for online serving.
2. **Not a warehouse replacement.** Dolt is MySQL-shaped. Snowflake and BigQuery will still win on a 10 TB scan. The target audience is teams under a terabyte who care about reproducibility more than petabyte throughput.
3. **Not my invention.** [Flock Safety has been running a Dolt-backed feature store in production since 2024](https://www.dolthub.com/blog/2024-03-07-dolt-flock/) — they just built the adapter in-house, without Feast. This plugin generalizes that pattern into something reusable.

## Why This Hasn’t Been Built Yet

I think the answer is structural rather than technical. Feature stores and versioned databases grew up in different subcultures — one in ML infrastructure, one in data engineering — and the people thinking hard about one tend not to think hard about the other. Feast assumed its offline store was a warehouse and built warehouse-shaped abstractions. Dolt assumed its users were data engineers and built collaboration-shaped abstractions. The overlap — *a feature store that treats data version control as a database-level primitive* — fell in the gap between the two communities.

It’s also possible I’m missing something obvious and the community will tell me so this week. That’s part of why the RFC exists before the code.

## Where Next

The RFC is live and awaiting feedback. If the direction holds, the headline implementation is `get_historical_features` using `AS OF` joins across feature views — the thing the spike has already proven out on paper. Phase two is a `DoltRegistry` that makes every `feast apply` a commit, with `git blame`-grade visibility into feature-view changes.

And if the feedback is *no, this is a bad fit* — that’s useful too. Much cheaper than finding out after 0.1.0 ships.

The spike, the RFC, and the plugin scaffold are all at [`korbonits/feast-dolt`](https://github.com/korbonits/feast-dolt).

-----

*korbonits.com is my personal blog. I write about ML, software, and books.*
