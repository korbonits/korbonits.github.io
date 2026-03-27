---
title: Migrating from GitHub Pages to Netlify in an Evening
date: 2026-03-25
draft: false
description: Another hosting migration I'd been putting off for years, done in 30 minutes with Claude Code.
tags:
  - meta
  - web-dev
  - netlify
  - astro
  - vibe-coding
---

I migrated korbonits.com from GitHub Pages to Netlify tonight. It took about 30 minutes. I'd been putting it off for months.

That's the post. But let me give you the details.

## Why Bother

GitHub Pages is fine. It's free, it's simple, and for a static Astro blog it works. But "fine" accumulates friction. Every deploy went through a GitHub Actions workflow that was honestly more complex than the site deserved — build job, artifact upload, deploy job, pages permissions, the whole thing. And the custom domain setup on GitHub Pages has always felt like an afterthought: a `CNAME` file in the repo root, DNS A-records pointing at GitHub's IPs, hoping it all holds together.

Netlify does this better. It's designed for it. The deploy pipeline is first-class, not bolted on.

## What It Actually Took

One config file. That's it.

```toml
[build]
  command = "npm run build"
  publish = "dist"

[build.environment]
  NODE_VERSION = "22"
```

Push that, connect the repo in the Netlify dashboard, and the first deploy runs automatically. The build command (`astro build && pagefind --site dist`) ran without modification. Environment variables transferred in under a minute.

The DNS migration — the part I'd always imagined being painful — was four nameservers pasted into Squarespace. Netlify gave me the values, Squarespace had a "custom nameservers" field, done. SSL provisioned automatically. The whole thing propagated faster than I expected.

One thing I missed: email forwarding. If your domain is registered through Squarespace (formerly Google Domains), email forwarding is handled via Mailgun under the hood. Switching to Netlify DNS drops those MX records. Mail stops bouncing but never arrives. The fix: add mxa.mailgun.org and mxb.mailgun.org as MX records (both priority 10) in Netlify DNS, plus a TXT record v=spf1 include:mailgun.org ~all. Then verify your forwarding rule still exists in account.squarespace.com under your domain's Email settings. Five minutes, but only if you remember to do it before you need it.

## The Cron Problem

My site has a daily rebuild — a GitHub Actions cron job that triggers at 6am UTC to refresh data on the `/stocks` and `/now` pages. Migrating to Netlify meant replacing that.

The solution: a Netlify build hook (a POST-able URL that triggers a deploy) plus a new, stripped-down GitHub Actions workflow:

```yaml
name: Daily Rebuild

on:
  schedule:
    - cron: '0 6 * * *'
  workflow_dispatch:

jobs:
  trigger:
    runs-on: ubuntu-latest
    steps:
      - run: curl -X POST -d '{}' ${{ secrets.NETLIFY_BUILD_HOOK }}
```

The hook URL lives in GitHub Secrets. The workflow does one thing. This is the right amount of complexity.

## What I Deleted

The old GitHub Pages deploy workflow was 47 lines. It needed `pages: write` and `id-token: write` permissions, two jobs with an artifact handoff between them, and three different GitHub Actions maintained by the GitHub team. It worked, but it was a lot of machinery for "put these files on the internet."

The replacement is 12 lines. The cognitive overhead is proportionally smaller.

## The Honest Take

I'd been putting this off because "hosting migration" sounds like a project. It isn't. The activation energy was the obstacle, not the work.

Claude Code walked through every step with me in real time — `netlify.toml` config, DNS walkthrough, the build hook setup, deleting the old workflow, writing the new one. I didn't have to context-switch into documentation or figure out the Netlify UI from scratch. I just described what I wanted and we did it together.

This is the pattern I keep coming back to: the things I've been deferring for months aren't actually hard. They just have enough surface area that starting them alone felt like work. With Claude Code, the surface area collapses.

Now korbonits.com deploys faster, the pipeline is simpler, and I never have to think about GitHub Pages again.

End of an era.

---

*korbonits.com is my personal blog. I write about ML, software, and books.*
