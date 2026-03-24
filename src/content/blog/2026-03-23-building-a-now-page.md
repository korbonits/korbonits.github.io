---
title: "Building a /now Page That Updates Itself"
date: 2026-03-23
description: "How I added a /now page to my Astro blog that pulls live GitHub activity and Goodreads data at build time — and rebuilds itself every morning without me touching anything."
tags: ["meta", "blogging", "astro", "web", "github", "books"]
draft: false
---

The [/now page](https://nownownow.com/about) is a concept from [Derek Sivers](https://sive.rs/now) — a page that answers the question "what are you focused on right now?" It's like an about page, but current. Thousands of personal sites have one.

I wanted one too, but I didn't want to manually update it. So I built it to update itself.

## How it works

My site is built with [Astro](https://astro.build) and deployed as a static site via GitHub Pages. Astro runs all page logic at build time, which means you can `fetch()` an API inside a `.astro` file and the result gets baked into the HTML — no client-side JavaScript, no backend.

The GitHub API is public for any user's activity. No auth token needed. So the `/now` page just fetches my recent push events at build time and renders them:

```astro
const res = await fetch('https://api.github.com/users/korbonits/events/public?per_page=100');
const events = await res.json();

const pushedRepos = [...new Map(
  events
    .filter(e => e.type === 'PushEvent')
    .map(e => [e.repo.name, { name: e.repo.name, url: `https://github.com/${e.repo.name}`, date: new Date(e.created_at) }])
).values()].slice(0, 5);
```

The `Map` deduplicates by repo name so you see each repo once, ordered by most recent push.

## Making it actually stay current

A static site only updates when it rebuilds. If I only rebuild on push, the "now" page would only be as fresh as my last blog post — which defeats the purpose.

The fix is a scheduled GitHub Actions workflow. One line added to `deploy.yml`:

```yaml
on:
  push:
    branches: [master]
  workflow_dispatch:
  schedule:
    - cron: '0 6 * * *'  # daily at 6am UTC
```

Now the site rebuilds every morning at 6am UTC regardless of whether I've pushed anything. The `/now` page always reflects the last 30 days of GitHub activity.

## The SSL gotcha

One thing that tripped me up locally: `fetch` failed with `unable to get local issuer certificate`. This is a corporate SSL inspection issue — my machine has a custom CA that Node doesn't trust by default.

The fix for local dev is just an env var:

```bash
NODE_TLS_REJECT_UNAUTHORIZED=0 npm run dev
```

This doesn't affect production builds running in GitHub Actions, where the cert issue doesn't exist.

## Adding Goodreads

GitHub activity is a good start but a `/now` page is really about more than code. I've been tracking my reading on [Goodreads](https://www.goodreads.com/user/show/49504536-alex-korbonits) since 2016, and it turns out their RSS feeds still work even though the API is officially deprecated. Each shelf has a public feed at:

```
https://www.goodreads.com/review/list_rss/<user_id>?shelf=currently-reading
```

Same pattern as GitHub — fetch at build time, parse the response, render it. The only wrinkle is that Goodreads wraps some fields in CDATA tags, so titles come back looking like `<![CDATA[The Mind Illuminated]]>`. A quick regex strips them:

```astro
const xml = await res.text();
const items = xml.match(/<item>([\s\S]*?)<\/item>/g) ?? [];
const books = items.slice(0, 5).map(item => ({
  title: (item.match(/<title>([\s\S]*?)<\/title>/)?.[1] ?? '').replace(/<!\[CDATA\[([\s\S]*?)\]\]>/, '$1'),
  author: (item.match(/<author_name>([\s\S]*?)<\/author_name>/)?.[1] ?? '').replace(/<!\[CDATA\[([\s\S]*?)\]\]>/, '$1'),
  url: (item.match(/<link>([\s\S]*?)<\/link>/)?.[1] ?? '').replace(/<!\[CDATA\[([\s\S]*?)\]\]>/, '$1'),
}));
```

Both fetches run in parallel with `Promise.all`, so there's no added latency at build time. The page now shows what I'm coding and what I'm reading, updated every morning automatically.

## The manual section

Not everything has an API. Where I am, what I'm working on, what I'm thinking about — that stuff needs to be written. I added a `src/content/now.md` file with a simple frontmatter schema:

```md
---
location: Seattle, WA
working_on: Migrating my blog from Jekyll to Astro and building out new personal projects.
thinking_about: How generative AI is changing the craft of software engineering, and what skills will matter most in five years.
---
```

Astro's content collections pick it up at build time just like blog posts. The page renders it at the top — location, what I'm working on, what I'm thinking about — and the automated sections follow below. This is the part I'll actually edit occasionally. The rest updates itself.

## Getting listed

[nownownow.com](https://nownownow.com) is Derek Sivers' directory of now pages. Once your page is live, you can submit it there. Mine is at [korbonits.com/now](https://korbonits.com/now) — go check it out, and if you want to build your own, everything above is all you need.
