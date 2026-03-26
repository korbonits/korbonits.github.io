---
title: Building a Self-Hosted Analytics Dashboard in One Session
date: 2026-03-25
draft: false
description: No Google Analytics, no third-party scripts, no cookies. Just a Netlify function, Supabase, and a Chart.js dashboard I actually own.
tags:
  - meta
  - web-dev
  - netlify
  - astro
  - supabase
  - vibe-coding
---

I don't want Google Analytics on this site. I don't want any third-party analytics script that phones home, sets cookies, or ends up in some ad network's data pipeline. But I also want to know if anyone is reading what I write.

The solution is obvious in theory: track it yourself. Tonight I built it.

## The Design

Five pieces:

1. A beacon script on every page that fires a non-blocking POST to `/api/track`
2. A Netlify serverless function that validates the request and writes to Supabase
3. A `page_views` table in Supabase with indexes on `path` and `created_at`
4. An `/analytics` page that queries Supabase at build time and renders charts
5. A Netlify edge function that puts basic auth in front of that page

The beacon sends `path`, `referrer`, `user_agent`, and `timestamp`. The function extracts the country from Netlify's `x-nf-country` header — a free geolocation signal that Netlify injects automatically on every request. No GeoIP lookup, no extra service.

The analytics page builds once per day (the site already has a daily Netlify rebuild). It shows total views, views today, views last 7 days, a 30-day line chart, a top-10-pages bar chart, and a referrer breakdown grouped by domain. Chart.js via CDN, dark mode aware, same font and color scheme as the rest of the site.

## The Bugs Worth Mentioning

**The beacon was silently doing nothing.** I put the tracking script in `Layout.astro`. It compiled fine, deployed fine, and fired on exactly zero page loads — because none of the pages actually use `Layout.astro`. They all use `BaseHead` directly. Classic case of writing to the right abstraction for the wrong reason. Moving the script to `BaseHead.astro` fixed it immediately.

There was a second wrinkle: this site uses Astro's `ClientRouter` for view transitions, which means pages don't fully reload on navigation. A regular `<script>` tag only fires on the initial load. The fix is to listen for `astro:page-load` instead, which fires on both initial load and every client-side navigation:

```javascript
document.addEventListener('astro:page-load', () => {
  fetch('/api/track', { method: 'POST', keepalive: true, ... });
});
```

**The edge function auth loop.** Protecting `/analytics` with basic auth via an edge function sounds simple. It isn't, because of trailing slashes. Astro builds the page at `/analytics/index.html`, so the canonical URL is `/analytics/`. If you put the edge function on `/analytics` (no slash), it passes auth, calls `context.next()`, Netlify 301-redirects to `/analytics/`, and the browser drops the `Authorization` header on the redirect. A second auth challenge fires, the browser retries, Chrome hits its retry limit, and you get `ERR_TOO_MANY_RETRIES` with no dialog ever appearing.

The fix: only guard `/analytics/`. Netlify's trailing-slash normalization happens before the edge function runs, so the redirect is transparent and the auth challenge fires exactly once on the canonical URL.

**Chrome cached credentials.** After debugging the loop, I was still getting `ERR_TOO_MANY_RETRIES` even after the fix deployed. Turns out Chrome had silently cached the bad credentials from the broken earlier deploys and was auto-retrying them on every request, never surfacing a dialog. Opening an incognito window confirmed the fix was working. Cleared the cached credentials and normal browsing worked too.

## The Honest Take

The whole thing — design, implementation, debugging, deploy — took one session. The interesting problems were not the ones I expected. I thought the hard part would be the Supabase integration or the Chart.js charts. Those were fine. The bugs were in the wiring: a layout component nobody uses, a script that doesn't survive view transitions, a redirect that strips auth headers.

These are exactly the kind of subtle environmental issues that are hard to anticipate in advance and fast to fix once you see them. I saw them because the feedback loop was tight — deploy, test, observe, fix, repeat. Claude Code kept the whole context in one place so I didn't have to.

The dashboard is live. The data is mine. No cookies, no third parties, no tracking I don't control.

---

*korbonits.com is my personal blog. I write about ML, software, and books.*
