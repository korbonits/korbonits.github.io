---
title: "I Vibe Coded a Blog Migration in an Hour."
date: 2026-03-17
description: "A year ago this would have taken a week. How I migrated an 11-year-old Jekyll blog to Astro in one evening with two toddlers in the house."
tags: ["meta", "blogging", "astro", "web", "ai"]
draft: true
---

I vibe coded a blog migration in an hour. A year ago this would have taken a week or more.

That's the post. But let me give you the details.

## The Setup

This blog has been running on Jekyll since 2015. Eleven years of posts, a Ruby dependency chain I hadn't touched in years, and a `Gemfile.lock` that was basically archaeology. I'd been meaning to modernize it forever.

Last night my toddler woke up at 10:30pm and needed to be held. I was sitting in the dark with a kid on my chest and my phone in one hand, and I started chatting with Claude about what a Jekyll-to-Astro migration would look like. By the time he fell back asleep I had a plan. By the time I went to bed, the site was live.

I have two toddlers, a full-time job as a Principal ML Engineer at Disney, and an 11-year-old blog I've been meaning to fix up. I shipped it in an afternoon during what was supposed to be quiet time.

## What I Used

[Kiro](https://kiro.dev) — Amazon's AI-powered IDE — did the heavy lifting. I described what I wanted, it read my existing Jekyll files, wrote the Astro components, fixed the build errors, and iterated. I mostly directed.

The workflow was: describe the goal → review the output → catch the edge cases → repeat. Less "write code for me" and more "let's figure this out together."

## What Actually Broke

A few Jekyll-specific things needed fixing:

**`{% highlight python %}` tags** — Jekyll's kramdown syntax for code blocks. Standard markdown fenced blocks (` ```python `) are the Astro equivalent. A one-liner sed command fixed all of them across every post.

**`{% post_url %}` cross-links** — Jekyll's way of linking between posts. These became plain `/blog/slug` paths.

**Astro v6's new content layer API** — Most tutorials online are for Astro v4/v5. In v6, the config file moved from `src/content/config.ts` to `src/content.config.ts`, and `post.render()` became `render(post)`. The error messages were clear enough but worth knowing upfront.

**TLS certificate errors** — `NODE_TLS_REJECT_UNAUTHORIZED=0` to scaffold the project, then `npm config set cafile /etc/ssl/cert.pem` to fix it permanently. Classic.

## The Moment It Just Worked

After fixing the content layer API errors, I ran `npm run build` and watched 12 pages generate cleanly. Then we kept going — dark mode, RSS, sitemap, Pagefind search, table of contents, reading time, tags, prev/next navigation, `robots.txt`, `llms.txt`, GitHub Actions deploy, custom domain. Each one took minutes.

The moment that got me was the search. Pagefind indexes your static build output and ships a search UI with zero backend. It just worked on the first try. 4,085 words indexed across 11 years of posts.

## What I'd Do Differently

The Astro CLI named my project `helpless-houston` (its default starter name) and now that's baked into my repo structure. I'd rename it from the start.

I'd also add descriptions and tags to posts *before* migrating rather than retrofitting them after. Doing it post-migration meant touching every file twice.

## The Honest Take

AI-assisted development has moved fast. Not "AI writes your code" fast — more like "you can now execute on ideas you'd have previously shelved because the activation energy was too high" fast.

I've had "modernize the blog" on my list for years. The actual work wasn't hard, it was just enough friction that it never happened. Last night the friction was gone. I had a plan from a phone conversation while holding a sleeping toddler, and a few hours later the site was live with more features than the Jekyll version ever had.

That's the shift. Not that AI writes better code than you. It's that the gap between "I should do that" and "I did that" just got a lot smaller.

The source is on [GitHub](https://github.com/korbonits/korbonits.github.io) if you want to see how it's put together.
