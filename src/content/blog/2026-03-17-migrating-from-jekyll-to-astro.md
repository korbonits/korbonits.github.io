---
title: "I Vibe-coded a Blog Migration in an Hour"
date: 2026-03-17
description: "A year ago this would have taken a week. How I migrated an 11-year-old Jekyll blog to Astro in one evening with two toddlers in the house."
tags: ["meta", "blogging", "astro", "web", "ai"]
draft: false
---

I vibe coded a blog migration in an hour. A year ago this would have taken a week or more.

That's the post. But let me give you the details.

12:21 AM...

```
Go to sleep! 😄

 ▸ Credits: 0.25 • Time: 16s

75% > ok. thanks kiro

> Good night! Great session — you went from a decade-old Jekyll site to a modern Astro blog with search, dark mode, RSS, tags, ToC, social cards, and auto-deploy. All in one evening while holding a
toddler. 🚀

 ▸ Credits: 0.25 • Time: 26s
```

## The Setup

This blog has been running on Jekyll since 2015. Eleven years of posts, a Ruby dependency chain I hadn't touched in years, and a `Gemfile.lock` that was basically archaeology. I'd been meaning to modernize it forever. Specifically, in the aftermath of the Musk Twitter acquisition in 2022, I had seen some great posts on Mastodon about migrating from Jekyll to Hyde (a Jekyll alternative). I was using Evernote religiously and tools like Obsidian and Notion were beginning to explode in popularity. It seemed like a good time to upgrade my stack even then, but I didn't have time while preparing for my first child, and working at AWS.

Last night, my toddler woke up at 10:30pm and needed to be held. I was sitting in the dark with a kid on my chest and my phone in one hand, and I started chatting with Claude about what a Jekyll-to-Astro migration would look like. I didn't even know what Astro was, btw. Kiro suggested it. By the time she fell back asleep and when I was able to transfer her to her crib, I had a plan. By the time I went to bed, the re-designed site was live. Jekyll -> Astro dot js. Boom 💥.

I have two toddlers, a full-time job as a Principal ML Engineer at Disney, and an 11-year-old blog I've been meaning to fix up. I shipped it in a late weeknight hour during what was supposed to be, frankly, my bedtime. 

## What I Used

[Kiro](https://kiro.dev) — Amazon's AI-powered IDE — did the heavy lifting (despite chatting with Claude on my iPhone -- to be sure, I was hitting Claude under the hood via both apps). I described what I wanted, it read my existing Jekyll files, wrote the Astro components, fixed the build errors, and iterated. I mostly directed in the `kiro-cli`.

At first I was quite strict with which commands I would trust kiro to run autonomously, but as kiro proved its worth on simpler tasks I allowed it to do more without additional second-guessing. It is a bit of an Admiral Akbar-eqsue trap however because the dopamine hit from the natural language interface combined with the execution is addicting. I had to consciously remember to read kiro's suggested next commands before allowing it continue, despite how it is designed to easily want to press the `return` button without reading anything at all. I've never played slot machines in Las Vegas, but the urge to hit return without thinking about it seemed analogous.

The workflow was: describe the goal → review the output → catch the edge cases → repeat. Less "write code for me" and more "let's figure this out together."

## What Actually Broke

A few Jekyll-specific things needed fixing:

**`{% highlight python %}` tags** — Jekyll's kramdown syntax for code blocks. Standard markdown fenced blocks (` ```python `) are the Astro equivalent. A one-liner sed command fixed all of them across every post.

**`{% post_url %}` cross-links** — Jekyll's way of linking between posts. These became plain `/blog/slug` paths.

**Astro v6's new content layer API** — Most tutorials online are for Astro v4/v5. In v6, the config file moved from `src/content/config.ts` to `src/content.config.ts`, and `post.render()` became `render(post)`. The error messages were clear enough but worth knowing upfront.

**TLS certificate errors** — `NODE_TLS_REJECT_UNAUTHORIZED=0` to scaffold the project, then `npm config set cafile /etc/ssl/cert.pem` to fix it permanently. Classic.

These are a few gotchas (and I'm sure there are a few tags that haven't been updated yet), but it worked pretty well. I did have a few issues like font colors/contrasts not going well together in Dark Mode, or the header not being in Dark Mode while the main body of the site was, etc.

What was amazing however is that I could simply point this out in natural language in the kiro-cli and it would be fixed within a couple of tries. I didn't need to dig into css or .js files to figure it out.

Note: I am not a front end guy. I barely have any clue what is happening here, and I recall it took me hours to make the smallest changes to css to get my original Jekyll setup going (which was already quite minimalist).

Scientific note: I was also able to get help converting old mathjax into LaTex via `rehype-katex` and `remark-math` which is something I was concerned about given the old formatting and even older libraries.

## The Moment It Just Worked

After fixing the content layer API errors, I ran `npm run build` and watched over a dozen posts from a dozen years generate cleanly. Then we kept going — dark mode, RSS, sitemap, Pagefind search, table of contents, reading time, tags, prev/next navigation, `robots.txt`, `llms.txt`, GitHub Actions deploy, custom domain. Each one took minutes.

Remarkably, this wasn't even a feature roadmap I had in my head. I continually prompted kiro, "what should we build" or "what's next" or "how can we improve this repo".

One moment that impressed me was adding search. Pagefind indexes your static build output and ships a search UI with zero backend. It just *worked* on the first try. 4,085 words indexed across 11 years of posts. After a few updates, we're close to 6k. It updates on every build. That's magic 🪄.

## What I'd Do Differently

The Astro CLI named my project `helpless-houston` (its default starter name) and now that's baked into my repo structure. I'd rename it from the start. `# TODO: rsync -a helpless-houston/ . && rm -rf helpless-houston`

I'd also add descriptions and tags to posts *before* migrating rather than retrofitting them after. Doing it post-migration meant touching every file twice.

Another thing I'd do is not stay up past my bedtime. Matthew Walker describes the pandemic of chronic sleep deprivation in his book Sleep. Which I read, but do not heed as I should :-P. Sometimes that revenge procrastination bedtime just hits different, you know?

## The Honest Take

AI-assisted development has moved fast. Not "AI writes your code" fast — more like "you can now execute on ideas you'd have previously shelved because the activation energy was too high" fast. It's a catalyst.

I've had "modernize the blog" on my list for years. The actual work wasn't hard, it was just enough friction that it never happened. Last night the friction was gone. I had a plan from a phone conversation while holding a sleeping toddler, and a mere hour later the site was live with more features than the Jekyll version ever had.

That's the shift. Not that AI writes better code than you. It's that the gap between "I should do that" and "I did that" just got a lot smaller. 🚀

The source is on [GitHub](https://github.com/korbonits/korbonits.github.io) if you want to see how it's put together. Seriously though, don't bother. Just start with kiro-cli, or Claude Code or BYO agentic IDE.

Real talk, I low key feel powerful. I just did *what* with a static website? In an hour? What should I do next? What about a non-static site? I can build new things to learn and share. The rate limiting factor is no longer code development (perhaps it never was -- it was playing meeting jenga in Outlook with decision makers) but the speed of your ideas. In an era of AI slop, the playing field has been leveled by code generation, but the true differentiation will be in taste. If you have taste and speed, you will succeed.

## Thoughts on next steps to try

I migrated an 11-year blog in an hour. Vibe-coding, early stages. What's next? Here are a few ideas I could consider trying that would incrementally take the challenge up a notch without going to 11, as it were.

1. Augment personal blog site into a portfolio site. Add other projects to showcase. Perhaps they can be vibe-coded as well.
2. Write about machine learning.
3. Show some cool machine learning mini apps, or ... ?
4. What about agents... maybe add an agentic component to the site?

Ideas welcome.