---
title: "Three Things Made My Blog Agent-Ready. Five I Skipped on Purpose."
date: 2026-04-30
draft: false
description: "Cloudflare's isitagentready.com scored korbonits.com at 25 / Level 1 'Basic Web Presence.' Two hours and three small additions later it scored 50 / Level 4 'Agent-Integrated.' The other half of the points came from checks that don't apply to a personal blog — and shipping fake compliance for them would be theatre, not value."
tags:
  - blog-development
  - agent-readiness
  - cloudflare
  - astro
  - netlify
  - vibe-coding
---

![Agent-readiness scan for korbonits.com: overall score 50, Level 4 "Agent-Integrated", with Discoverability 100 (3/3), Content 100 (1/1), Bot Access Control 100 (2/2), API/Auth/MCP/Skill 0 (0/6), Commerce not checked](/images/agent-readiness-after.png)

Cloudflare runs a checker at [isitagentready.com](https://isitagentready.com). You paste in a URL and it grades how friendly the site is to AI agents — robots.txt, markdown negotiation, MCP, OAuth, the works. I tried it on korbonits.com expecting it to laugh at me.

It scored me **25 / 100, Level 1: "Basic Web Presence."**

Two hours later it scored me **50 / 100, Level 4: "Agent-Integrated."** (Above.)

The interesting part isn't the score change. It's the gap between *what the scanner checks* and *what a personal blog should actually publish.* Half of the score is locked behind features that only make sense for sites with APIs, OAuth, MCP servers, or commerce — none of which a personal blog has any business publishing. Adding fake compliance for those checks would be theatre. The right move was to ship the three things that actually applied and skip the rest on purpose.

## The score, before and after

| Category | Before | After |
|---|---|---|
| Discoverability | 67 (2/3) | **100 (3/3)** |
| Content | 0 (0/1) | **100 (1/1)** |
| Bot Access Control | 50 (1/2) | **100 (2/2)** |
| API / Auth / MCP / Skill Discovery | 0 (0/6) | 0 (0/6) ← intentional |
| Commerce | not checked | not checked |
| **Overall** | **25 / Level 1** | **50 / Level 4** |

The "Level 4" upgrade is the surprise. Even though the raw score moved by only 25 points (out of 100), the scanner appears to promote you in larger jumps when you fully clear *categories* than when you accumulate raw points. Two categories at 100 took me from Level 1 to Level 2; the third category cleared jumped me from Level 2 to Level 4 (Level 3 was skipped — presumably reserved for sites that clear at least one of the API/Auth checks). Either way, three of the four scored categories at 100 was enough for "Agent-Integrated" with the API/Auth column still at zero. That's a fair grading rubric for a content site.

## What I shipped

### 1. Content Signals in robots.txt

The single highest-leverage change. One line, takes thirty seconds. Communicates the site's stance on AI consumption explicitly instead of leaving it implicit:

```
User-agent: *
Allow: /
Content-Signal: search=yes, ai-input=yes, ai-train=yes

Sitemap: https://korbonits.com/sitemap-index.xml
```

The directive comes from [contentsignals.org](https://contentsignals.org) and an IETF draft. Three values: `search` (let search engines crawl), `ai-input` (let AI agents read your content for one-shot answers), `ai-train` (let your content be used as training data). I went with `yes` across the board — I write a blog about ML and use Claude as a commit co-author, so anything else would be inconsistent. Your values may differ; the point is that the signal is now *explicit*.

### 2. Link headers via Netlify `_headers`

[RFC 8288](https://www.rfc-editor.org/rfc/rfc8288) Link headers let you advertise resources in HTTP response headers in addition to (or instead of) inline `<link>` tags. AI agents that don't render the page often check headers first. New file at `public/_headers`:

```
/*
  Link: </rss.xml>; rel="alternate"; type="application/rss+xml"; title="korbonits.com"
  Link: </sitemap-index.xml>; rel="sitemap"
```

Site-wide. The RSS feed and sitemap were already discoverable via HTML — this just makes them discoverable via headers too.

### 3. Markdown content negotiation

The biggest piece, and the one I bungled on the first try.

The scanner's "Markdown for Agents" check sends a request with `Accept: text/markdown` and expects a markdown response with `Content-Type: text/markdown` instead of the HTML you'd serve a browser. The idea: an AI agent doesn't want to parse your CSS-decorated HTML and re-extract the prose; it wants the raw markdown, ideally with frontmatter intact.

For an Astro site on Netlify, the cleanest implementation is:

1. **Build-time markdown variants** of every page that has a markdown source. New files at `src/pages/index.md.ts` (homepage) and `src/pages/blog/[slug].md.ts` (each published post). These are Astro endpoints that emit the post's frontmatter + body at `/index.md` or `/blog/<slug>.md` with `Content-Type: text/markdown`.
2. **A Netlify edge function** that intercepts requests to `/` and `/blog/<slug>/`, inspects the `Accept` header, and rewrites the request to the corresponding `.md` URL when `text/markdown` is preferred. Sets `Vary: Accept` on both branches so caches don't conflate the two representations.

The edge function is short. The interesting bit is the q-value comparison — if the agent sends `Accept: text/markdown, text/html;q=0.9`, the edge function should serve markdown; if it sends `Accept: */*` (the lazy default), the edge function should fall through to HTML so browsers don't get markdown.

```js
function prefersMarkdown(accept) {
  if (!accept) return false;
  const types = accept.split(',').map(s => {
    const [type, ...params] = s.trim().toLowerCase().split(';').map(p => p.trim());
    const qParam = params.find(p => p.startsWith('q='));
    const q = qParam ? parseFloat(qParam.slice(2)) : 1.0;
    return { type, q: Number.isFinite(q) ? q : 1.0 };
  });
  const md = types.find(t => t.type === 'text/markdown');
  if (!md) return false;
  const html = types.find(t => t.type === 'text/html');
  return !html || md.q >= html.q;
}
```

#### The mistake worth naming

My first pass shipped markdown negotiation only for `/blog/<slug>` URLs. I assumed the scanner would test a deep page with rich content. The score moved 25 → 42 instead of the predicted 58. Re-reading the scanner output, the failing check was `GET homepage (Accept: text/markdown)` — they probe the *homepage*, not a blog post.

This is a common shape: build the right thing for the wrong path. One extra commit added a homepage variant (`src/pages/index.md.ts`) and expanded the edge function path from `/blog/*` to also include `/`. Ten minutes of work, one extra deploy round-trip — wouldn't have happened if I'd read the scanner's per-check description more carefully the first time.

The full code lives in [the commit history of korbonits.github.io](https://github.com/korbonits/korbonits.github.io). The relevant files are `public/_headers`, `public/robots.txt`, `netlify.toml`, `netlify/edge-functions/markdown-negotiation.js`, `src/pages/index.md.ts`, and `src/pages/blog/[slug].md.ts`.

## What I skipped on purpose (and why)

The scanner reports five failing checks I did not address. Each one's a real spec. None of them apply to a personal blog:

- **API Catalog (RFC 9727)** — `/.well-known/api-catalog` advertises an OpenAPI spec, service docs, and a status endpoint. I don't have an API. I have a blog. Publishing an empty linkset to clear the check would be misleading.
- **OAuth metadata, two flavors** — `/.well-known/openid-configuration` (OIDC discovery) and `/.well-known/oauth-protected-resource` (RFC 9728). The first advertises an authorization server; the second describes a protected resource that needs one. I run neither. Same N/A for the same reason: nothing on the site requires authentication.
- **MCP Server Card (SEP-1649)** — `/.well-known/mcp/server-card.json` describes an MCP server you operate. I do not operate an MCP server. The blog is not a server.
- **Agent Skills index** — `/.well-known/agent-skills/index.json` lists callable skills. The blog has no skills to call. Posts are content, not actions.
- **WebMCP** — `navigator.modelContext.provideContext()` exposes JavaScript-callable tools to in-browser agents. The blog has no tools to expose. Reading is the entire interaction.

Plus the four commerce protocols (x402, MPP, UCP, ACP) — also N/A, also explicitly listed by the scanner as "informational; not affecting score."

The pattern across all five: they are gates for sites that **publish APIs, run protected services, or sell things.** A personal blog publishes content. A meaningful slice of the scanner's checks are simply not the right questions to ask of it.

This is the same lesson I wrote about [last week](/blog/2026-04-12-bulk-oss-contributions-ruff-and-ci): there's a difference between *correct* and *wanted*. The scanner gives you a checklist; clearing it without thinking is correctness theatre. The right move is to ship what your audience (in this case, AI agents reading prose) actually needs, and skip what they don't.

## The deeper point

A 25 → 50 score change reads modest. A Level 1 → Level 4 designation reads heroic. Both are accurate. The difference is what gets weighted: a high raw score requires you to publish things you don't have, and a high level just requires you to fully address the things that apply to you.

The scanner's design quietly endorses this. For a content site like this one, "Level 4: Agent-Integrated" appears to be awarded for clearing the applicable categories rather than for accumulating points across all of them. That's a smarter rubric than the headline number suggests, at least for the slice of the scanner that maps to a personal blog.

For a personal blog, "agent-ready" boils down to three additions: an explicit AI stance in robots.txt, response headers that point at your feed, and a markdown variant of every page agents are likely to read. Two hours of work, one trap to avoid (test the path the scanner actually probes), one deploy. Everything else is for sites you don't have.

---

*korbonits.com is my personal blog. I write about ML, software, and books.*

---

## TODO before publish

- [x] Screenshot of the final 50 / Level 4 scan result embedded above the opener
- [x] Vibe-coding tag kept (sits alongside the curriculum posts; matches the read-scan / ship-three-files / re-scan / write-up shape)
- [x] Pre-publish pass (2026-04-30):
  - Title count fixed: collapsed the two OAuth bullets into one ("OAuth metadata, two flavors"), so the body now lists the five it claims in the title
  - Updated the "all six" callback in the same section to "all five" + softened "half the scanner's checks" to "a meaningful slice"
  - Softened the Level claim: clarified that 2 categories cleared took the score to Level 2 and the third jumped it to Level 4 (Level 3 was skipped, presumably reserved for sites that clear at least one of the API/Auth checks)
  - Scoped the Level 4 framing: "for a content site like this one, *appears* to be awarded for clearing applicable categories" rather than asserting it as a general rule
  - Added the standard footer line for consistency with the bulk-OSS and SMT posts
  - Verified the `prefersMarkdown` snippet matches the live edge function semantically (the post version uses one `.map` instead of two for readability — same logic)
- [ ] Optional: also screenshot the original 25 / Level 1 result for a "before" image.  This is harder to reproduce; if no screenshot exists, the prose-only version stands fine.
- [ ] **Follow-up (after publish): package the pattern as `astro-markdown-for-agents`.**  The build-time `.md` endpoint + edge-function negotiation are reusable across any Astro/Netlify site.  Shape: a one-line `npm install`, an Astro integration that auto-registers the `.md` endpoints for any content collection (defaulting to `blog`), and a generated edge function the user drops into `netlify/edge-functions/`.  Stretch: an adapter for Cloudflare Pages.  If shipped, this post gets a link to the package and a one-line "I extracted the pattern from this post into" callout.
