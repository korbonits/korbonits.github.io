---
title: Adding a CMS to My Astro Blog (and the OAuth Rabbit Hole That Followed)
date: 2026-03-24
description: How I added Decap CMS to my static Astro site, and the authentication debugging that followed.
tags: [astro, cms, decap-cms, netlify, web]
---

I recently migrated this blog from Jekyll to Astro, which I wrote about [here](/blog/2026-03-16-migrating-from-jekyll-to-astro). The content model is simple: markdown files with YAML frontmatter in `src/content/blog/`. Writing a post means opening a terminal, creating a file, and pushing to GitHub. Fine for code — not great when I want to jot something down on my phone or hand off editing to someone else.

So I added a CMS layer.

## The choice: Decap CMS

Decap CMS (formerly Netlify CMS) is a git-based CMS that stores content directly in your repo as markdown files. There's no database, no separate content API, no schema migration — just a web UI that commits to GitHub on your behalf. For a static Astro site, this is a perfect fit: the CMS speaks the same language as the content collections I already had.

The admin panel is a single HTML page that loads from CDN, so adding it was literally two files: `public/admin/index.html` and `public/admin/config.yml`. The config maps directly to my Astro content schema — the same fields I had in `content.config.ts` just described in YAML:

```yaml
collections:
  - name: blog
    folder: src/content/blog
    slug: "{{year}}-{{month}}-{{day}}-{{slug}}"
    fields:
      - { label: Title, name: title, widget: string }
      - { label: Date, name: date, widget: datetime }
      - { label: Tags, name: tags, widget: list }
      - { label: Body, name: body, widget: markdown }
```

This is config-driven architecture in a small but concrete form: the behavior of the system is described in a file, not hardcoded. Changing which fields appear in the editor is a YAML edit, not a code change.

## The OAuth rabbit hole

The tricky part is authentication. The CMS needs to commit to GitHub on your behalf, which requires OAuth. For a static site with no server, you need an OAuth relay — a server that exchanges the GitHub auth code for a token. The standard approach is to use Netlify's relay even if your site isn't hosted on Netlify.

This is where things got interesting. I set up a GitHub OAuth App, pointed it at Netlify's callback URL, configured a blank Netlify site as the relay, and got: `bad_verification_code`. Every time.

After debugging the callback URL, regenerating the client secret twice, and confirming the credentials were correct, I found the issue by accident: the CMS worked fine when accessed from `beamish-cassata-fc006d.netlify.app/admin/` but failed from `korbonits.com/admin/`. Netlify's OAuth relay only trusts requests from domains it recognizes.

The fix: add `korbonits.com` as a custom domain on the Netlify site. One step in the Netlify dashboard. That was it.

## The result

`korbonits.com/admin/` now loads a full CMS with my blog posts, the now page, and the about page all editable from the browser. Writing this post required no terminal.
