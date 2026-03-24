---
title: "One GitHub Account, Many Domains: Deploying Family Sites with GitHub Pages"
date: 2026-03-23
description: "How I spun up four placeholder sites for family domains in under an hour using GitHub Pages, Astro, and GitHub Actions — all from a single account."
tags: ["meta", "blogging", "astro", "web", "github", "dns"]
draft: false
---

I bought a handful of domains for my family. Nothing fancy — just wanted to park them somewhere clean while I figure out what to do with them. I assumed I'd need separate GitHub accounts for each site. Turns out I didn't.

Here's how it works and how I set it up in an evening.

## One account, many sites

GitHub only gives you one user site per account: `username.github.io`. But you can have unlimited *project* sites — one per repo. Each repo gets its own GitHub Pages deployment, and each can have its own custom domain via a `CNAME` file.

So the setup is:
- One repo per domain (e.g. `korbonits/sohaili.org`)
- A `CNAME` file in `public/` containing the domain name
- DNS A records pointing to GitHub's IPs
- GitHub Actions to build and deploy on every push to `main`

## The stack

I used the same stack as this blog: [Astro](https://astro.build) with a minimal `package.json` and no extra dependencies. For placeholder sites, the whole thing is a single `index.astro` file — just a centered "coming soon" page with the same Atkinson font and color scheme as korbonits.com.

## The workflow

Each repo gets this GitHub Actions workflow:

```yaml
name: Deploy to GitHub Pages

on:
  push:
    branches: [main]
  workflow_dispatch:

permissions:
  contents: read
  pages: write
  id-token: write

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-node@v4
        with:
          node-version: 22
      - uses: actions/configure-pages@v4
      - run: npm ci
      - run: npm run build
      - uses: actions/upload-pages-artifact@v3
        with:
          path: dist

  deploy:
    needs: build
    runs-on: ubuntu-latest
    environment:
      name: github-pages
      url: ${{ steps.deployment.outputs.page_url }}
    steps:
      - id: deployment
        uses: actions/deploy-pages@v4
```

Push to `main`, GitHub builds and deploys automatically. No servers, no hosting fees.

## DNS

At your registrar, add these A records for the root domain and a CNAME for `www`:

```
A     @    185.199.108.153
A     @    185.199.109.153
A     @    185.199.110.153
A     @    185.199.111.153
CNAME www  korbonits.github.io
```

One gotcha: after DNS propagates, you also need to set the custom domain in the GitHub repo under Settings → Pages. The `CNAME` file handles the routing, but GitHub needs to know about it on their end too. Once you do that, GitHub provisions the HTTPS cert automatically.

## A few gotchas

**Enable GitHub Actions as the Pages source.** In each repo, go to Settings → Pages → Source and select "GitHub Actions". It defaults to branch-based deployment which won't work with this workflow.

**Allow `main` in the environment.** GitHub sometimes restricts which branches can deploy to the `github-pages` environment. If you see "Branch is not allowed to deploy", go to Settings → Environments → github-pages and add `main` to the allowed branches.

**Don't copy `node_modules` between repos.** I scaffolded new repos by copying an existing one. The `node_modules` came along and got corrupted. Always run `npm install` fresh in each new repo.

**Commit a `package-lock.json`.** The workflow uses `npm ci`, which requires a lockfile. Generate it locally with `npm install` and commit it before pushing.

## The result

Four domains, four repos, one GitHub account, zero hosting costs. Each site is a clean "coming soon" page that I can build out whenever I'm ready. When I do, it's just Astro — I can add content, components, or a full blog without changing any of the infrastructure.

The whole process took about an hour, most of which was DNS propagation and figuring out the GitHub Pages environment settings.
