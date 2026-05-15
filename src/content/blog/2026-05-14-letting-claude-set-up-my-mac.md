---
title: "Letting Claude Set Up My Mac: Six Things It Couldn't Decide"
date: 2026-05-14
draft: true
description: "I pointed Claude Code at a fresh MacBook and asked what was missing from my dev setup. Two hours later the box was reproducible, the dotfiles were in a private repo, and SSH-signed commits were verifying end-to-end. But the part worth writing about is the half-dozen moments where the AI either asked, guessed wrong, or hit a wall — and execution stopped while judgment did the work."
tags:
  - dev-environment
  - dotfiles
  - claude
  - ai-tools
  - mac
---

I unboxed a new MacBook this week. The setup ritual most engineers know — install Homebrew, copy a Brewfile, generate keys, configure git — is the kind of task that's tedious but rarely interesting. I'd put together a draft `Brewfile` and `.zshrc` the night before. The next morning I opened Claude Code and typed:

> analyze my new dev environment setup. what am i missing

What followed was roughly two hours of work that ended with a private [dotfiles repo](https://github.com/korbonits/dotfiles), a `bootstrap.sh` that builds the box from scratch, SSH-signed commits verifying with green badges on GitHub, and the Astro/pnpm path confirmed end-to-end at [korbonits.org](https://korbonits.org).

Most of it was boring. Modern Rust-based CLI tools, `mise` for runtime versions, `git-delta` as the diff pager, `zsh-autosuggestions` + `zsh-syntax-highlighting` loaded in the right order, a stow-based dotfiles layout — none of that needs me. Claude wrote it, I confirmed it, we moved on.

The interesting part is the six places execution stopped.

## 1. Where the AI was wrong about something it couldn't know

I have two domains: `korbonits.org` and `korbonits.com`. Both have a GitHub repo with a confusingly similar name (`korbonits.org` and `korbonits.github.io` respectively). When Claude was helping me migrate the `korbonits.org` repo to pnpm and rename `master` → `main`, it inspected the `korbonits.github.io` repo's Pages config and reported it was in an "errored" state. Then it suggested:

> Want me to disable GitHub Pages on `korbonits.github.io` to clear the red X?

The recommendation was based on a complete-looking picture: `korbonits.org` repo serves korbonits.org, `korbonits.github.io` is a stale user-site repo doing nothing, kill it. The picture was wrong. `korbonits.github.io` is the source repo for korbonits.com — but the deployment goes through Netlify, not Pages. That fact isn't visible anywhere Claude looked. There's no `netlify.toml` field that says "this is the canonical deploy target"; there's no annotation on the GH Pages settings saying "don't worry about this, we use Netlify."

I typed: *"korbonits.github.io points to netlify to serve korbonits.com"* and the model immediately recontextualized. The "errored" status wasn't a problem to fix, it was a vestigial config. We changed the GH Pages build type from `legacy` (Jekyll) to `workflow` to stop the auto-build attempts, and left everything else alone.

The interesting thing isn't that Claude got it wrong. Of course it got it wrong; there was no way to be right. The interesting thing is the *shape* of the wrongness. A confident-sounding recommendation built on an incomplete picture of the world. The corrective signal is *me*, telling the model about a deployment path it can't see. This is what people mean when they say "AI can't replace context." The context isn't in the codebase; it's in my head.

## 2. Where renaming a branch broke a coupling neither of us tracked

We renamed `master` → `main` on the `korbonits.org` repo. Updated `.github/workflows/deploy.yml` to trigger on `main`. Pushed. CI built fine. Deploy step failed:

> Branch "main" is not allowed to deploy to github-pages due to environment protection rules.

GitHub Pages stores a deployment branch policy on the `github-pages` environment, separate from the workflow trigger and separate from the repository's default branch. When the repo was first created, GitHub auto-created a policy allowing `master`. Renaming the branch didn't propagate. The default-branch change didn't propagate. The repo settings UI doesn't surface this until your deploy fails.

Claude didn't know to check for this either — it didn't surface in my Pages plan, I didn't ask, and the failure message in CI was what triggered the fix. Three API calls:

```sh
gh api -X DELETE repos/korbonits/korbonits.org/environments/github-pages/deployment-branch-policies/45490773
gh api -X POST   repos/korbonits/korbonits.org/environments/github-pages/deployment-branch-policies -f name=main
gh run rerun 25901906628 --failed
```

This isn't a Claude failure mode — it's a GitHub failure mode. But the relevant point for AI-assisted ops is that the model can't pre-empt invisible coupling. The way you find it is by trying the thing and reading the error. That's the loop. The model is excellent inside the loop and useless before it.

## 3. Where the AI knew enough to ask, not enough to decide

Three formulae in my `Brewfile` were dead. Two were straightforward: `tap "homebrew/bundle"` had been deprecated into core, and `linear-linear` had been renamed to `linear`. Claude fixed both without asking.

The third was `terraform` — HashiCorp relicensed it under BSL in 2023, Homebrew dropped the formula, and there's no obvious default. Claude asked:

- Switch to OpenTofu (Linux Foundation fork, MPL-licensed, 99% drop-in compatible)
- Keep HashiCorp Terraform via `hashicorp/tap`
- Drop terraform entirely — install per-project when needed

The model could have picked any of the three and justified it. Instead, it asked. And it asked because the right answer depends on something not present in any file: whether I have existing Terraform Cloud workflows, whether my employer pins a specific version, whether I plan to use Terraform at all on this box. I haven't used Terraform in years on personal projects. I dropped it.

This is the pattern I want more of: the model recognizing that the decision is upstream of the code, and routing around to ask. It cost me ten seconds of reading three radio buttons.

## 4. Where the AI's own guardrails fired

I asked Claude to write a user-level `CLAUDE.md` — a cross-project preferences file that loads into every Claude Code session. The write was denied:

> Permission for this action was denied by the Claude Code auto mode classifier. Reason: Writing ~/.claude/CLAUDE.md is Self-Modification of agent config without explicit user authorization for that specific file.

This is Anthropic's own guardrail: Auto Mode (the permission system) is designed to refuse certain classes of writes — like modifying agent configuration — without an explicit per-file authorization. The classifier interpreted "write a file in `~/.claude/`" as agent self-modification. Which, technically, is what it is.

The fix was to disable Auto Mode and explicitly authorize the write. But the moment is interesting: the *agent that wanted to help me* was blocked by *the agent platform's policy*. Not a bug — a deliberate, conservative default. I'd rather have it block-then-ask than write quietly. Especially when the file in question shapes every future Claude session's behavior.

If you're building an agent platform: this is the kind of friction worth keeping. The cost of an extra prompt is small. The cost of an agent silently rewriting its own constitution is large.

## 5. Where the tooling itself failed in an agent context

I needed to log into Hugging Face. I typed:

```
! hf auth login
```

The `!` prefix runs the command in my shell so output lands in this session — useful for interactive flows. `hf auth login` immediately failed:

```
GetPassWarning: Can not control echo on the terminal.
Warning: Password input may be echoed.
Enter your token (input will not be visible):
Aborted!
```

The CLI tries to read a hidden token from stdin. There's no TTY available, so the hidden-input fallback can't run, and the read aborts. So I did what an impatient human does:

> here is the token: hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

…and pasted the token into the chat. Which means the token is now in the conversation transcript, and Anthropic (and any model serving that conversation) sees it.

This is a UX failure with a generalizable shape: CLI tools designed for interactive humans don't degrade gracefully when the "shell" is an LLM context window. The right answer is to use the non-interactive flag (`hf auth login --token <token>`) — which Claude immediately did once I pasted the token — but the natural human action was to paste. And the token leaked.

Lesson, both for me and for anyone reading this who's about to do the same: when an interactive CLI prompts you in an agent-mediated shell, **do not paste the secret**. Either run it in a real terminal outside the agent session, or use the non-interactive form from the start. The token I leaked was read-only and has been revoked — but the failure mode generalizes to every CLI that does hidden-input prompts.

## 6. Where I almost shipped a small leak myself

The same risk pattern showed up an hour earlier in a way I caught only because Claude flagged it. After linking the local Netlify project, the command was:

```sh
netlify env:list --plain
```

Which dumped the production environment variables — keys *and* values — straight into the conversation. Anthropic API key, analytics secret, basic-auth password. Nothing dramatic; the site is small. But the values are now in a chat transcript I can't fully control.

Claude wrote, immediately:

> **Quick heads-up:** I used `--plain` which dumped the actual values into this transcript. Three of them are real secrets you may want to rotate now that they've been in a chat context.

The model recognized the leak in the same response that caused it. It would have been better not to use `--plain` in the first place — `netlify env:list` without `--plain` returns masked values, and `netlify env:pull` writes a `.env` file without ever putting secrets through stdout. The model's reflex to use `--plain` because it was about to also write the values to a file was correct in intent and wrong in execution.

Two takeaways. One, the model is good at self-auditing — it caught its own mistake. Two, "good at self-auditing" is not the same as "incapable of making the mistake." The audit step came after the damage. For high-stakes secrets — production API keys, customer data, signing keys — *the agent should not be the last line of defense*. Use a CLI flow that can't leak by design.

## What stayed mine

Here's the list of things I had to decide during a routine dev-environment setup that an AI couldn't decide for me:

- Whether `korbonits.github.io` is dead or alive (it's alive, just elsewhere)
- Whether to keep Terraform, switch to OpenTofu, or drop it
- Whether to make the dotfiles repo public or private (private — it enumerates my projects)
- Whether to rename `master` → `main` while we were renaming things (yes)
- Whether to migrate `vibe-token` from Hardhat to Foundry to match my CLAUDE.md preference (no, the suite works, Hardhat ecosystem is fine, defer)
- Whether to rotate the leaked HF token immediately (yes, did) and the dumped Netlify secrets (held off, low-risk)
- Whether the work was worth a blog post (you're reading the answer)

None of these are technical decisions in any meaningful sense. They're product decisions, taste decisions, judgment calls about cost and timing. The model is better than I am at most of the *technical* work in this list — installing tools, writing workflows, debugging branch protection rules. It is not better than I am at deciding which of those technical actions to take, in what order, with what tradeoffs.

This is the same observation I made in [I Shipped a Feature From My Phone During Lunch](/blog/2026-03-25-i-shipped-a-feature-from-my-phone-during-lunch) and [19 Open Source Pull Requests in One Afternoon](/blog/2026-04-12-bulk-oss-contributions-ruff-and-ci): execution is getting cheaper, judgment isn't. Each post finds new evidence for the same thesis. Setting up a fresh dev box used to take a Saturday morning. Today it takes a couple of conversational hours, with six pauses for judgment in the middle. Those six pauses are where the value of being there at all still lives.

The dotfiles are at [github.com/korbonits/dotfiles](https://github.com/korbonits/dotfiles). The `bootstrap.sh` will install Homebrew, run `brew bundle`, stow the dotfiles into `$HOME`, install Rust stable, and seed mise globals — on any fresh Mac, idempotently. The judgment is not in the script.

---

*korbonits.com is my personal blog. I write about ML, software, and books.*
