---
title: "Planning an ML Survey Paper: Notes from a Notebook"
date: 2026-04-30
draft: false
description: "An operational artifact, not a polished essay. The numbered checklist I worked out across three notebook entries in April 2024 — what venue, what topic, how to find the bibliography, solo or co-author, milestones — when I was deciding whether to write an ML survey paper. Useful for anyone in a similar spot."
tags:
  - machine-learning
  - writing
  - academic
  - personal
  - notebook
---

![Notebook page on writing a survey paper, with the numbered process and small sketches of papers and journals, April 8 2024](/images/notebook-survey-7036.jpg)

*The middle of three sittings on the same question. The little sketches in the bottom right — a stack of papers, a process diagram, a "Profit?" label — are the writer trying to convert "I should write a survey" into a workflow. The post is the workflow.*

## I started it in Overleaf, missed the internal deadline, and got pulled into a transfer search

That's the honest disclosure to lead with. Across April 2024 I spent three notebook sittings — five days apart — working out how I'd write an ML survey paper. The plan was to submit it to AMLC, Amazon's internal ML conference. I started a draft in Overleaf. I missed the internal submission deadline. And then return-to-team kicked in: I had to spend my discretionary cycles finding an internal transfer rather than finishing the paper. Between the missed deadline and the org pressure to land somewhere new, the survey stopped.

That's a specific failure mode, and it's worth naming because a lot of people who plan papers will recognize at least one half of it:

- The planning was real (three notebook sittings, an outline, a venue).
- The drafting was real (Overleaf project, opening sections written).
- The deadline was real and external (a fixed AMLC submission window).
- The competing priority was real and non-negotiable (return-to-team meant the alternative to "find a transfer" wasn't "write a survey," it was "be without a team").
- The discretionary work lost. As discretionary work usually does when something non-discretionary lands on top of it.

What follows is the planning artifact — the numbered lists I worked out, the venue research I did *after* (not before) committing to a topic, and what I'd do differently if I started over. If you're in the same decision phase, this might save you a notebook of your own. If you're looking for a how-to from someone who shipped a survey, you're in the wrong post.

## The sequence (across three sittings)

### Sitting 1 (2024-04-05): What topic?

> *Many interesting papers, projects: Geodesic RAG, intrinsic dim/MoE, Mistral mixture, AR + LLMs. … After I conceived of it, I realized it's still possible to need to pick something embedding-specific and run with it. AND LLMs + AR feels very exciting.*

The trap: scope creep at topic-pick time. Write down five candidates, force a ranking, pick one, *commit before you start the lit review.* The lit review will tell you whether the topic is real, but it can't pick the topic for you.

### Sitting 2 (2024-04-08): What venue?

> *ML journals do exist — JMLR, JAIR, IEEE Transactions … This should provide additional confidence venues / targets of journal sources / publication.*

The five-step plan that emerged:

1. **Topic.**
2. **Submission guidelines.** Read the venue's actual instructions before you write a single sentence.
3. **"Survey & Pray."** Write the thing.
4. *(blank)*
5. **Profit.**

(The blank step 4 is in the original notebook, deliberately left there. The honest step 4 is "submit and revise indefinitely," which the writer was clearly avoiding.)

A side-question that the notebook flags but doesn't resolve: **solo or co-author?** Both are legitimate; the answer probably depends on whether you're writing a survey to *establish a position* (solo, slower, more credit) or to *consolidate a community* (co-authored, faster, more political).

### Sitting 3 (2024-04-10): Refined process

> *Heard back from the AAAS suggest, like a survey/review that should be as long as standard papers (max 8 pages) and published.*

The numbered list got tighter:

1. **Subject.** A literature review of work concepts.
2. **Bibliography first.** Use citations of recent papers in your area to find the actual surface area.
3. **Determine the action focus.** "I know what the conclusion is" — write toward it, don't discover it during writing.
4. **Identify other authors / writers to cite as the abstract.**
5. **Etc.**
6. **Aim milestones.** Plan to about — when? (Originally blank.)

## What the venues actually want (verified)

The notebook listed JMLR, JAIR, and IEEE Transactions as candidates without checking what they said about surveys.  Now checked:

- **JMLR** ([author info](https://www.jmlr.org/author-info.html); [submission portal](https://jmlr.csail.mit.edu/manudb/)).  Direct quote from their guidelines: *"JMLR occasionally publishes surveys by invitation from the editorial board; we do not consider unsolicited survey papers."*  This is the single most important fact the notebook missed — JMLR is **not** a venue for a self-initiated survey.  Cross JMLR off the list unless you can get an invitation first.
- **JAIR** ([about/submissions](https://www.jair.org/index.php/jair/about/submissions); [submission wizard](https://jair.org/index.php/jair/submission/wizard)).  Accepts surveys, but with teeth: *"Surveys that merely summarise a selection of existing literature without providing new insights or conceptual contributions"* face a "high probability of summary rejection."  In practice this means the survey needs an analytical framework or a re-organization of the field, not just a literature scan.
- **IEEE TPAMI** ([author info via CSDL](https://www.computer.org/csdl/journal/tp/write-for-us/author)).  Accepts both regular and survey papers.  Survey papers typically run longer and need pre-clearance with the editor; the canonical guidance lives behind the IEEE Computer Society portal which renders client-side, so verify the current page limits and category names directly before submitting.

The bigger lesson: the venue's stance on *unsolicited* surveys is the gating constraint, and you find that out by reading their author info page, not by looking at what they've published.  JMLR has a beautiful catalog of surveys, all by invitation.  A new author has functionally zero shot.

## What I'd do differently if I started today

With the benefit of two years of hindsight (the missed AMLC deadline, the venue research the notebook didn't do, the org turbulence that killed the project), here's the plan I'd run instead:

1. **Pick the framework before the topic.** The notebook treated "subject" as step 1 and bibliography as step 2.  JAIR's guidelines explicitly invert that: a survey without an analytical framework or conceptual contribution gets summary-rejected.  The framework *is* the contribution.  The topic is just where you apply it.
2. **Write the position essay first.**  If the framework isn't crisp enough to defend in 2,000 words on a blog, it's not crisp enough to carry a 30-page survey.  The blog draft costs a weekend; the survey costs months.  Doing them in that order means you find out cheaply.
3. **Cross JMLR off the list on day one.**  Invitation-only.  The catalog is gorgeous and irrelevant for an unsolicited submission.  Real candidate venues for a first survey: JAIR (with a defensible framework), IEEE TPAMI (with editor pre-clearance), or a strong workshop or conference survey track.
4. **Decide solo-or-co-author before the bibliography, not after.**  Solo establishes a position; co-authored consolidates a community.  These are different problems with different bibliographies, different review styles, and different venues.  Picking later means redoing work.
5. **Don't scope a side-project deadline against an internal-org deadline you don't control.**  AMLC was a real submission window; the return-to-team timeline was a non-discretionary one that landed on top of it.  If I'd started six weeks earlier — *or* picked an external venue with a flexible timeline (JAIR has rolling submissions) — the org turbulence wouldn't have killed the paper.  Don't put a discretionary side-project on the same critical path as a non-discretionary work obligation, ever.

The pattern across all five is the same: *commit to the constraints early so the discretionary work is robust to the non-discretionary work that will inevitably show up.*  The original notebook plan was none of this.  It was "topic, then bibliography, then submission."  That order works for someone with a research group and a clear runway.  For someone planning a side-project survey on personal time, it's exactly backwards.

---

*korbonits.com is my personal blog. I write about ML, software, and books.*
