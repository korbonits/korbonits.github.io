// src/data/projects.ts
// Add new projects here — newest first.
// The /projects page renders directly from this array.

export type ProjectTag =
  | "startup"
  | "ml"
  | "oss"
  | "agents"
  | "personal-brand";

export type ProjectStatus = "active" | "coming-soon" | "archived";

export interface Project {
  name: string;
  slug: string;              // used for aria labels and future detail pages
  description: string;       // 1–2 sentences max
  date: string;              // ISO format, used for sort — displayed as "Month YYYY"
  tags: ProjectTag[];
  status: ProjectStatus;
  links: {
    primary?: { label: string; url: string };   // main CTA (landing page, hosted tool)
    repo?: string;                               // GitHub URL if public
  };
}

export const projects: Project[] = [
  {
    name: "Sheaf",
    slug: "sheaf",
    description:
      "vLLM solved inference for text LLMs. The same gap exists for every other class of foundation model — time series, tabular, molecular, and more. Sheaf fills it: a unified serving layer with standard batching contracts across model classes.",
    date: "2026-04-14",
    tags: ["ml", "oss"],
    status: "active",
    links: {
      primary: { label: "Read the post", url: "/blog/2026-04-14-sheaf-vllm-for-non-text-foundation-models" },
      repo: "https://github.com/korbonits/sheaf",
    },
  },
  {
    name: "Priorly.ai",
    slug: "priorly",
    description:
      "Text-based trademark search misses visually conflicting marks — so I built one that doesn't. Uses CLIP and DINOv2 embeddings with pgvector to surface conflicts by appearance, not just name.",
    date: "2026-03-28",
    tags: ["startup", "ml"],
    status: "active",
    links: {
      primary: { label: "Visit site", url: "https://priorly.ai" },
    },
  },
  // {
  //   name: "Vanna",
  //   slug: "vanna",
  //   description:
  //     "Cross-venue risk infrastructure for DeFi options. Unified Greeks, cross-venue margin, and automated hedging for options desks running books across Derive, Aevo, and Hyperliquid.",
  //   date: "2026-03-27",
  //   tags: ["startup"],
  //   status: "coming-soon",
  //   links: {
  //     primary: { label: "Visit site", url: "https://vanna.fi" },
  //   },
  // },
  {
    name: "Semantic Blog Search",
    slug: "semantic-blog-search",
    description:
      "Keyword search misses posts that use different words for the same idea. This uses nomic-embed-text-v1.5 on Modal to embed all posts at build time, then runs cosine similarity in the browser at query time — no search index, no database.",
    date: "2026-04-05",
    tags: ["ml", "oss"],
    status: "active",
    links: {
      primary: { label: "Try it", url: "/search" },
    },
  },
  {
    name: "Embedding Similarity",
    slug: "embedding-similarity",
    description:
      "Paste any two snippets and see how a real embedding model (nomic-embed-text-v1.5 via Modal) and Claude's judgment compare — they often disagree, and the gap is where the interesting explanation lives.",
    date: "2026-04-04",
    tags: ["ml", "agents"],
    status: "active",
    links: {
      primary: { label: "Try it", url: "/tools/embedding-similarity" },
    },
  },
  {
    name: "Now Page — Living Data",
    slug: "now-page",
    description:
      "Most /now pages go stale within weeks. This one stays current automatically — a nightly GitHub Actions cron triggers a Netlify rebuild that pulls live GitHub activity and Goodreads data at build time, no database needed.",
    date: "2026-03-25",
    tags: ["oss", "personal-brand"],
    status: "active",
    links: {
      primary: { label: "View now page", url: "/now" },
    },
  },
  {
    name: "Blog Analytics Dashboard",
    slug: "blog-analytics",
    description:
      "I wanted visibility into what people actually read without handing data to a third party. Built a Netlify serverless function that captures page views into Supabase, with a Chart.js dashboard behind basic auth.",
    date: "2026-03-25",
    tags: ["oss"],
    status: "active",
    links: {
      primary: { label: "View dashboard", url: "/analytics" },
    },
  },
  {
    name: "VIBE Token",
    slug: "vibe-token",
    description:
      "Most people interact with blockchains through abstractions. I wanted to understand Ethereum from the ground up — so I wrote an ERC-20 token in Solidity from scratch, no frameworks, deployed to mainnet, and documented it publicly.",
    date: "2026-03-26",
    tags: ["oss"],
    status: "archived",
    links: {
      primary: { label: "Visit site", url: "https://thevibetoken.xyz" },
      repo: "https://github.com/korbonits/vibe-token",
    },
  },
  {
    name: "korbonits.com",
    slug: "korbonits-com",
    description:
      "A home for writing and thinking in public — built to own the stack end to end. Astro + Netlify with self-hosted analytics, a living /now page, and no dependencies on third-party platforms.",
    date: "2015-03-01",
    tags: ["personal-brand"],
    status: "active",
    links: {
      primary: { label: "You're here", url: "https://korbonits.com" },
      repo: "https://github.com/korbonits/korbonits.com",
    },
  },
  // ── Placeholder: swap in your first Claude artifact when it ships ──
  // {
  //   name: "Your artifact name",
  //   slug: "artifact-slug",
  //   description: "One line description.",
  //   date: "2025-MM-DD",
  //   tags: ["agents"],
  //   status: "active",
  //   links: {
  //     primary: { label: "Try it", url: "https://..." },
  //   },
  // },
];
