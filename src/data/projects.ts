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
    name: "Priorly.ai",
    slug: "priorly",
    description:
      "Trademark visual similarity search using CLIP and DINOv2 embeddings with pgvector. Built to surface visually conflicting marks that text-based search misses.",
    date: "2026-03-28",
    tags: ["startup", "ml"],
    status: "coming-soon",
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
    name: "VIBE Token",
    slug: "vibe-token",
    description:
      "An ERC-20 token deployed on Ethereum mainnet, written in Solidity from scratch with no frameworks. A first-principles exercise in how Ethereum actually works, documented in public.",
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
      "Personal site and writing hub. Built with Astro, deployed on Netlify, with blog analytics via Supabase and Netlify Functions.",
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
