// scripts/build-embeddings.js
// Runs before `astro build` in Netlify to generate semantic embeddings for all posts.
// Writes public/embeddings.json — served statically, loaded by the search page.
//
// Requires: MODAL_EMBED_URL set in Netlify environment variables.
// If unset, exits cleanly so local dev builds are unaffected.

import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const BLOG_DIR = path.join(__dirname, "../src/content/blog");
const OUT_FILE = path.join(__dirname, "../public/embeddings.json");
// MODAL_EMBED_URL should now be the base URL (e.g. https://korbonits--korbonits-embed-api.modal.run)
// For backwards compat with the old per-function URL, strip a trailing /embed if present.
const MODAL_BASE = process.env.MODAL_EMBED_URL?.replace(/\/embed$/, "");

if (!MODAL_BASE) {
  console.warn("[embeddings] MODAL_EMBED_URL not set — skipping. Search will use existing embeddings.json if present.");
  process.exit(0);
}

const MODAL_URL = `${MODAL_BASE}/embed`;

function parseFrontmatter(raw) {
  const match = raw.match(/^---\r?\n([\s\S]*?)\r?\n---\r?\n([\s\S]*)$/);
  if (!match) return { meta: {}, body: raw };
  const meta = {};
  for (const line of match[1].split(/\r?\n/)) {
    const colon = line.indexOf(":");
    if (colon === -1) continue;
    const key = line.slice(0, colon).trim();
    const val = line.slice(colon + 1).trim().replace(/^["']|["']$/g, "");
    meta[key] = val;
  }
  return { meta, body: match[2] };
}

function stripMarkdown(md) {
  return md
    .replace(/```[\s\S]*?```/g, "")              // fenced code blocks
    .replace(/`[^`]+`/g, "")                      // inline code
    .replace(/#{1,6}\s+/g, "")                    // headings
    .replace(/[*_]{1,2}([^*_\n]+)[*_]{1,2}/g, "$1") // bold / italic
    .replace(/\[([^\]]+)\]\([^)]+\)/g, "$1")      // links → label
    .replace(/!\[[^\]]*\]\([^)]+\)/g, "")         // images
    .replace(/^>\s+/gm, "")                        // blockquotes
    .replace(/\n{3,}/g, "\n\n")
    .trim();
}

const files = fs
  .readdirSync(BLOG_DIR)
  .filter((f) => f.endsWith(".md") || f.endsWith(".mdx"));

const posts = files
  .map((file) => {
    const raw = fs.readFileSync(path.join(BLOG_DIR, file), "utf-8");
    const { meta, body } = parseFrontmatter(raw);
    if (meta.draft === "true") return null;
    const slug = file.replace(/\.mdx?$/, "");
    const title = meta.title || slug;
    const stripped = stripMarkdown(body);
    const excerpt = stripped.slice(0, 400).replace(/\s+/g, " ");
    return { slug, title, url: `/blog/${slug}/`, excerpt, embedText: `${title}. ${excerpt}` };
  })
  .filter(Boolean);

console.log(`[embeddings] Embedding ${posts.length} posts via Modal…`);

const res = await fetch(MODAL_URL, {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({ texts: posts.map((p) => p.embedText) }),
});

if (!res.ok) {
  const err = await res.text();
  console.error("[embeddings] Modal request failed:", err);
  process.exit(1);
}

const { embeddings } = await res.json();

// Project embeddings to 2D with PCA for the scatter plot
console.log(`[embeddings] Projecting to 2D via Modal…`);
const projRes = await fetch(`${MODAL_BASE}/project`, {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({ embeddings }),
});

if (!projRes.ok) {
  console.error("[embeddings] PCA projection failed:", await projRes.text());
  process.exit(1);
}

const { coords, mean, components } = await projRes.json();

const output = {
  pca: { mean, components },
  posts: posts.map((p, i) => ({
    slug: p.slug,
    title: p.title,
    url: p.url,
    excerpt: p.excerpt,
    embedding: embeddings[i],
    x: coords[i][0],
    y: coords[i][1],
  })),
};

fs.writeFileSync(OUT_FILE, JSON.stringify(output));
console.log(`[embeddings] Wrote ${output.posts.length} entries → public/embeddings.json`);
