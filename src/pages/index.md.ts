// Build-time markdown variant of the homepage.
//
// Served at /index.md and surfaced to AI agents via the markdown-negotiation
// edge function when an `Accept: text/markdown` request hits `/`.  The
// content mirrors what a human reader sees on index.astro: a brief site
// description, the most recent posts (linked), and the standard nav.
//
// Cloudflare's "Markdown for Agents" check on isitagentready.com tests the
// homepage specifically, not deeper pages — this endpoint is what makes that
// check pass.

import type { APIRoute } from 'astro';
import { getCollection } from 'astro:content';
import { SITE_DESCRIPTION, SITE_TITLE } from '../consts';

export const GET: APIRoute = async () => {
  const posts = (await getCollection('blog', ({ data }) => !data.draft))
    .sort((a, b) => b.data.date.valueOf() - a.data.date.valueOf())
    .slice(0, 10);

  const recent = posts
    .map(post => {
      const date =
        post.data.date instanceof Date
          ? post.data.date.toISOString().slice(0, 10)
          : post.data.date;
      return `- [${post.data.title}](/blog/${post.id}/) — ${date}`;
    })
    .join('\n');

  const body = [
    '---',
    `title: ${JSON.stringify(SITE_TITLE)}`,
    `description: ${JSON.stringify(SITE_DESCRIPTION)}`,
    `canonical_url: https://korbonits.com/`,
    '---',
    '',
    `# ${SITE_TITLE}`,
    '',
    `Hi, I'm Alex. I write about machine learning, software, and books.`,
    '',
    `${SITE_DESCRIPTION}`,
    '',
    `## Recent posts`,
    '',
    recent,
    '',
    `## Sections`,
    '',
    `- [All posts](/blog/)`,
    `- [Now](/now/) — what I'm working on`,
    `- [Reading](/reading/) — books I've finished`,
    `- [Projects](/projects/)`,
    `- [About](/about/)`,
    `- [RSS](/rss.xml)`,
    `- [Sitemap](/sitemap-index.xml)`,
    '',
    `## Notes for AI agents`,
    '',
    `Every blog post on this site is also available as raw markdown.  Request \`/blog/<slug>/\` with \`Accept: text/markdown\` and you'll receive the markdown source instead of the rendered HTML.  The \`/index.md\` you're reading now is the homepage variant of the same negotiation.`,
    '',
  ].join('\n');

  return new Response(body, {
    headers: {
      'Content-Type': 'text/markdown; charset=utf-8',
      'Cache-Control': 'public, max-age=300',
    },
  });
};
