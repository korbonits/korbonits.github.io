// Build-time markdown endpoint — emits a `.md` variant of every blog post
// at /blog/<slug>.md so that AI agents requesting `Accept: text/markdown`
// can be served the raw source via the markdown-negotiation edge function.
//
// The output is the post's frontmatter + body, lightly normalized: a single
// `# Title` line is prepended (so the markdown renders correctly when read
// in isolation) and the original frontmatter is preserved as a leading
// YAML block so date / tags / description survive.

import type { APIRoute } from 'astro';
import { getCollection } from 'astro:content';

export async function getStaticPaths() {
  // Match the published-post filter used by [slug].astro so we never
  // expose the markdown for drafts.
  const posts = await getCollection('blog', ({ data }) => !data.draft);
  return posts.map(post => ({ params: { slug: post.id }, props: { post } }));
}

export const GET: APIRoute = async ({ props }) => {
  const post = (props as any).post;
  const fm = post.data;

  // Re-emit a YAML frontmatter block from the structured data so consumers
  // that parse the markdown have title/date/description/tags available
  // without re-fetching the HTML page.
  const yaml = [
    '---',
    `title: ${JSON.stringify(fm.title)}`,
    `date: ${fm.date instanceof Date ? fm.date.toISOString().slice(0, 10) : fm.date}`,
    fm.description ? `description: ${JSON.stringify(fm.description)}` : null,
    Array.isArray(fm.tags) && fm.tags.length
      ? `tags: [${fm.tags.map((t: string) => JSON.stringify(t)).join(', ')}]`
      : null,
    `canonical_url: https://korbonits.com/blog/${post.id}/`,
    '---',
    '',
  ]
    .filter(Boolean)
    .join('\n');

  const heading = `\n# ${fm.title}\n\n`;

  return new Response(yaml + heading + (post.body ?? ''), {
    headers: {
      'Content-Type': 'text/markdown; charset=utf-8',
      'Cache-Control': 'public, max-age=300',
    },
  });
};
