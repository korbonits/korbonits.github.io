import { getCollection } from 'astro:content';
import { SITE_DESCRIPTION, SITE_TITLE } from '../consts';

export async function GET(context) {
  const posts = (await getCollection('blog', ({ data }) => !data.draft))
    .sort((a, b) => b.data.date.valueOf() - a.data.date.valueOf());

  const lines = [
    `# ${SITE_TITLE}`,
    ``,
    `> ${SITE_DESCRIPTION}`,
    ``,
    `## Blog`,
    ``,
    ...posts.map(post => {
      const url = new URL(`/blog/${post.id}/`, context.site);
      const tags = post.data.tags?.length ? ` [${post.data.tags.join(', ')}]` : '';
      const desc = post.data.description ? ` — ${post.data.description}` : '';
      return `- [${post.data.title}](${url})${tags}${desc}`;
    }),
    ``,
    `## Pages`,
    ``,
    `- [About](${new URL('/about/', context.site)})`,
  ];

  return new Response(lines.join('\n'));
}
