import { defineConfig } from 'astro/config';
import sitemap from '@astrojs/sitemap';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';

export default defineConfig({
  site: 'https://korbonits.com',
  integrations: [sitemap()],
  markdown: {
    shikiConfig: { theme: 'github-dark' },
    remarkPlugins: [remarkMath],
    rehypePlugins: [rehypeKatex],
  },
});
