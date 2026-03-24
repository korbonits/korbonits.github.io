import { defineCollection, z } from 'astro:content';
import { glob } from 'astro/loaders';

const blog = defineCollection({
  loader: glob({ pattern: '**/*.md', base: './src/content/blog' }),
  schema: z.object({
    title: z.string(),
    date: z.coerce.date(),
    draft: z.boolean().optional(),
    description: z.string().optional(),
    tags: z.array(z.string()).optional(),
  }),
});

const now = defineCollection({
  loader: glob({ pattern: 'now.md', base: './src/content' }),
  schema: z.object({
    location: z.string().optional(),
    working_on: z.string().optional(),
    thinking_about: z.string().optional(),
  }),
});

export const collections = { blog, now };
