import satori from 'satori';
import { Resvg } from '@resvg/resvg-js';
import { readFileSync } from 'fs';
import { getCollection } from 'astro:content';
import type { APIRoute, GetStaticPaths } from 'astro';

const fontRegular = readFileSync(new URL('../../../public/fonts/atkinson-regular.woff', import.meta.url));
const fontBold = readFileSync(new URL('../../../public/fonts/atkinson-bold.woff', import.meta.url));

export const getStaticPaths: GetStaticPaths = async () => {
  const posts = await getCollection('blog', p => !p.data.draft);
  const blogPaths = posts.map(p => ({ params: { slug: `blog/${p.id}` }, props: { title: p.data.title } }));
  const staticPaths = [
    { params: { slug: 'now' }, props: { title: 'Now' } },
    { params: { slug: 'reading' }, props: { title: 'Reading' } },
  ];
  return [...blogPaths, ...staticPaths];
};

async function renderOG(title: string): Promise<Buffer> {
  const svg = await satori(
    {
      type: 'div',
      props: {
        style: {
          width: '1200px', height: '630px',
          background: 'linear-gradient(135deg, #0f1219 0%, #1a1f2e 100%)',
          display: 'flex', flexDirection: 'column',
          justifyContent: 'center', padding: '80px',
          fontFamily: 'Atkinson',
        },
        children: [
          {
            type: 'div',
            props: {
              style: { display: 'flex', width: '60px', height: '6px', background: '#2337ff', borderRadius: '3px', marginBottom: '40px' },
            },
          },
          {
            type: 'div',
            props: {
              style: { display: 'flex', fontSize: title.length > 40 ? '56px' : '72px', fontWeight: 700, color: '#ffffff', lineHeight: 1.15, marginBottom: '40px' },
              children: title,
            },
          },
          {
            type: 'div',
            props: {
              style: { display: 'flex', fontSize: '32px', color: '#7b8fff' },
              children: 'korbonits.com',
            },
          },
        ],
      },
    },
    {
      width: 1200, height: 630,
      fonts: [
        { name: 'Atkinson', data: fontRegular, weight: 400 },
        { name: 'Atkinson', data: fontBold, weight: 700 },
      ],
    }
  );

  return new Resvg(svg).render().asPng();
}

export const GET: APIRoute = async ({ props }) => {
  const png = await renderOG(props.title);
  return new Response(png, { headers: { 'Content-Type': 'image/png' } });
};
