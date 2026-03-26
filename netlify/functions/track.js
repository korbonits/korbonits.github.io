import { createClient } from '@supabase/supabase-js';

export const handler = async (event) => {
  if (event.httpMethod !== 'POST') {
    return { statusCode: 405, body: 'Method Not Allowed' };
  }

  const secret = event.headers['x-analytics-secret'];
  if (!secret || secret !== process.env.ANALYTICS_SECRET) {
    return { statusCode: 401, body: 'Unauthorized' };
  }

  let body;
  try {
    body = JSON.parse(event.body ?? '{}');
  } catch {
    return { statusCode: 400, body: 'Bad Request: invalid JSON' };
  }

  const { path, referrer, user_agent, timestamp } = body;
  if (!path || typeof path !== 'string') {
    return { statusCode: 400, body: 'Bad Request: missing path' };
  }

  const supabaseUrl = process.env.SUPABASE_URL;
  const supabaseKey = process.env.SUPABASE_ANON_KEY;
  if (!supabaseUrl || !supabaseKey) {
    console.error('Supabase env vars not configured');
    return { statusCode: 500, body: 'Internal Server Error' };
  }

  const supabase = createClient(supabaseUrl, supabaseKey);
  const country = event.headers['x-nf-country'] ?? null;

  const { error } = await supabase.from('page_views').insert({
    path,
    referrer: referrer || null,
    user_agent: user_agent || null,
    country,
    created_at: timestamp || new Date().toISOString(),
  });

  if (error) {
    console.error('Supabase insert error:', error);
    return { statusCode: 500, body: 'Internal Server Error' };
  }

  return {
    statusCode: 200,
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ ok: true }),
  };
};
