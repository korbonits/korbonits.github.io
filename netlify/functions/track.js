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

  const { path, referrer, user_agent, nav_source, timestamp } = body;
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

  const VALID_NAV = ['none', 'cross-site', 'same-site', 'same-origin', 'unknown'];

  const row = {
    path,
    referrer: referrer || null,
    user_agent: user_agent || null,
    country,
    created_at: timestamp || new Date().toISOString(),
  };

  // `nav_source` needs the migration in sql/2026-09-08-nav-source.sql. Until it
  // runs, PostgREST rejects the whole insert with PGRST204; drop the column and
  // retry rather than lose the page view.
  let { error } = await supabase
    .from('page_views')
    .insert({ ...row, nav_source: VALID_NAV.includes(nav_source) ? nav_source : null });

  if (error && (error.code === 'PGRST204' || /nav_source/.test(error.message ?? ''))) {
    ({ error } = await supabase.from('page_views').insert(row));
  }

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
