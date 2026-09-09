import { createClient } from '@supabase/supabase-js';

// What this endpoint can and cannot defend against.
//
// PUBLIC_ANALYTICS_SECRET is embedded in every page, so it is not a secret and
// the header check below is not authentication — it only filters requests that
// never read the page. What follows raises the cost of casual forgery and stops
// a forger from choosing values the server can determine itself:
//
//   - Origin must be this site. A curl or script POSTing from anywhere else is
//     rejected; browsers always send it on a cross-origin JSON POST.
//   - created_at is server time. The client no longer picks when it happened,
//     so rows can't be backdated into a window you've already looked at.
//   - user_agent comes from the request header, not the body.
//   - Every string is length-capped and path must look like a path.
//
// It does NOT stop an automated browser. Anything running a real engine gets
// the secret, the Origin and the cookies for free and is indistinguishable
// here. The fix for that is logging in the edge function instead, which sees
// every request including the ones that never execute JavaScript.

const ALLOWED_ORIGINS = new Set([
  'https://korbonits.com',
  'https://www.korbonits.com',
]);

const VALID_NAV = ['none', 'cross-site', 'same-site', 'same-origin', 'unknown'];

const LIMITS = { path: 512, referrer: 1024, user_agent: 512 };

const clamp = (v, max) =>
  typeof v === 'string' && v.length ? v.slice(0, max) : null;

export const handler = async (event) => {
  if (event.httpMethod !== 'POST') {
    return { statusCode: 405, body: 'Method Not Allowed' };
  }

  const h = event.headers ?? {};

  // Same-origin only. Fall back to Referer when Origin is absent.
  const origin =
    h.origin ?? (h.referer ? new URL(h.referer).origin : null);
  if (!origin || !ALLOWED_ORIGINS.has(origin)) {
    return { statusCode: 403, body: 'Forbidden' };
  }

  const secret = h['x-analytics-secret'];
  if (!secret || secret !== process.env.ANALYTICS_SECRET) {
    return { statusCode: 401, body: 'Unauthorized' };
  }

  if (typeof event.body === 'string' && event.body.length > 4096) {
    return { statusCode: 413, body: 'Payload Too Large' };
  }

  let body;
  try {
    body = JSON.parse(event.body ?? '{}');
  } catch {
    return { statusCode: 400, body: 'Bad Request: invalid JSON' };
  }

  const path = clamp(body.path, LIMITS.path);
  if (!path || !path.startsWith('/') || /[\s\r\n]/.test(path)) {
    return { statusCode: 400, body: 'Bad Request: invalid path' };
  }

  const supabaseUrl = process.env.SUPABASE_URL;
  const supabaseKey = process.env.SUPABASE_ANON_KEY;
  if (!supabaseUrl || !supabaseKey) {
    console.error('Supabase env vars not configured');
    return { statusCode: 500, body: 'Internal Server Error' };
  }

  const supabase = createClient(supabaseUrl, supabaseKey);

  // Prefer the edge function's context.geo, relayed on the _nav cookie. The
  // x-nf-country header is not populated for serverless functions, which is why
  // this column was null on every row from March to September 2026.
  const claimed =
    typeof body.country === 'string' ? body.country.toUpperCase() : '';
  const country =
    (/^[A-Z]{2}$/.test(claimed) ? claimed : null) ??
    (h['x-nf-country'] || null);

  const referrer = clamp(body.referrer, LIMITS.referrer);

  const row = {
    path,
    referrer,
    // Header, not body: the body value is entirely caller-controlled.
    user_agent: clamp(h['user-agent'] ?? body.user_agent, LIMITS.user_agent),
    country,
    // Server time. Deliberately ignores any client-supplied timestamp.
    created_at: new Date().toISOString(),
  };

  // `nav_source` needs the migration in sql/2026-09-08-nav-source.sql. Until it
  // runs, PostgREST rejects the whole insert with PGRST204; drop the column and
  // retry rather than lose the page view.
  let { error } = await supabase
    .from('page_views')
    .insert({
      ...row,
      nav_source: VALID_NAV.includes(body.nav_source) ? body.nav_source : null,
    });

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
