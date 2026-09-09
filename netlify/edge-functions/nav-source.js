// Stamps the navigation's `Sec-Fetch-Site` onto a short-lived cookie so the
// client-side tracker in BaseHead.astro can report it with the page view.
//
// Why this can't be done in the tracker itself: the tracker reaches the server
// through `fetch('/api/track')`, and that request's own Sec-Fetch-Site is
// always `same-origin`.  The header is only meaningful on the document request,
// which is what an edge function sees.
//
// What it buys: `document.referrer` being empty says only that no referrer
// reached the page.  Sec-Fetch-Site says why.
//   none        -> typed, bookmarked, or otherwise opened with no context
//   cross-site  -> something on another site linked here and stripped the
//                  referrer (apps, referrer-policy, most newsletter clients)
//   same-site   -> a subdomain
//   same-origin -> in-site navigation, including ClientRouter fetches
//
// The cookie is read and immediately cleared by the tracker, is not HttpOnly
// (the client has to read it), carries no identifier, and expires in a minute.

export default async (request, context) => {
  const response = await context.next();

  // HTML documents only.  Assets, .md variants and function responses skip.
  const contentType = response.headers.get('content-type') ?? '';
  if (!contentType.includes('text/html')) return response;

  const site = request.headers.get('sec-fetch-site') ?? 'unknown';
  if (!/^(none|cross-site|same-site|same-origin|unknown)$/.test(site)) return response;

  // `country` has been null on every row since March: x-nf-country doesn't
  // reach serverless functions the way context.geo reaches edge ones. Carry it
  // on the same cookie rather than adding a second header.
  const cc = context.geo?.country?.code ?? '';
  const country = /^[A-Z]{2}$/.test(cc) ? cc : '';

  response.headers.append(
    'Set-Cookie',
    `_nav=${site}|${country}; Path=/; Max-Age=60; SameSite=Lax; Secure`,
  );
  return response;
};
