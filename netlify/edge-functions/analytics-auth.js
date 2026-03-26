export default async (request, context) => {
  const user = Deno.env.get('ANALYTICS_USER');
  const password = Deno.env.get('ANALYTICS_PASSWORD');

  if (!user || !password) {
    return new Response('Analytics auth not configured', { status: 500 });
  }

  const auth = request.headers.get('Authorization') ?? '';
  const expected = 'Basic ' + btoa(`${user}:${password}`);

  if (auth !== expected) {
    return new Response('Unauthorized', {
      status: 401,
      headers: { 'WWW-Authenticate': 'Basic realm="Analytics"' },
    });
  }

  return context.next();
};
