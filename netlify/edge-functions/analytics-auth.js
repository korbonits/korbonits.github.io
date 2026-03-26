export default async (request, context) => {
  const expectedUser = Deno.env.get('ANALYTICS_USER') ?? '';
  const expectedPass = Deno.env.get('ANALYTICS_PASSWORD') ?? '';

  const authHeader = request.headers.get('authorization') ?? '';

  if (authHeader.startsWith('Basic ')) {
    try {
      const decoded = atob(authHeader.slice(6));
      const colon = decoded.indexOf(':');
      if (colon !== -1) {
        const user = decoded.slice(0, colon);
        const pass = decoded.slice(colon + 1);
        if (user === expectedUser && pass === expectedPass) {
          return context.next();
        }
      }
    } catch (_) {}
  }

  return new Response('Unauthorized', {
    status: 401,
    headers: { 'WWW-Authenticate': 'Basic realm="Analytics"' },
  });
};
