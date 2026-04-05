// netlify/functions/embed-query.js
// Proxies a search query to the Modal embedding endpoint.
// Keeps MODAL_EMBED_URL server-side.

export default async function handler(req) {
  if (req.method !== "POST") {
    return new Response(JSON.stringify({ error: "Method not allowed" }), {
      status: 405,
      headers: { "Content-Type": "application/json" },
    });
  }

  let body;
  try {
    body = await req.json();
  } catch {
    return new Response(JSON.stringify({ error: "Invalid JSON" }), {
      status: 400,
      headers: { "Content-Type": "application/json" },
    });
  }

  const { text } = body;
  if (!text?.trim()) {
    return new Response(JSON.stringify({ error: "text is required" }), {
      status: 400,
      headers: { "Content-Type": "application/json" },
    });
  }

  const modalBase = process.env.MODAL_EMBED_URL?.replace(/\/embed$/, "");
  const modalUrl = modalBase ? `${modalBase}/embed` : null;
  if (!modalUrl) {
    return new Response(JSON.stringify({ error: "Embedding service not configured" }), {
      status: 500,
      headers: { "Content-Type": "application/json" },
    });
  }

  try {
    const upstream = await fetch(modalUrl, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ texts: [text], query: true }),
    });

    if (!upstream.ok) {
      const err = await upstream.text();
      console.error("[embed-query] Modal error:", err);
      return new Response(JSON.stringify({ error: "Embedding service error" }), {
        status: 502,
        headers: { "Content-Type": "application/json" },
      });
    }

    const { embeddings } = await upstream.json();
    return new Response(JSON.stringify({ embedding: embeddings[0] }), {
      status: 200,
      headers: { "Content-Type": "application/json" },
    });
  } catch (e) {
    console.error("[embed-query] error:", e);
    return new Response(JSON.stringify({ error: "Internal server error" }), {
      status: 500,
      headers: { "Content-Type": "application/json" },
    });
  }
}

export const config = { path: "/api/embed-query" };
