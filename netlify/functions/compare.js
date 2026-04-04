// netlify/functions/compare.js
// Proxies requests to the Anthropic API, keeping the key server-side.

export default async function handler(req, context) {
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
    return new Response(JSON.stringify({ error: "Invalid JSON body" }), {
      status: 400,
      headers: { "Content-Type": "application/json" },
    });
  }

  const { textA, textB } = body;

  if (!textA?.trim() || !textB?.trim()) {
    return new Response(JSON.stringify({ error: "Both textA and textB are required" }), {
      status: 400,
      headers: { "Content-Type": "application/json" },
    });
  }

  const apiKey = process.env.ANTHROPIC_API_KEY;
  if (!apiKey) {
    return new Response(JSON.stringify({ error: "API key not configured" }), {
      status: 500,
      headers: { "Content-Type": "application/json" },
    });
  }

  const prompt = `You are an expert in natural language processing and text embeddings.

A user has provided two text snippets. Your job:
1. Estimate a cosine similarity score between 0 and 100 (integer), where 0 = completely unrelated, 50 = topically related but semantically distinct, 100 = essentially identical in meaning.
2. Write a 2–3 sentence plain-English explanation of WHY they score that way. Explain what they share and what differs. Do not use jargon — imagine explaining this to a business executive who has never heard of embeddings.

Text A: """${textA}"""
Text B: """${textB}"""

Respond in this exact JSON format, no other text:
{"score": <integer 0-100>, "explanation": "<2-3 sentence explanation>"}`;

  try {
    const upstream = await fetch("https://api.anthropic.com/v1/messages", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "x-api-key": apiKey,
        "anthropic-version": "2023-06-01",
      },
      body: JSON.stringify({
        model: "claude-sonnet-4-20250514",
        max_tokens: 1000,
        messages: [{ role: "user", content: prompt }],
      }),
    });

    if (!upstream.ok) {
      const err = await upstream.text();
      console.error("Anthropic API error:", err);
      return new Response(JSON.stringify({ error: "Upstream API error" }), {
        status: 502,
        headers: { "Content-Type": "application/json" },
      });
    }

    const data = await upstream.json();
    const text = data.content?.find((b) => b.type === "text")?.text || "";
    const clean = text.replace(/```json|```/g, "").trim();
    const parsed = JSON.parse(clean);

    return new Response(JSON.stringify(parsed), {
      status: 200,
      headers: { "Content-Type": "application/json" },
    });
  } catch (e) {
    console.error("Function error:", e);
    return new Response(JSON.stringify({ error: "Internal server error" }), {
      status: 500,
      headers: { "Content-Type": "application/json" },
    });
  }
}

export const config = { path: "/api/compare" };
