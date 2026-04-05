# modal_embed.py
# Deploy once: modal deploy modal_embed.py
# Then set MODAL_EMBED_URL in Netlify environment variables.
#
# Endpoint: POST /embed
#   body: { "texts": ["..."], "query": false }
#   response: { "embeddings": [[...], ...] }
#
# Pass query=true when embedding a search query (applies "search_query:" prefix).
# Omit or pass query=false for documents (applies "search_document:" prefix).
# nomic-embed-text-v1.5 is prefix-aware and benefits from this distinction.

import modal

app = modal.App("korbonits-embed")

# Cache model weights across cold starts — avoids re-downloading on every build.
volume = modal.Volume.from_name("korbonits-embed-cache", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("fastembed>=0.4.0", "numpy", "fastapi[standard]")
)


@app.function(
    image=image,
    volumes={"/root/.cache/fastembed": volume},
    min_containers=1,  # one warm container — eliminates cold start latency for debounced search
)
@modal.fastapi_endpoint(method="POST")
def embed(item: dict) -> dict:
    from fastembed import TextEmbedding

    model = TextEmbedding("nomic-ai/nomic-embed-text-v1.5")
    texts = item.get("texts", [])
    prefix = "search_query: " if item.get("query") else "search_document: "
    prefixed = [prefix + t for t in texts]
    embeddings = list(model.embed(prefixed))
    return {"embeddings": [e.tolist() for e in embeddings]}
