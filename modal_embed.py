# modal_embed.py
# Deploy: modal deploy modal_embed.py
# Set MODAL_EMBED_URL in Netlify to the base URL (without trailing slash).
# The build script appends /embed and /project as needed.
#
# POST /embed   { "texts": ["..."], "query": false } → { "embeddings": [[...]] }
# POST /project { "embeddings": [[...]] }            → { "coords": [[x,y]], "mean": [...], "components": [[...],[...]] }

import modal
from fastapi import FastAPI

app = modal.App("korbonits-embed")

volume = modal.Volume.from_name("korbonits-embed-cache", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("fastembed>=0.4.0", "numpy", "scikit-learn", "fastapi[standard]")
)

web_app = FastAPI()


@web_app.post("/embed")
def embed(item: dict) -> dict:
    from fastembed import TextEmbedding

    model = TextEmbedding("nomic-ai/nomic-embed-text-v1.5")
    texts = item.get("texts", [])
    prefix = "search_query: " if item.get("query") else "search_document: "
    prefixed = [prefix + t for t in texts]
    embeddings = list(model.embed(prefixed))
    return {"embeddings": [e.tolist() for e in embeddings]}


@web_app.post("/project")
def project(item: dict) -> dict:
    import numpy as np
    from sklearn.decomposition import PCA

    embeddings = np.array(item["embeddings"], dtype=np.float32)
    pca = PCA(n_components=2)
    coords = pca.fit_transform(embeddings)
    return {
        "coords": coords.tolist(),
        "mean": pca.mean_.tolist(),
        "components": pca.components_.tolist(),
    }


@app.function(
    image=image,
    volumes={"/root/.cache/fastembed": volume},
    min_containers=1,
)
@modal.asgi_app()
def api():
    return web_app
