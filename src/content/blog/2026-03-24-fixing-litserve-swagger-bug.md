---
title: "My First Open Source Contribution to LitServe"
date: 2026-03-24
description: "How I tracked down a Swagger UI bug in LitServe, fixed it in three lines, and shipped a PR — with a little help from Kiro."
tags: ["open-source", "python", "fastapi", "litserve", "ai"]
draft: true
---

Last night I fixed a bug in [LitServe](https://github.com/Lightning-AI/LitServe), Lightning AI's inference server framework. Here's the story.

## The Bug

[Issue #667](https://github.com/Lightning-AI/LitServe/issues/667): if you start a LitServe server and open `/docs`, the Swagger UI shows no input form for the `/predict` endpoint. You click "Execute", it sends an empty body, and the server crashes with:

```
json.decoder.JSONDecodeError: Expecting value: line 1 column 1 (char 0)
```

Not a great first impression for anyone trying to explore the API interactively.

## Why It Happened

LitServe inspects the type annotation on your `decode_request` method to figure out what request type to use. If you don't annotate it (which is totally valid and common), it falls back to Starlette's raw `Request` type:

```python
request_type = decode_request_signature.parameters["request"].annotation
if request_type == decode_request_signature.empty:
    request_type = Request  # ← FastAPI can't generate a schema for this
```

FastAPI knows how to generate OpenAPI schemas for Pydantic models and primitive types, but `Request` is an opaque Starlette object — there's nothing to introspect. So Swagger renders nothing, and sends an empty body.

## The Fix

Three small changes to `server.py`:

**1.** Add `Any, Dict` to the `typing` import.

**2.** When `request_type is Request`, use `Dict[str, Any]` as the endpoint's parameter type instead. FastAPI sees this, generates a proper JSON schema, and Swagger renders the input form. FastAPI also parses the body into a dict automatically — so no manual `request.json()` call needed.

```python
swagger_request_type = Dict[str, Any] if request_type is Request else request_type

async def endpoint_handler(request: swagger_request_type) -> response_type:
    return await handler.handle_request(request, request_type)
```

**3.** In `_prepare_request`, add an early return for when the request is already a dict (which it will be when FastAPI parses the `Dict[str, Any]` body):

```python
async def _prepare_request(self, request, request_type) -> dict:
    if isinstance(request, dict):
        return request
    # ... rest of existing logic
```

The fix is fully backward compatible — users who already annotate `decode_request` with a Pydantic model are completely unaffected.

## The CI/CD

One thing that impressed me: LitServe's CI is thorough. The moment I opened [PR #670](https://github.com/Lightning-AI/LitServe/pull/670), pre-commit.ci automatically pushed a formatting fix commit, and the full test suite kicked off across multiple Python versions and platforms. Ten code owners were auto-requested for review. All of this before I had to do anything.

It's a good reminder that a well-maintained open source project has a lot of invisible infrastructure keeping it healthy.

## The Test

A regression test to make sure this never breaks again:

```python
def test_swagger_request_body_without_decode_request_annotation():
    server = LitServer(NoAnnotationLitAPI(), accelerator="cpu", devices=1, timeout=5)
    schema = server.app.openapi()
    predict_post = schema["paths"]["/predict"]["post"]
    assert "requestBody" in predict_post
    assert "application/json" in predict_post["requestBody"]["content"]
```

The test verifies the OpenAPI schema directly — no need to spin up a real server or make HTTP requests.

## Takeaway

The bug was small, the fix was small, and the whole thing — from reading the issue to pushing the PR — took about an hour. If you've been meaning to make your first open source contribution, bugs labeled [`help wanted`](https://github.com/Lightning-AI/LitServe/issues?q=is%3Aopen+is%3Aissue+label%3A%22help+wanted%22) are a great place to start.
