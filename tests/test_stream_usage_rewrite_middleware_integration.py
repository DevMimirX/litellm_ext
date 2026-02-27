import json
import types

import pytest
from fastapi import FastAPI, Request
from starlette.responses import JSONResponse, StreamingResponse
from starlette.testclient import TestClient

import litellm_ext.extensions.asgi.stream_usage_rewrite as sur


def _fake_proxy_app():
    app = FastAPI()
    seen = {}

    @app.post("/v1/messages")
    async def _messages(request: Request):
        payload = await request.json()
        seen["payload"] = payload
        seen["post_body_msg_1"] = await request._receive()

        if payload.get("stream"):
            async def gen():
                # message_start with zeroed usage (the bug we're fixing)
                obj = {
                    "type": "message_start",
                    "message": {
                        "id": "msg_test",
                        "type": "message",
                        "role": "assistant",
                        "model": payload.get("model"),
                        "content": [],
                        "stop_reason": None,
                        "stop_sequence": None,
                        "usage": {"input_tokens": 0, "output_tokens": 0},
                    },
                }
                yield "event: message_start\n"
                yield "data: " + json.dumps(obj, separators=(",", ":"), ensure_ascii=False) + "\n\n"
                yield "event: message_stop\n"
                yield 'data: {"type":"message_stop"}\n\n'

            # Deliberately set via media_type (some SSE libs don't set headers early)
            return StreamingResponse(gen(), media_type="text/event-stream")

        # Non-stream Anthropic-like response body
        obj = {
            "id": "msg_test",
            "type": "message",
            "role": "assistant",
            "model": payload.get("model"),
            "content": [{"type": "text", "text": "hi"}],
            "stop_reason": "end_turn",
            "stop_sequence": None,
            "usage": {"input_tokens": 0, "output_tokens": 0},
        }
        return JSONResponse(obj)

    return app, seen


def _install_middleware(app):
    dummy_mod = types.SimpleNamespace(app=app)
    sur._patch_proxy_app(dummy_mod)


def test_middleware_patches_stream_usage_rewrite(monkeypatch):
    app, seen = _fake_proxy_app()

    monkeypatch.setattr(sur, "estimate_input_tokens_best_effort", lambda model, payload: 2)
    monkeypatch.setattr(sur, "autocompact_multiplier_for_model", lambda model: 1.5)

    _install_middleware(app)
    client = TestClient(app)

    req = {
        "model": "glm-4.7",
        "stream": True,
        "max_tokens": 10,
        "messages": [{"role": "user", "content": "hi"}],
    }

    with client.stream("POST", "/v1/messages", json=req) as r:
        text = "".join(list(r.iter_text()))

    assert text.count("event: message_start") == 1
    assert seen.get("payload") == req, "middleware must not consume request body"
    assert seen["post_body_msg_1"]["type"] == "http.request"

    lines = [ln for ln in text.splitlines() if ln.startswith("data: ")]
    assert lines, "expected data line in SSE"
    data = json.loads(lines[0].split("data: ", 1)[1])

    assert data["type"] == "message_start"
    assert data["message"]["usage"]["input_tokens"] == 3


def test_middleware_patches_nonstream_json_usage(monkeypatch):
    app, seen = _fake_proxy_app()

    monkeypatch.setattr(sur, "estimate_input_tokens_best_effort", lambda model, payload: 2)
    monkeypatch.setattr(sur, "autocompact_multiplier_for_model", lambda model: 1.5)

    _install_middleware(app)
    client = TestClient(app)

    req = {"model": "glm-4.7", "messages": [{"role": "user", "content": "hi"}]}
    res = client.post("/v1/messages", json=req)

    assert res.status_code == 200
    assert seen.get("payload") == req
    assert seen["post_body_msg_1"]["type"] == "http.request"
    obj = res.json()
    assert obj["usage"]["input_tokens"] == 3
