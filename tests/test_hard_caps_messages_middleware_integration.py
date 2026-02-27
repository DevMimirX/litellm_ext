import types

from fastapi import FastAPI, Request
from starlette.responses import JSONResponse
from starlette.testclient import TestClient

import litellm_ext.extensions.asgi.hard_caps_messages as hcm
from litellm_ext.core.model_alias import reset_model_alias_cache


def _fake_proxy_app():
    app = FastAPI()
    seen = {}

    @app.post("/v1/messages")
    async def _messages(request: Request):
        payload = await request.json()
        seen["payload"] = payload
        seen["post_body_msg_1"] = await request._receive()
        return JSONResponse({"ok": True})

    return app, seen


def _install_middleware(app):
    dummy_mod = types.SimpleNamespace(app=app)
    hcm._patch_proxy_app(dummy_mod)


def test_hard_caps_messages_replay_never_synthesizes_disconnect():
    app, seen = _fake_proxy_app()
    _install_middleware(app)
    client = TestClient(app)

    req = {
        "model": "doubao-seed-2.0-code",
        "max_tokens": 20000,
        "messages": [{"role": "user", "content": "ping"}],
    }
    res = client.post("/v1/messages", json=req)

    assert res.status_code == 200
    assert seen["payload"]["model"] == req["model"]
    assert seen["post_body_msg_1"]["type"] == "http.request"
    assert seen["post_body_msg_1"].get("more_body") is False


def test_hard_caps_messages_log_fields_prefer_alias(monkeypatch, tmp_path):
    cfg = tmp_path / "litellm.yaml"
    cfg.write_text(
        """
model_list:
  - model_name: glm-5-ali
    litellm_params:
      model: anthropic/glm-5
      api_base: "https://coding.dashscope.aliyuncs.com/apps/anthropic"
""".strip()
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("LITELLM_CONFIG", str(cfg))
    reset_model_alias_cache()

    got = hcm._model_log_fields("glm-5", "coding.dashscope.aliyuncs.com")
    assert got == "model='glm-5-ali' resolved_model='glm-5'"


def test_hard_caps_messages_logs_compact_route_when_detected(monkeypatch):
    logs = []

    class _FakeLogger:
        def debug(self, msg):
            logs.append(msg)

    monkeypatch.setattr(hcm, "_LOG", _FakeLogger())
    monkeypatch.setattr(hcm, "looks_like_compact", lambda payload: True)

    def _fake_enforce(payload, *, allow_raise=False):
        payload["model"] = "doubao-seed-2.0-code"
        return True

    monkeypatch.setattr(hcm, "enforce", _fake_enforce)

    app, _seen = _fake_proxy_app()
    _install_middleware(app)
    client = TestClient(app)

    req = {
        "model": "glm-5",
        "max_tokens": 20000,
        "messages": [{"role": "user", "content": "whatever"}],
    }
    res = client.post("/v1/messages", json=req)

    assert res.status_code == 200
    assert any("route /compact 'glm-5' -> 'doubao-seed-2.0-code'" in m for m in logs)
