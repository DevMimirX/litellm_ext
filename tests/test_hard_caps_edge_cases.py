from __future__ import annotations

import json
import types
import sys
import httpx


def test_hard_caps_httpx_mutator_clamps_and_mutates_request_body_and_length():
    import litellm_ext.core.registry as reg
    import litellm_ext.extensions.httpx.hard_caps as hc

    reg.install_httpx_patch()
    hc.install()

    original = {"model": "deepseek/deepseek-chat", "max_tokens": 9000, "messages": [{"role": "user", "content": "hi"}]}
    req = httpx.Request(
        "POST",
        "https://example.com/v1/messages",
        headers={"content-type": "application/json"},
        content=b"  " + json.dumps(original).encode("utf-8"),  # leading whitespace
    )

    transport = httpx.MockTransport(lambda r: httpx.Response(200, json={"ok": True}, request=r))
    with httpx.Client(transport=transport) as c:
        _ = c.send(req)

    mutated = json.loads(req.content.decode("utf-8"))
    assert mutated["max_tokens"] == 8192, "deepseek-chat must clamp outputs to 8192"
    assert req.headers.get("content-length") == str(len(req.content))


def test_hard_caps_skips_count_tokens_paths_entirely():
    import litellm_ext.core.registry as reg
    import litellm_ext.extensions.httpx.hard_caps as hc

    reg.install_httpx_patch()
    hc.install()

    original = {"model": "deepseek/deepseek-chat", "max_tokens": 9000}
    req = httpx.Request(
        "POST",
        "https://example.com/v1/messages/count_tokens",
        headers={"content-type": "application/json"},
        content=json.dumps(original).encode("utf-8"),
    )

    transport = httpx.MockTransport(lambda r: httpx.Response(200, json={"ok": True}, request=r))
    with httpx.Client(transport=transport) as c:
        _ = c.send(req)

    # Must remain unchanged because skip-path prevents parsing/enforcing
    still = json.loads(req.content.decode("utf-8"))
    assert still["max_tokens"] == 9000


def test_hard_caps_sets_default_max_tokens_when_missing():
    import litellm_ext.core.registry as reg
    import litellm_ext.extensions.httpx.hard_caps as hc

    reg.install_httpx_patch()
    hc.install()

    original = {"model": "deepseek/deepseek-chat", "messages": [{"role": "user", "content": "hi"}]}
    req = httpx.Request(
        "POST",
        "https://example.com/v1/messages",
        headers={"content-type": "application/json"},
        content=json.dumps(original).encode("utf-8"),
    )

    transport = httpx.MockTransport(lambda r: httpx.Response(200, json={"ok": True}, request=r))
    with httpx.Client(transport=transport) as c:
        _ = c.send(req)

    mutated = json.loads(req.content.decode("utf-8"))
    assert mutated["max_tokens"] == 8192, "missing max_tokens should be set to the model cap"


def test_litellm_wrapper_positional_model_clamps_before_calling_orig():
    """
    Edge case: model passed positionally (args[0]) must be enforced without duplicating model.
    """
    # Install a fake litellm before importing hard caps patch
    captured = {}

    fake_litellm = types.ModuleType("litellm")

    def completion(model, *args, **kwargs):
        captured["model"] = model
        captured["kwargs"] = dict(kwargs)
        return {"ok": True}

    async def acompletion(model, *args, **kwargs):
        captured["amodel"] = model
        captured["akwargs"] = dict(kwargs)
        return {"ok": True}

    fake_litellm.completion = completion
    fake_litellm.acompletion = acompletion
    # token_counter optional; not needed for this test

    sys.modules["litellm"] = fake_litellm

    import litellm_ext.core.registry as reg
    import litellm_ext.extensions.httpx.hard_caps as hc

    reg.install_httpx_patch()
    hc.install()

    import litellm  # our fake
    litellm.completion("deepseek/deepseek-chat", messages=[{"role": "user", "content": "hi"}], max_tokens=9000)

    assert captured["model"] == "deepseek/deepseek-chat"
    assert captured["kwargs"]["max_tokens"] == 8192


def test_router_wrapper_is_patched_and_clamps():
    captured = {}

    fake_litellm = types.ModuleType("litellm")

    class Router:
        def completion(self, model, *args, **kwargs):
            captured["model"] = model
            captured["kwargs"] = dict(kwargs)
            return {"ok": True}

    fake_litellm.Router = Router

    # Provide litellm.router.Router import path too
    fake_router_mod = types.ModuleType("litellm.router")
    fake_router_mod.Router = Router

    sys.modules["litellm"] = fake_litellm
    sys.modules["litellm.router"] = fake_router_mod

    import litellm_ext.core.registry as reg
    import litellm_ext.extensions.httpx.hard_caps as hc

    reg.install_httpx_patch()
    hc.install()

    r = Router()
    r.completion("deepseek/deepseek-chat", messages=[{"role": "user", "content": "hi"}], max_tokens=999999)
    assert captured["kwargs"]["max_tokens"] == 8192


def test_hard_caps_logs_alias_for_provider_model_when_host_matches(monkeypatch, tmp_path):
    import litellm_ext.extensions.httpx.hard_caps as hc

    cfg = tmp_path / "litellm.yaml"
    cfg.write_text(
        """
model_list:
  - model_name: glm-5
    litellm_params:
      model: anthropic/glm-5
      api_base: "https://api.z.ai/api/anthropic"
  - model_name: glm-5-ali
    litellm_params:
      model: anthropic/glm-5
      api_base: "https://coding.dashscope.aliyuncs.com/apps/anthropic"
""".strip()
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("LITELLM_CONFIG", str(cfg))

    logs = []

    class _FakeLogger:
        def enabled(self):
            return True

        def debug(self, msg):
            logs.append(msg)

    monkeypatch.setattr(hc, "_LOG", _FakeLogger())

    payload = {
        "model": "glm-5",
        "max_tokens": 20000,
        "messages": [{"role": "user", "content": "hello"}],
    }
    req = httpx.Request(
        "POST",
        "https://coding.dashscope.aliyuncs.com/apps/anthropic/v1/messages",
        headers={"content-type": "application/json"},
        content=json.dumps(payload).encode("utf-8"),
    )

    _ = hc._hard_caps_mutator(req)

    assert any("caps unchanged model='glm-5-ali' resolved_model='glm-5'" in m for m in logs)


def test_hard_caps_enforces_using_host_alias_then_restores_provider_model(monkeypatch, tmp_path):
    import litellm_ext.extensions.httpx.hard_caps as hc

    cfg = tmp_path / "litellm.yaml"
    cfg.write_text(
        """
model_list:
  - model_name: glm-5
    litellm_params:
      model: anthropic/glm-5
      api_base: "https://api.z.ai/api/anthropic"
  - model_name: glm-5-ali
    litellm_params:
      model: anthropic/glm-5
      api_base: "https://coding.dashscope.aliyuncs.com/apps/anthropic"
""".strip()
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("LITELLM_CONFIG", str(cfg))

    seen = {}

    def _fake_enforce(payload, *, allow_raise=True):
        seen["model"] = payload.get("model")
        return False

    monkeypatch.setattr(hc, "enforce", _fake_enforce)

    payload = {
        "model": "glm-5",
        "max_tokens": 20000,
        "messages": [{"role": "user", "content": "hello"}],
    }
    req = httpx.Request(
        "POST",
        "https://coding.dashscope.aliyuncs.com/apps/anthropic/v1/messages",
        headers={"content-type": "application/json"},
        content=json.dumps(payload).encode("utf-8"),
    )

    _ = hc._hard_caps_mutator(req)

    assert seen["model"] == "glm-5-ali"
    final_payload = json.loads(req.content.decode("utf-8"))
    assert final_payload["model"] == "glm-5"
