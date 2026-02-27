import importlib
import os
import textwrap
from pathlib import Path

import httpx


def _write_config(path: Path, text: str) -> None:
    path.write_text(textwrap.dedent(text).lstrip(), encoding="utf-8")


def _reload_all() -> None:
    # Ensure new config is read
    for m in list(__import__("sys").modules):
        if m.startswith("litellm_ext"):
            __import__("sys").modules.pop(m, None)

    import litellm_ext.core.config  # noqa: F401
    import litellm_ext.policy  # noqa: F401
    import litellm_ext.extensions.httpx.count_tokens_stub  # noqa: F401

    importlib.reload(litellm_ext.core.config)
    importlib.reload(litellm_ext.policy)
    importlib.reload(litellm_ext.extensions.httpx.count_tokens_stub)


def _make_count_tokens_req(model: str) -> httpx.Request:
    body = {
        "model": model,
        "messages": [{"role": "user", "content": "hello world"}],
    }
    import json

    b = json.dumps(body).encode("utf-8")
    return httpx.Request(
        "POST",
        "https://example.com/v1/messages/count_tokens",
        headers={"content-type": "application/json"},
        content=b,
    )


def test_autocompact_tuning_scales_stubbed_tokens(tmp_path, monkeypatch):
    cfg = tmp_path / "cfg.yaml"
    monkeypatch.setenv("LITELLM_EXT_CONFIG_PATH", str(cfg))

    _write_config(
        cfg,
        """
        version: 1
        extensions:
          count_tokens_stub:
            enabled: true
            host_substr: ""
            target_suffixes:
              - /v1/messages/count_tokens

        policy:
            autocompact_tuning:
              enabled: true
              allow_underreport: true
              default_multiplier: 1.0
              multipliers:
                deepseek-chat: 2.0
        """,
    )

    _reload_all()

    import litellm_ext.extensions.httpx.count_tokens_stub as ct

    req = _make_count_tokens_req("deepseek-chat")
    resp = ct._count_tokens_mutator(req)
    assert resp is not None
    obj = resp.json()
    import json
    from litellm_ext.policy import estimate_input_tokens_heuristic

    base = estimate_input_tokens_heuristic(json.loads(req.content.decode('utf-8')))
    assert obj["input_tokens"] == int(base * 2)


def test_autocompact_tuning_disabled_reports_unscaled_tokens(tmp_path, monkeypatch):
    cfg = tmp_path / "cfg.yaml"
    monkeypatch.setenv("LITELLM_EXT_CONFIG_PATH", str(cfg))

    _write_config(
        cfg,
        """
        version: 1
        extensions:
          count_tokens_stub:
            enabled: true
            host_substr: ""
            target_suffixes:
              - /v1/messages/count_tokens

        policy:
            autocompact_tuning:
              enabled: false
              allow_underreport: true
              default_multiplier: 1.0
              multipliers:
                deepseek-chat: 999.0
        """,
    )

    _reload_all()

    import litellm_ext.extensions.httpx.count_tokens_stub as ct

    req = _make_count_tokens_req("deepseek-chat")
    resp = ct._count_tokens_mutator(req)
    assert resp is not None
    obj = resp.json()
    import json
    from litellm_ext.policy import estimate_input_tokens_heuristic
    base = estimate_input_tokens_heuristic(json.loads(req.content.decode('utf-8')))
    assert obj["input_tokens"] == base
