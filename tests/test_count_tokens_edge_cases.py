
from __future__ import annotations

import importlib
import json
import math
import os
import sys
from pathlib import Path

import httpx
import pytest


def _write_config(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def _reset_patch_modules() -> None:
    # Ensure config changes take effect: patches read YAML at import-time.
    for name in list(sys.modules.keys()):
        if name.startswith("litellm_ext"):
            sys.modules.pop(name, None)


def _reload_patches() -> None:
    _reset_patch_modules()
    import litellm_ext.core.registry  # noqa: F401
    import litellm_ext.policy  # noqa: F401
    import litellm_ext.extensions.httpx.count_tokens_stub  # noqa: F401


def test_count_tokens_whitespace_json_and_missing_content_type_sniffing(tmp_path, monkeypatch):
    cfg = tmp_path / "cfg.yaml"
    monkeypatch.setenv("LITELLM_EXT_CONFIG_PATH", str(cfg))

    _write_config(
        cfg,
        """version: 1
extensions:
  httpx_registry:
    enabled: true
  policy:
    enabled: true
    autocompact_tuning:
      enabled: true
      allow_underreport: false
      default_multiplier: 1.0
      multipliers:
        deepseek-chat: 1.25
    model_limits:
      "*": {max_output: 20000, max_context: null}
  count_tokens_stub:
    enabled: true
    host_substr: ""   # no restriction
""",
    )
    _reload_patches()

    from litellm_ext.core.registry import install_httpx_patch
    from litellm_ext.extensions.httpx.count_tokens_stub import install
    from litellm_ext.policy import estimate_input_tokens_heuristic, autocompact_multiplier_for_model

    install_httpx_patch()
    install()

    req_body = {"model": "deepseek-chat", "messages": [{"role": "user", "content": "hello"}]}
    raw = b"   " + json.dumps(req_body).encode("utf-8")
    request = httpx.Request(
        "POST",
        "https://api.anthropic.com/v1/messages/count_tokens",
        headers={},  # missing content-type
        content=raw,
    )

    transport = httpx.MockTransport(lambda r: httpx.Response(500, json={"should_not": "hit"}, request=r))
    with httpx.Client(transport=transport) as client:
        resp = client.send(request)

    assert resp.status_code == 200
    base = estimate_input_tokens_heuristic(req_body)
    mult = autocompact_multiplier_for_model("deepseek-chat")
    expected = int(math.ceil(base * mult))
    assert resp.json()["input_tokens"] == expected


def test_count_tokens_host_restriction_blocks_non_anthropic(tmp_path, monkeypatch):
    cfg = tmp_path / "cfg.yaml"
    monkeypatch.setenv("LITELLM_EXT_CONFIG_PATH", str(cfg))

    _write_config(
        cfg,
        """version: 1
extensions:
  httpx_registry:
    enabled: true
  policy:
    enabled: true
    autocompact_tuning:
      enabled: true
      allow_underreport: true
      default_multiplier: 1.0
      multipliers: {}
    model_limits:
      "*": {max_output: 20000, max_context: null}
  count_tokens_stub:
    enabled: true
    host_substr: anthropic.com
""",
    )
    _reload_patches()

    from litellm_ext.core.registry import install_httpx_patch
    from litellm_ext.extensions.httpx.count_tokens_stub import install

    install_httpx_patch()
    install()

    # Same path suffix but non-matching host: should fall through to transport
    request = httpx.Request(
        "POST",
        "https://open.bigmodel.cn/api/anthropic/v1/messages/count_tokens",
        headers={"content-type": "application/json"},
        content=b'{"messages":[]}' ,
    )

    transport = httpx.MockTransport(lambda r: httpx.Response(200, json={"ok": True}, request=r))
    with httpx.Client(transport=transport) as client:
        resp = client.send(request)

    assert resp.status_code == 200
    assert resp.json() == {"ok": True}


def test_count_tokens_invalid_json_still_returns_stub_and_never_raises(tmp_path, monkeypatch):
    cfg = tmp_path / "cfg.yaml"
    monkeypatch.setenv("LITELLM_EXT_CONFIG_PATH", str(cfg))

    _write_config(
        cfg,
        """version: 1
extensions:
  httpx_registry:
    enabled: true
  policy:
    enabled: true
    autocompact_tuning:
      enabled: true
      allow_underreport: false
      default_multiplier: 1.10
      multipliers: {}
    model_limits:
      "*": {max_output: 20000, max_context: null}
  count_tokens_stub:
    enabled: true
    host_substr: ""   # no restriction
""",
    )
    _reload_patches()

    from litellm_ext.core.registry import install_httpx_patch
    from litellm_ext.extensions.httpx.count_tokens_stub import install
    from litellm_ext.policy import estimate_input_tokens_heuristic, autocompact_multiplier_for_model

    install_httpx_patch()
    install()

    request = httpx.Request(
        "POST",
        "https://api.anthropic.com/v1/messages/count_tokens",
        headers={"content-type": "application/json"},
        content=b"{this is not json",
    )
    transport = httpx.MockTransport(lambda r: httpx.Response(500, json={"should_not": "hit"}, request=r))
    with httpx.Client(transport=transport) as client:
        resp = client.send(request)

    assert resp.status_code == 200
    data = {}
    base = estimate_input_tokens_heuristic(data)
    mult = autocompact_multiplier_for_model(None)
    expected = int(math.ceil(base * mult))
    assert resp.json()["input_tokens"] == expected
