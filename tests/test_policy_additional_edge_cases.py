from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any, Dict

import pytest


def _write_default_cfg(tmp_path: Path) -> Path:
    # Copy the repository default config into a temp location so tests don't depend on CWD.
    repo_cfg = Path(__file__).resolve().parents[1] / "config" / "extensions.yaml"
    p = tmp_path / "extensions.yaml"
    p.write_text(repo_cfg.read_text())
    return p


def _reload_policy_modules() -> None:
    import litellm_ext.core.config as cfg  # type: ignore
    import litellm_ext.policy as policy  # type: ignore
    importlib.reload(cfg)
    importlib.reload(policy)


def test_enforce_clamps_and_syncs_alias_output_keys(tmp_path: Path, monkeypatch):
    cfg_path = _write_default_cfg(tmp_path)
    monkeypatch.setenv("LITELLM_EXT_CONFIG_PATH", str(cfg_path))
    _reload_policy_modules()

    import litellm_ext.policy as policy  # type: ignore

    payload: Dict[str, Any] = {
        "model": "deepseek/deepseek-chat",
        "max_output_tokens": 999999,
        "max_completion_tokens": 999999,
        "messages": [{"role": "user", "content": "hi"}],
    }
    changed = policy.enforce(payload, allow_raise=True)
    assert changed is True
    assert payload["max_tokens"] == 8192
    assert payload["max_output_tokens"] == 8192
    assert payload["max_completion_tokens"] == 8192


def test_enforce_sets_default_max_tokens_when_missing(tmp_path: Path, monkeypatch):
    cfg_path = _write_default_cfg(tmp_path)
    monkeypatch.setenv("LITELLM_EXT_CONFIG_PATH", str(cfg_path))
    _reload_policy_modules()

    import litellm_ext.policy as policy  # type: ignore

    payload = {"model": "deepseek-chat", "messages": [{"role": "user", "content": "hi"}]}
    changed = policy.enforce(payload, allow_raise=True)
    assert changed is True
    assert payload["max_tokens"] == 8192


def test_invalid_overflow_policy_falls_back_to_reduce_then_trim(tmp_path: Path, monkeypatch):
    cfg_path = tmp_path / "extensions.yaml"
    cfg_path.write_text(
        """extensions:
  httpx_registry: {enabled: true}
  count_tokens_stub: {enabled: true}
  hard_caps: {enabled: true}
  streaming_sse: {enabled: true}

policy:
  safety_buffer_tokens: 1024
  overflow_policy: not-a-real-policy
  min_tail_messages: 6
  max_trim_steps: 200
  model_limits:
    "*": {max_output: 16000, max_context: null}
    "deepseek-chat": {max_output: 8192, max_context: 131072}

compact:
  enabled: true
  target_model: glm-4.7
  patterns:
    - '<command-name>\\s*/compact\\s*</command-name>'
"""
    )
    monkeypatch.setenv("LITELLM_EXT_CONFIG_PATH", str(cfg_path))
    _reload_policy_modules()

    import litellm_ext.policy as policy  # type: ignore

    assert policy.OVERFLOW_POLICY == "reduce_then_trim"


def test_model_normalization_strips_provider_prefix(tmp_path: Path, monkeypatch):
    cfg_path = _write_default_cfg(tmp_path)
    monkeypatch.setenv("LITELLM_EXT_CONFIG_PATH", str(cfg_path))
    _reload_policy_modules()

    import litellm_ext.policy as policy  # type: ignore

    assert policy.normalize_model("deepseek/deepseek-chat") == "deepseek-chat"
    assert policy.normalize_model("zai/glm-4.7") == "glm-4.7"
    assert policy.normalize_model("glm-4.7") == "glm-4.7"
    assert policy.normalize_model(None) == ""
