from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any, Dict


def _write_cfg(tmp_path: Path, text: str) -> Path:
    cfg_path = tmp_path / "extensions.yaml"
    cfg_path.write_text(text, encoding="utf-8")
    return cfg_path


def _reload_policy(cfg_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("LITELLM_EXT_CONFIG_PATH", str(cfg_path))
    import litellm_ext.core.config as cfg  # type: ignore
    import litellm_ext.policy as policy  # type: ignore

    importlib.reload(cfg)
    importlib.reload(policy)


def test_model_mapping_reasoning_overrides_family(tmp_path: Path, monkeypatch):
    cfg_path = _write_cfg(
        tmp_path,
        """version: 1
policy:
  model_mapping:
    enabled: true
    reasoning_model: claude-reasoner
    family_overrides:
      sonnet: sonnet-mapped
    default_model: default-mapped
""",
    )
    _reload_policy(cfg_path, monkeypatch)

    import litellm_ext.policy as policy  # type: ignore

    payload: Dict[str, Any] = {
        "model": "claude-sonnet-4-5",
        "thinking": {"type": "enabled"},
        "messages": [{"role": "user", "content": "hi"}],
    }
    policy.enforce(payload, allow_raise=True)
    assert payload["model"] == "claude-reasoner"


def test_model_mapping_family_match(tmp_path: Path, monkeypatch):
    cfg_path = _write_cfg(
        tmp_path,
        """version: 1
policy:
  model_mapping:
    enabled: true
    family_overrides:
      sonnet: sonnet-mapped
""",
    )
    _reload_policy(cfg_path, monkeypatch)

    import litellm_ext.policy as policy  # type: ignore

    payload = {"model": "claude-sonnet-4-5", "messages": [{"role": "user", "content": "hi"}]}
    policy.enforce(payload, allow_raise=True)
    assert payload["model"] == "sonnet-mapped"


def test_model_mapping_default_fallback(tmp_path: Path, monkeypatch):
    cfg_path = _write_cfg(
        tmp_path,
        """version: 1
policy:
  model_mapping:
    enabled: true
    default_model: default-mapped
""",
    )
    _reload_policy(cfg_path, monkeypatch)

    import litellm_ext.policy as policy  # type: ignore

    payload = {"model": "random-model", "messages": [{"role": "user", "content": "hi"}]}
    policy.enforce(payload, allow_raise=True)
    assert payload["model"] == "default-mapped"


def test_model_mapping_patterns_limit_scope(tmp_path: Path, monkeypatch):
    cfg_path = _write_cfg(
        tmp_path,
        """version: 1
policy:
  model_mapping:
    enabled: true
    apply_to_patterns:
      - claude-*
    default_model: default-mapped
""",
    )
    _reload_policy(cfg_path, monkeypatch)

    import litellm_ext.policy as policy  # type: ignore

    payload = {"model": "not-claude", "messages": [{"role": "user", "content": "hi"}]}
    policy.enforce(payload, allow_raise=True)
    assert payload["model"] == "not-claude"


def test_compact_route_takes_precedence_over_mapping(tmp_path: Path, monkeypatch):
    cfg_path = _write_cfg(
        tmp_path,
        """version: 1
policy:
  compact_routing:
    enabled: true
    target_model: glm-4.7
  model_mapping:
    enabled: true
    default_model: default-mapped
""",
    )
    _reload_policy(cfg_path, monkeypatch)

    import litellm_ext.policy as policy  # type: ignore

    payload = {
        "model": "claude-sonnet-4-5",
        "messages": [{"role": "user", "content": "<command-name>/compact</command-name>"}],
    }
    policy.enforce(payload, allow_raise=True)
    assert payload["model"] == "glm-4.7"


def test_compact_already_target_still_takes_precedence_over_mapping(tmp_path: Path, monkeypatch):
    cfg_path = _write_cfg(
        tmp_path,
        """version: 1
policy:
  compact_routing:
    enabled: true
    target_model: glm-4.7
  model_mapping:
    enabled: true
    default_model: default-mapped
""",
    )
    _reload_policy(cfg_path, monkeypatch)

    import litellm_ext.policy as policy  # type: ignore

    payload = {
        "model": "GLM-4.7",
        "messages": [{"role": "user", "content": "<command-name>/compact</command-name>"}],
    }
    policy.enforce(payload, allow_raise=True)
    assert payload["model"] == "GLM-4.7"
