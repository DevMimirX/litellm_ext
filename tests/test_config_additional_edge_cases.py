from __future__ import annotations

import importlib
from pathlib import Path

import pytest


def test_cfg_bool_parses_strings(tmp_path: Path, monkeypatch):
    cfg_path = tmp_path / "extensions.yaml"
    cfg_path.write_text(
        """extensions:
  httpx_registry: {enabled: true}
  count_tokens_stub: {enabled: true}
  hard_caps: {enabled: true}
  streaming_sse: {enabled: true}
policy:
  safety_buffer_tokens: 1024
  overflow_policy: reduce_then_trim
  min_tail_messages: 6
  max_trim_steps: 200
  model_limits:
    "*": {max_output: 16000, max_context: null}
compact:
  enabled: "true"
  target_model: glm-4.7
  patterns: []
"""
    )
    monkeypatch.setenv("LITELLM_EXT_CONFIG_PATH", str(cfg_path))

    import litellm_ext.core.config as cfg  # type: ignore
    importlib.reload(cfg)

    assert cfg.get_bool("compact", "enabled", default=False) is True


def test_missing_config_path_raises_clear_error(tmp_path: Path, monkeypatch):
    # Force a missing file path
    missing = tmp_path / "does_not_exist.yaml"
    monkeypatch.setenv("LITELLM_EXT_CONFIG_PATH", str(missing))

    import litellm_ext.core.config as cfg  # type: ignore
    importlib.reload(cfg)

    with pytest.raises(FileNotFoundError) as ei:
        cfg.load_config(force=True)
    assert "Config file not found" in str(ei.value)


def test_multiple_configs_raise_without_env(tmp_path: Path, monkeypatch):
    cwd_dir = tmp_path / "cwd"
    cwd_dir.mkdir()
    (cwd_dir / "extensions.yaml").write_text("version: 1\n", encoding="utf-8")

    monkeypatch.chdir(cwd_dir)
    monkeypatch.delenv("LITELLM_EXT_CONFIG_PATH", raising=False)

    import litellm_ext.core.config as cfg  # type: ignore
    importlib.reload(cfg)

    with pytest.raises(RuntimeError) as ei:
        cfg.load_config(force=True)
    assert "Multiple config files" in str(ei.value)


def test_config_debug_flag_from_yaml(tmp_path: Path, monkeypatch, capsys):
    cfg_path = tmp_path / "extensions.yaml"
    cfg_path.write_text("version: 1\ndebug:\n  config: true\n", encoding="utf-8")
    monkeypatch.setenv("LITELLM_EXT_CONFIG_PATH", str(cfg_path))

    import litellm_ext.core.config as cfg  # type: ignore
    import importlib
    importlib.reload(cfg)

    cfg.load_config(force=True)
    captured = capsys.readouterr()
    assert "[litellm_ext.config]" in captured.err
