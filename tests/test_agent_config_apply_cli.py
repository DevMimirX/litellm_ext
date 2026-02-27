from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def run_apply(
    tmp_path: Path,
    *args: str,
    env_overrides: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["CLAUDE_CONFIG_DIR"] = str(tmp_path / "claude")
    if env_overrides:
        env.update(env_overrides)
    return subprocess.run(
        [sys.executable, "-m", "litellm_ext.agent_config.cli", *args],
        capture_output=True,
        text=True,
        env=env,
        cwd=REPO_ROOT,
        check=False,
    )


def test_list_tools_includes_claude(tmp_path: Path) -> None:
    result = run_apply(tmp_path, "--list-tools")
    assert result.returncode == 0
    assert "claude" in result.stdout


def test_tool_flag_applies_claude_merge(tmp_path: Path) -> None:
    source = tmp_path / "claude.settings.json"
    source.write_text(json.dumps({"theme": "repo", "nested": {"a": 1}}), encoding="utf-8")

    settings_path = tmp_path / "claude" / "settings.json"
    settings_path.parent.mkdir(parents=True, exist_ok=True)
    settings_path.write_text(json.dumps({"user": "dev_user", "nested": {"b": 2}}), encoding="utf-8")

    result = run_apply(
        tmp_path,
        "--tool",
        "claude",
        "--config",
        str(source),
        "--strategy",
        "merge",
        "--no-backup",
    )
    assert result.returncode == 0
    merged = json.loads(settings_path.read_text(encoding="utf-8"))
    assert merged == {"user": "dev_user", "theme": "repo", "nested": {"b": 2, "a": 1}}


def test_tool_specific_env_strategy_overrides_cli(tmp_path: Path) -> None:
    source = tmp_path / "claude.settings.json"
    source.write_text(json.dumps({"theme": "repo"}), encoding="utf-8")

    settings_path = tmp_path / "claude" / "settings.json"
    settings_path.parent.mkdir(parents=True, exist_ok=True)
    settings_path.write_text(json.dumps({"user": "dev_user"}), encoding="utf-8")

    result = run_apply(
        tmp_path,
        "--tool",
        "claude",
        "--config",
        str(source),
        "--strategy",
        "overwrite",
        "--no-backup",
        env_overrides={"LITELLM_EXT_CLAUDE_APPLY_STRATEGY": "merge"},
    )
    assert result.returncode == 0
    merged = json.loads(settings_path.read_text(encoding="utf-8"))
    assert merged == {"user": "dev_user", "theme": "repo"}


def test_optional_missing_config_returns_success(tmp_path: Path) -> None:
    missing = tmp_path / "missing.json"
    result = run_apply(
        tmp_path,
        "--tool",
        "claude",
        "--config",
        str(missing),
        "--optional",
        "--quiet",
    )
    assert result.returncode == 0
    assert not (tmp_path / "claude" / "settings.json").exists()


def test_if_changed_skips_backup_and_write(tmp_path: Path) -> None:
    source = tmp_path / "claude.settings.json"
    source_data = {"theme": "repo", "nested": {"a": 1}}
    source.write_text(json.dumps(source_data), encoding="utf-8")

    settings_path = tmp_path / "claude" / "settings.json"
    settings_path.parent.mkdir(parents=True, exist_ok=True)
    settings_path.write_text(json.dumps(source_data), encoding="utf-8")

    result = run_apply(
        tmp_path,
        "--tool",
        "claude",
        "--config",
        str(source),
        "--if-changed",
    )
    assert result.returncode == 0
    assert "No changes" in result.stderr
    assert not settings_path.with_suffix(".json.bak").exists()


def test_missing_env_source_falls_back_to_default(tmp_path: Path) -> None:
    result = run_apply(
        tmp_path,
        "--tool",
        "claude",
        "--no-backup",
        env_overrides={"LITELLM_EXT_CLAUDE_CONFIG": str(tmp_path / "missing.json")},
    )
    assert result.returncode == 0
    assert "Falling back to default source config for claude" in result.stderr
    assert (tmp_path / "claude" / "settings.json").exists()


def test_merge_rejects_non_object_target(tmp_path: Path) -> None:
    source = tmp_path / "claude.settings.json"
    source.write_text(json.dumps({"theme": "repo"}), encoding="utf-8")

    target = tmp_path / "target.json"
    target.write_text(json.dumps(["invalid"]), encoding="utf-8")

    result = run_apply(
        tmp_path,
        "--tool",
        "claude",
        "--config",
        str(source),
        "--target",
        str(target),
        "--strategy",
        "merge",
        "--no-backup",
    )

    assert result.returncode == 1
    assert "Target config must contain a JSON object for merge strategy." in result.stderr
