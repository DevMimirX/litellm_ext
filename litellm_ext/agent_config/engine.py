from __future__ import annotations

import json
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

from ..core.logging import format_log_line


_LOGGER_NAME = "litellm_ext.agent_config_apply"


@dataclass(frozen=True)
class ApplyOptions:
    backup: bool = True
    strategy: str = "overwrite"
    if_changed: bool = False
    quiet: bool = False


@dataclass(frozen=True)
class ApplyResult:
    changed: bool
    source_path: Path
    target_path: Path
    backup_path: Path | None = None


def _log(message: str, *, quiet: bool = False) -> None:
    if quiet:
        return
    print(format_log_line(_LOGGER_NAME, message), file=sys.stderr, flush=True)


def merge_config(existing: object, override: object) -> object:
    """Recursively merge JSON objects where override values win."""
    if isinstance(existing, dict) and isinstance(override, dict):
        merged = dict(existing)
        for key, value in override.items():
            merged[key] = merge_config(existing.get(key), value)
        return merged
    return override


def load_json_object(path: Path, *, label: str) -> dict[str, object]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"{label} not found: {path}") from None
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in {label}: {e}") from e

    if not isinstance(data, dict):
        raise ValueError(f"{label} must contain a JSON object at the top level.")
    return data


def apply_json_config(
    source_path: Path,
    target_path: Path,
    *,
    options: ApplyOptions,
) -> ApplyResult:
    source_config = load_json_object(source_path, label="source config")

    target_path.parent.mkdir(parents=True, exist_ok=True)

    existing_config: dict[str, object] = {}
    existing_raw: object | None = None
    if target_path.exists():
        try:
            with open(target_path, "r", encoding="utf-8") as f:
                existing_raw = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in target config: {e}") from e
        except OSError as e:
            raise OSError(f"Could not read target config: {e}") from e
        if isinstance(existing_raw, dict):
            existing_config = existing_raw

    if options.strategy == "merge":
        if existing_raw is not None and not isinstance(existing_raw, dict):
            raise ValueError("Target config must contain a JSON object for merge strategy.")
        target_config = merge_config(existing_config, source_config)
    elif options.strategy == "overwrite":
        target_config = source_config
    else:
        raise ValueError(f"Unsupported strategy: {options.strategy}")

    if options.if_changed and existing_raw == target_config:
        _log("No changes: settings already up to date.", quiet=options.quiet)
        return ApplyResult(
            changed=False,
            source_path=source_path,
            target_path=target_path,
        )

    backup_path: Path | None = None
    if options.backup and target_path.exists():
        backup_path = target_path.with_suffix(".json.bak")
        try:
            shutil.copy2(target_path, backup_path)
            _log(f"Backed up existing settings to: {backup_path}", quiet=options.quiet)
        except OSError as e:
            print(f"Warning: Could not create backup: {e}", file=sys.stderr)
            backup_path = None

    try:
        with open(target_path, "w", encoding="utf-8") as f:
            json.dump(target_config, f, indent=2)
            f.write("\n")
    except OSError as e:
        raise OSError(f"Could not write target config: {e}") from e

    _log(f"Applied settings from: {source_path}", quiet=options.quiet)
    _log(f"Target: {target_path}", quiet=options.quiet)
    return ApplyResult(
        changed=True,
        source_path=source_path,
        target_path=target_path,
        backup_path=backup_path,
    )
