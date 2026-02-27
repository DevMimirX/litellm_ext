from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from ..core.logging import format_log_line


_LOGGER_NAME = "litellm_ext.agent_config_apply"


def log(message: str, *, quiet: bool = False) -> None:
    if quiet:
        return
    print(format_log_line(_LOGGER_NAME, message), file=sys.stderr, flush=True)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _claude_target_path() -> Path:
    config_dir = os.environ.get("CLAUDE_CONFIG_DIR")
    if config_dir:
        return Path(config_dir).expanduser() / "settings.json"
    return Path.home() / ".claude" / "settings.json"


@dataclass(frozen=True)
class AgentConfigAdapter:
    name: str
    description: str
    source_env_vars: tuple[str, ...]
    source_default_paths: tuple[Path, ...]
    target_resolver: Callable[[], Path]
    option_env_prefix: str

    def resolve_source(self, explicit_path: Path | None, *, quiet: bool = False) -> Path | None:
        if explicit_path is not None:
            path = explicit_path.expanduser()
            if path.exists():
                return path
            log(f"Warning: config path does not exist: {path}", quiet=quiet)
            return None

        stale_env_overrides: list[tuple[str, Path]] = []
        for env_var in self.source_env_vars:
            value = os.environ.get(env_var)
            if not value:
                continue
            path = Path(value).expanduser()
            if path.exists():
                return path
            log(f"Warning: {env_var} points to non-existent file: {path}", quiet=quiet)
            stale_env_overrides.append((env_var, path))
            # Fall back to adapter defaults when env var points at stale paths.
            continue

        for path in self.source_default_paths:
            if path.exists():
                if stale_env_overrides and not quiet:
                    names = ", ".join(name for name, _ in stale_env_overrides)
                    log(
                        f"Info: Falling back to default source config for {self.name}: {path} "
                        f"(stale override env: {names})",
                        quiet=quiet,
                    )
                return path
        return None

    def resolve_target(self, explicit_path: Path | None) -> Path:
        if explicit_path is not None:
            return explicit_path.expanduser()
        return self.target_resolver()


CLAUDE_ADAPTER = AgentConfigAdapter(
    name="claude",
    description="Claude Code settings",
    source_env_vars=("LITELLM_EXT_CLAUDE_CONFIG",),
    source_default_paths=(
        _repo_root() / "config" / "claude.settings.json",
        _repo_root().parent / "config" / "claude.settings.json",
    ),
    target_resolver=_claude_target_path,
    option_env_prefix="LITELLM_EXT_CLAUDE_APPLY",
)


ADAPTERS: dict[str, AgentConfigAdapter] = {
    CLAUDE_ADAPTER.name: CLAUDE_ADAPTER,
}
