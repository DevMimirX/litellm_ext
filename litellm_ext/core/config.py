from __future__ import annotations

"""YAML-backed configuration loader for LiteLLM extensions.

Lookup order for config file:
1) $LITELLM_EXT_CONFIG_PATH
2) <repo>/config/extensions.yaml (relative to this module)
3) <repo>/extensions.yaml
4) ./config/extensions.yaml
5) ./extensions.yaml

If multiple distinct config files are found and no env override is set,
loading fails fast to avoid ambiguous configuration.

The config is cached and hot-reloaded when the file mtime changes.
"""

import os
import sys
from datetime import datetime
from pathlib import Path
from threading import RLock
from typing import Any, Dict, Iterable, Optional

import yaml

_DEBUG_ENV = os.environ.get("LITELLM_EXT_CONFIG_DEBUG")

_TRUE = {"1", "true", "yes", "y", "on"}
_FALSE = {"0", "false", "no", "n", "off"}


def _boolish(val: object) -> bool | None:
    if isinstance(val, bool):
        return val
    if isinstance(val, (int, float)):
        return bool(val)
    if isinstance(val, str):
        s = val.strip().lower()
        if s in _TRUE:
            return True
        if s in _FALSE:
            return False
    return None


def _debug_enabled(data: Optional[Dict[str, Any]] = None) -> bool:
    env = _boolish(_DEBUG_ENV)
    if env is not None:
        return env
    if data is None:
        data = _CACHED_CFG
    if isinstance(data, dict):
        dbg_cfg = data.get("debug")
        if isinstance(dbg_cfg, dict):
            dbg = dbg_cfg.get("config")
        else:
            dbg = dbg_cfg
        val = _boolish(dbg)
        if val is not None:
            return val
    return False

DEFAULT_ENV_VARS = ("LITELLM_EXT_CONFIG_PATH",)
DEFAULT_FILENAMES = ("config/extensions.yaml", "extensions.yaml")

_LOCK = RLock()
_CACHED_PATH: Optional[Path] = None
_CACHED_SOURCE: Optional[str] = None
_CACHED_MTIME: Optional[float] = None
_CACHED_CFG: Dict[str, Any] = {}


def _dbg(msg: str, *, enabled: Optional[bool] = None) -> None:
    if enabled is None:
        enabled = _debug_enabled()
    if enabled:
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        print(f"{ts} [litellm_ext.config] {msg}", file=sys.stderr, flush=True)


def _iter_candidate_paths_with_source() -> Iterable[tuple[Path, str]]:
    for env in DEFAULT_ENV_VARS:
        env_path = os.environ.get(env)
        if env_path:
            yield Path(env_path).expanduser(), f"env:{env}"
            return

    module_dir = Path(__file__).resolve().parent
    repo_root = module_dir.parent.parent

    for name in DEFAULT_FILENAMES:
        yield repo_root / name, f"repo:{name}"

    cwd = Path.cwd()
    for name in DEFAULT_FILENAMES:
        yield cwd / name, f"cwd:{name}"

    for name in DEFAULT_FILENAMES:
        yield module_dir / name, f"module:{name}"


def _resolve_config_path() -> tuple[Path, str, list[tuple[Path, str]]]:
    first: tuple[Path, str] | None = None
    chosen: tuple[Path, str] | None = None
    existing: list[tuple[Path, str]] = []
    seen: set[str] = set()
    env_set = any(os.environ.get(env) for env in DEFAULT_ENV_VARS)

    for path, source in _iter_candidate_paths_with_source():
        if first is None:
            first = (path, source)
        if path.exists():
            try:
                key = str(path.resolve())
            except Exception:
                key = str(path)
            if key not in seen:
                existing.append((path, source))
                seen.add(key)
            if chosen is None:
                chosen = (path, source)

    if chosen is not None:
        if not env_set and len(existing) > 1:
            detail = ", ".join(f"{p} ({src})" for p, src in existing)
            raise RuntimeError(
                "Multiple config files found; set LITELLM_EXT_CONFIG_PATH to choose one. "
                f"Found: {detail}"
            )
        return chosen[0], chosen[1], existing

    if first is None:
        raise FileNotFoundError("Config search yielded no candidates")

    return first[0], first[1], existing


def load_config(*, force: bool = False) -> Dict[str, Any]:
    """Load (and cache) the YAML config.

    Raises FileNotFoundError if the resolved config path does not exist.
    """
    global _CACHED_PATH, _CACHED_SOURCE, _CACHED_MTIME, _CACHED_CFG

    with _LOCK:
        path, source, existing = _resolve_config_path()
        if not path.exists():
            raise FileNotFoundError(
                f"Config file not found at {str(path)!r}. "
                "Set LITELLM_EXT_CONFIG_PATH or create the file."
            )

        try:
            mtime = path.stat().st_mtime
        except Exception:
            mtime = None

        if not force and _CACHED_PATH == path and _CACHED_MTIME == mtime and _CACHED_PATH is not None:
            return _CACHED_CFG

        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        if data is None:
            data = {}
        if not isinstance(data, dict):
            raise TypeError("Config root must be a mapping/dict")

        _CACHED_PATH = path
        _CACHED_SOURCE = source
        _CACHED_MTIME = mtime
        _CACHED_CFG = data
        _dbg(
            f"loaded config path={path} source={source} keys={list(data.keys())} "
            f"mtime={mtime} bytes={path.stat().st_size if path.exists() else 'n/a'} "
            f"candidates={len(existing)}",
            enabled=_debug_enabled(data),
        )
        return _CACHED_CFG


def get(*keys: str, default: Any = None) -> Any:
    """Fetch a nested config value by path segments."""
    cfg = load_config()
    cur: Any = cfg
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def get_bool(*keys: str, default: bool = False) -> bool:
    v = get(*keys, default=default)
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return bool(v)
    if isinstance(v, str):
        s = v.strip().lower()
        if s in {"1", "true", "yes", "y", "on"}:
            return True
        if s in {"0", "false", "no", "n", "off"}:
            return False
    return bool(default)


def get_int(*keys: str, default: int = 0) -> int:
    v = get(*keys, default=default)
    try:
        return int(v)
    except Exception:
        return int(default)


def get_str(*keys: str, default: str = "") -> str:
    v = get(*keys, default=default)
    if v is None:
        return default
    return str(v)


def get_dict(*keys: str, default: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    v = get(*keys, default=default or {})
    return v if isinstance(v, dict) else (default or {})



def get_config_source() -> Optional[str]:
    if _CACHED_SOURCE is not None:
        return _CACHED_SOURCE
    try:
        _path, source, _existing = _resolve_config_path()
        return source
    except Exception:
        return None


def get_config_candidates() -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    seen: set[str] = set()
    try:
        for path, source in _iter_candidate_paths_with_source():
            if not path.exists():
                continue
            try:
                key = str(path.resolve())
            except Exception:
                key = str(path)
            if key in seen:
                continue
            seen.add(key)
            out.append((str(path), source))
    except Exception:
        return out
    return out


def get_list(*keys: str, default: Optional[list] = None) -> list:
    v = get(*keys, default=default or [])
    return v if isinstance(v, list) else (default or [])
