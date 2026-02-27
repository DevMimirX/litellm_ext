from __future__ import annotations

import os
import sys
from datetime import datetime
from typing import Callable, Iterable, Optional

from .config import get_bool


TRUE_SET = {"1", "true", "yes", "y", "on"}
FALSE_SET = {"0", "false", "no", "n", "off"}


def env_flag(name: str) -> Optional[bool]:
    val = os.environ.get(name)
    if val is None:
        return None
    s = val.strip().lower()
    if s in TRUE_SET:
        return True
    if s in FALSE_SET:
        return False
    return None


def env_flag_any(names: Iterable[str]) -> Optional[bool]:
    for n in names:
        v = env_flag(n)
        if v is not None:
            return v
    return None


class PatchLogger:
    def __init__(self, name: str, enabled_fn: Callable[[], bool]) -> None:
        self._name = name
        self._enabled_fn = enabled_fn

    def enabled(self) -> bool:
        try:
            return bool(self._enabled_fn())
        except Exception:
            return False

    def debug(self, msg: str) -> None:
        if self.enabled():
            print(format_log_line(self._name, msg), file=sys.stderr, flush=True)


def debug_flag(name: str, envs: Iterable[str], config_path: tuple[str, ...], default: bool) -> bool:
    env_val = env_flag_any(envs)
    if env_val is not None:
        return env_val
    return get_bool(*config_path, default=default)


def _timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]


def format_log_line(name: str, msg: str) -> str:
    return f"{_timestamp()} [{name}] {msg}"
