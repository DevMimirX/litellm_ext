from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

from .config import get_bool
from .logging import env_flag_any, PatchLogger, debug_flag


def _default_envs(name: str, suffix: str = "") -> tuple[str, ...]:
    key = name.upper()
    return (
        f"LITELLM_EXT_{key}{suffix}",
    )


@dataclass(frozen=True)
class PatchSettings:
    name: str
    enabled_default: bool = True
    debug_default: bool = False
    enabled_path: Optional[tuple[str, ...]] = None
    debug_path: Optional[tuple[str, ...]] = None
    enabled_envs: Optional[Iterable[str]] = None
    debug_envs: Optional[Iterable[str]] = None

    def is_enabled(self) -> bool:
        envs = tuple(self.enabled_envs or _default_envs(self.name))
        env_val = env_flag_any(envs)
        if env_val is not None:
            return env_val
        path = self.enabled_path or ("extensions", self.name, "enabled")
        return get_bool(*path, default=self.enabled_default)

    def is_debug(self) -> bool:
        envs = tuple(self.debug_envs or _default_envs(self.name, "_DEBUG"))
        path = self.debug_path or ("debug", self.name)
        return debug_flag(self.name, envs, path, self.debug_default)

    def logger(self, prefix: Optional[str] = None) -> PatchLogger:
        name = prefix or f"litellm_ext.{self.name}"
        return PatchLogger(name, self.is_debug)
