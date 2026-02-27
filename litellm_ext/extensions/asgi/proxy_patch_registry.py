from __future__ import annotations

"""Central registry to patch litellm.proxy.proxy_server safely."""

import importlib.abc
import importlib.machinery
import importlib.util
import sys
import threading
from typing import Callable, Optional, Sequence, Tuple, List

from ...core.logging import format_log_line

TARGET_MOD = "litellm.proxy.proxy_server"

def _log_error(msg: str) -> None:
    print(format_log_line("litellm_ext.proxy_patch_registry", msg), file=sys.stderr, flush=True)

_PATCH_ATTR = "_litellm_ext_proxy_patch_registry_applied"
_FINDER_INSTALLED = False
_LOCK = threading.RLock()

PatchFn = Callable[[object], None]
_PatchEntry = Tuple[int, str, PatchFn]
_PATCHERS: List[_PatchEntry] = []

_BANNER_EMITTED = False


def register(name: str, fn: PatchFn, *, order: int = 100) -> None:
    if not callable(fn):
        raise TypeError("proxy_patch_registry register() requires a callable")
    with _LOCK:
        # de-dupe by name
        global _PATCHERS
        _PATCHERS = [e for e in _PATCHERS if e[1] != name]
        _PATCHERS.append((int(order), str(name), fn))
        _PATCHERS.sort(key=lambda e: (e[0], e[1]))

    mod = sys.modules.get(TARGET_MOD)
    if mod is not None:
        _apply(mod)


def _applied_set(module) -> set[str]:
    s = getattr(module, _PATCH_ATTR, None)
    if isinstance(s, set):
        return s
    s = set()
    setattr(module, _PATCH_ATTR, s)
    return s


def _emit_banner(patchers: list[_PatchEntry]) -> None:
    global _BANNER_EMITTED
    if _BANNER_EMITTED:
        return
    names = [name for _order, name, _fn in patchers]
    _log_error("proxy patches registered: " + ", ".join(names))
    _BANNER_EMITTED = True


def _apply(module) -> None:
    applied = _applied_set(module)
    with _LOCK:
        patchers = list(_PATCHERS)
    _emit_banner(patchers)
    for _order, name, fn in patchers:
        if name in applied:
            continue
        try:
            fn(module)
            applied.add(name)
        except ModuleNotFoundError as e:
            # Only suppress missing litellm during early sitecustomize boot.
            if getattr(e, "name", None) != "litellm":
                raise
            _log_error(f"patcher failed name={name}: {type(e).__name__}: {e}")
            continue
        except Exception as e:
            _log_error(f"patcher failed name={name}: {type(e).__name__}: {e}")
            continue


class _PatchFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname: str, path: Optional[Sequence[str]], target=None):
        if fullname != TARGET_MOD:
            return None

        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        if not spec or not spec.loader:
            return spec

        original_loader = spec.loader

        class _PatchedLoader(importlib.abc.Loader):
            def create_module(self, spec_):
                create = getattr(original_loader, "create_module", None)
                return create(spec_) if callable(create) else None

            def exec_module(self, module):
                execm = getattr(original_loader, "exec_module", None)
                if not callable(execm):
                    raise ImportError(f"Loader for {fullname} has no exec_module()")
                execm(module)
                _apply(module)

        spec.loader = _PatchedLoader()
        return spec


def install() -> None:
    global _FINDER_INSTALLED

    mod = sys.modules.get(TARGET_MOD)
    if mod is not None:
        _apply(mod)
        return

    spec = importlib.util.find_spec(TARGET_MOD)
    if spec is None:
        # LiteLLM not importable in this process; skip to avoid sitecustomize errors.
        return

    if _FINDER_INSTALLED:
        return

    for f in sys.meta_path:
        if isinstance(f, _PatchFinder):
            _FINDER_INSTALLED = True
            return

    sys.meta_path.insert(0, _PatchFinder())
    _FINDER_INSTALLED = True
