"""Runtime patch for LiteLLM passthrough streaming logging."""

from __future__ import annotations

import importlib.abc
import importlib.machinery
import sys
from typing import List, Optional, Sequence

from ...core.patch import PatchSettings

TARGET_MOD = "litellm.proxy.pass_through_endpoints.streaming_handler"

SETTINGS = PatchSettings(
    "streaming_sse",
    enabled_default=True,
    debug_default=False,
)
_LOG = SETTINGS.logger("litellm_ext.streaming_sse")

_PATCH_FLAG = "_litellm_ext_streaming_sse_applied"
_FINDER_INSTALLED = False


def _filter_sse_lines_from_bytes(raw_bytes: List[bytes]) -> List[str]:
    if not raw_bytes:
        return []

    text = b"".join(raw_bytes).decode("utf-8", errors="replace")

    out: List[str] = []
    buf = ""

    for part in text.splitlines(True):
        buf += part
        if not buf.endswith("\n") and not buf.endswith("\r"):
            continue

        line = buf.strip()
        buf = ""

        if not line or line.startswith(":"):
            continue

        if line.startswith("data:"):
            payload = line[5:].strip()
            if payload and payload != "[DONE]":
                out.append("data: " + payload)
            continue

        if line[:1] in ("{", "["):
            out.append("data: " + line)
            continue

    if buf:
        line = buf.strip()
        if line and not line.startswith(":"):
            if line.startswith("data:"):
                payload = line[5:].strip()
                if payload and payload != "[DONE]":
                    out.append("data: " + payload)
            elif line[:1] in ("{", "["):
                out.append("data: " + line)

    return out


def _patch_streaming_handler(module) -> None:
    if getattr(module, _PATCH_FLAG, False):
        return

    handler = getattr(module, "PassThroughStreamingHandler", None)
    if handler is None:
        handler = getattr(module, "StreamingHandler", None)
    if handler is None:
        _LOG.debug(f"streaming_handler imported but no handler found; skipping {TARGET_MOD}")
        setattr(module, _PATCH_FLAG, True)
        return

    if getattr(handler, "_litellm_ext_streaming_sse", False):
        setattr(module, _PATCH_FLAG, True)
        return

    # Patch line conversion (name varies across LiteLLM versions)
    if hasattr(handler, "_convert_raw_bytes_to_str_lines"):
        orig = handler._convert_raw_bytes_to_str_lines

        def patched(raw_bytes: List[bytes]):
            try:
                return _filter_sse_lines_from_bytes(raw_bytes)
            except Exception as e:
                _LOG.debug(f"SSE filter failed: {type(e).__name__}: {e}")
                return orig(raw_bytes)

        handler._convert_raw_bytes_to_str_lines = staticmethod(patched)  # type: ignore[assignment]
        _LOG.debug("patched PassThroughStreamingHandler._convert_raw_bytes_to_str_lines")
    elif hasattr(handler, "_stream_to_sse_lines"):
        orig = handler._stream_to_sse_lines

        def patched(self, raw_bytes: List[bytes]):
            try:
                return _filter_sse_lines_from_bytes(raw_bytes)
            except Exception as e:
                _LOG.debug(f"SSE filter failed: {type(e).__name__}: {e}")
                return orig(self, raw_bytes)

        handler._stream_to_sse_lines = patched  # type: ignore[assignment]
        _LOG.debug("patched StreamingHandler._stream_to_sse_lines")

    # Patch logging to suppress exceptions
    if hasattr(handler, "_route_streaming_logging_to_handler"):
        orig = handler._route_streaming_logging_to_handler

        async def patched_route(*args, **kwargs):
            try:
                return await orig(*args, **kwargs)
            except Exception as e:
                _LOG.debug(f"streaming logging failed: {type(e).__name__}: {e}")
                return None

        handler._route_streaming_logging_to_handler = staticmethod(patched_route)  # type: ignore[assignment]
        _LOG.debug("patched streaming logging handler")

    setattr(handler, "_litellm_ext_streaming_sse", True)
    setattr(module, _PATCH_FLAG, True)


class _PatchFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname: str, path: Optional[Sequence[str]], target=None):
        if fullname != TARGET_MOD:
            return None

        try:
            enabled = SETTINGS.is_enabled()
        except Exception:
            enabled = False

        if not enabled:
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
                _patch_streaming_handler(module)

        spec.loader = _PatchedLoader()
        return spec


def install() -> None:
    global _FINDER_INSTALLED

    if not SETTINGS.is_enabled():
        _LOG.debug("disabled (streaming_sse patch not enabled)")
        return

    mod = sys.modules.get(TARGET_MOD)
    if mod is not None:
        _patch_streaming_handler(mod)
        return

    if _FINDER_INSTALLED:
        return

    for f in sys.meta_path:
        if isinstance(f, _PatchFinder):
            _FINDER_INSTALLED = True
            return

    sys.meta_path.insert(0, _PatchFinder())
