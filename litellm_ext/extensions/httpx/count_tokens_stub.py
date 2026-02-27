from __future__ import annotations

import json
import math
import os
import tempfile
from typing import Any, Dict, Optional, Tuple

import httpx

from ...core.config import get_list, get_str
from ...core.patch import PatchSettings
from ...core.registry import install_httpx_patch, register_request_mutator
from ...policy import autocompact_multiplier_for_model, estimate_input_tokens_heuristic

SETTINGS = PatchSettings(
    "count_tokens_stub",
    enabled_default=True,
    debug_default=False,
    enabled_envs=("LITELLM_EXT_COUNT_TOKENS_STUB",),
    debug_envs=("LITELLM_EXT_COUNT_TOKENS_STUB_DEBUG",),
)
_LOG = SETTINGS.logger("litellm_ext.count_tokens_stub")

_env_host = os.environ.get("LITELLM_EXT_COUNT_TOKENS_HOST_SUBSTR")
_HOST_SUBSTR = (
    _env_host.strip().lower()
    if _env_host is not None and _env_host.strip() != ""
    else get_str("extensions", "count_tokens_stub", "host_substr", default="").strip().lower()
)

_TARGET_SUFFIXES_RAW = get_list("extensions", "count_tokens_stub", "target_suffixes", default=[])
if isinstance(_TARGET_SUFFIXES_RAW, list) and all(isinstance(x, str) for x in _TARGET_SUFFIXES_RAW) and _TARGET_SUFFIXES_RAW:
    _TARGET_SUFFIXES: Tuple[str, ...] = tuple(_TARGET_SUFFIXES_RAW)
else:
    _TARGET_SUFFIXES = (
        "/v1/messages/count_tokens",
        "/messages/count_tokens",
        "/count_tokens",
    )

JsonDict = Dict[str, Any]

# Global flag to track if we've already installed (prevents race conditions in same process)
_INSTALL_CALLED = False

# File-based marker for cross-process idempotency (handles conda run subprocesses)
def _get_install_marker_path() -> str:
    """Get path to a file-based marker for cross-process install tracking."""
    root_pid = os.environ.get("LITELLM_EXT_INSTALL_PID") or str(os.getpid())
    return os.path.join(tempfile.gettempdir(), f"litellm_ext_count_tokens_installed_{root_pid}")


def _is_installed_marker_present() -> bool:
    """Check if install marker file exists (indicates install already done in this session)."""
    try:
        return os.path.exists(_get_install_marker_path())
    except Exception:
        return False


def _set_installed_marker() -> None:
    """Create install marker file to indicate install is complete."""
    try:
        with open(_get_install_marker_path(), "w") as f:
            f.write("1")
    except Exception:
        pass


def _req_summary(request: httpx.Request) -> str:
    try:
        url = request.url
        host = getattr(url, "host", "")
        path = getattr(url, "path", "")
        method = request.method.upper() if request.method else ""
        return f"{method} {host}{path}"
    except Exception:
        return "<request>"


def _is_target_request(request: httpx.Request) -> bool:
    try:
        url = request.url
        host = (getattr(url, "host", "") or "").lower()
        path = getattr(url, "path", "") or ""
        if _HOST_SUBSTR and _HOST_SUBSTR not in host:
            return False
        return any(path.endswith(s) for s in _TARGET_SUFFIXES)
    except Exception:
        return False


def _get_request_body_bytes(request: httpx.Request) -> bytes:
    content = getattr(request, "content", None)
    if isinstance(content, (bytes, bytearray)):
        return bytes(content)
    if isinstance(content, str):
        return content.encode("utf-8", errors="replace")
    return b""


def _try_parse_json_request(request: httpx.Request) -> JsonDict:
    body = _get_request_body_bytes(request)
    if not body:
        return {}
    b2 = body.lstrip()
    ctype = (request.headers.get("content-type") or "").lower()
    if "application/json" not in ctype and b2[:1] not in (b"{", b"["):
        return {}
    try:
        obj = json.loads(b2.decode("utf-8", errors="replace"))
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


def _make_stub_response(request: httpx.Request, tokens: int) -> httpx.Response:
    return httpx.Response(
        status_code=200,
        json={"input_tokens": int(tokens)},
        request=request,
        headers={"content-type": "application/json"},
    )


def _count_tokens_mutator(request: httpx.Request) -> Optional[httpx.Response]:
    if request.method.upper() != "POST":
        return None
    if not _is_target_request(request):
        return None

    body = _try_parse_json_request(request)
    n = estimate_input_tokens_heuristic(body)
    try:
        mult = autocompact_multiplier_for_model(body.get("model"))
    except Exception:
        mult = 1.0

    model = body.get("model") if isinstance(body, dict) else None

    if mult != 1.0:
        n2 = int(math.ceil(n * mult)) if n > 0 else n
        if n2 <= 0:
            n2 = 1
        _LOG.debug(
            f"stubbed count_tokens model={model!r} req={_req_summary(request)} "
            f"input_tokens≈{n} * {mult:g} => {n2}"
        )
        n = n2
    else:
        _LOG.debug(
            f"stubbed count_tokens model={model!r} req={_req_summary(request)} "
            f"input_tokens≈{n}"
        )

    return _make_stub_response(request, n)


def install() -> None:
    global _INSTALL_CALLED
    if not SETTINGS.is_enabled():
        _LOG.debug("disabled (count_tokens patch not enabled)")
        return

    # Check both in-process flag and file-based marker for cross-process idempotency
    # This handles cases like `conda run` which spawns subprocesses
    if _INSTALL_CALLED or _is_installed_marker_present():
        return

    install_httpx_patch()
    register_request_mutator("count_tokens_stub", _count_tokens_mutator, priority=10)
    _INSTALL_CALLED = True
    _set_installed_marker()
    _LOG.debug(
        "installed count_tokens stub "
        f"host_substr={_HOST_SUBSTR!r} target_suffixes={list(_TARGET_SUFFIXES)}"
    )
