from __future__ import annotations

"""Reasoning replay patch for tool-calling + thinking-enabled models.

Captures reasoning_content on tool-call responses and re-injects it into
subsequent tool-call messages when required.
"""

import importlib.abc
import importlib.machinery
import json
import os
import sys
import threading
import time
from fnmatch import fnmatch
from typing import Any, Dict, List, Optional, Sequence, Tuple

import httpx

from ...core.config import get, get_list
from ...core.patch import PatchSettings
from ...core.registry import (
    install_httpx_patch,
    register_async_response_mutator,
    register_request_mutator,
    register_response_mutator,
)
from ...policy import normalize_model

TARGET_MOD = "litellm.proxy.proxy_server"

JsonDict = Dict[str, Any]

SETTINGS = PatchSettings(
    "reasoning_replay",
    enabled_default=True,
    debug_default=False,
    enabled_envs=("LITELLM_EXT_REASONING_REPLAY",),
    debug_envs=("LITELLM_EXT_REASONING_REPLAY_DEBUG",),
)
_LOG = SETTINGS.logger("litellm_ext.reasoning_replay")

_PATCH_FLAG = "_litellm_ext_reasoning_replay_applied"
_FINDER_INSTALLED = False

_MODEL_PATTERNS = get_list("extensions", "reasoning_replay", "model_patterns", default=["kimi-*"])
_PATH_SUFFIXES = tuple(
    get_list(
        "extensions",
        "reasoning_replay",
        "path_suffixes",
        default=["/v1/chat/completions", "/chat/completions"],
    )
)
def _get_int_config(key: str, default: int) -> int:
    val = get("extensions", "reasoning_replay", key, default=None)
    if val is None:
        return default
    try:
        return int(val)
    except (TypeError, ValueError):
        return default


_CACHE_TTL = _get_int_config("cache_ttl_seconds", 3600)
_CACHE_MAX = _get_int_config("cache_max_entries", 10000)
_FALLBACK_REASONING = get("extensions", "reasoning_replay", "fallback_reasoning", default=" ")
if not isinstance(_FALLBACK_REASONING, str):
    _FALLBACK_REASONING = " "

_LOCK = threading.RLock()
_CACHE: Dict[str, Tuple[float, str]] = {}


def _now() -> float:
    return time.time()


def _purge_locked(now: float) -> None:
    if _CACHE_TTL > 0:
        cutoff = now - float(_CACHE_TTL)
        for k, (ts, _) in list(_CACHE.items()):
            if ts < cutoff:
                _CACHE.pop(k, None)

    if _CACHE_MAX > 0 and len(_CACHE) > _CACHE_MAX:
        items = sorted(_CACHE.items(), key=lambda kv: kv[1][0])
        extra = len(items) - _CACHE_MAX
        for i in range(extra):
            _CACHE.pop(items[i][0], None)


def _cache_put(keys: List[str], reasoning: str) -> None:
    if not keys:
        return
    now = _now()
    with _LOCK:
        _purge_locked(now)
        for k in keys:
            if not k:
                continue
            _CACHE[k] = (now, reasoning)


def _cache_get(keys: List[str]) -> Optional[str]:
    if not keys:
        return None
    now = _now()
    with _LOCK:
        _purge_locked(now)
        for k in keys:
            if not k:
                continue
            v = _CACHE.get(k)
            if v is None:
                continue
            ts, reasoning = v
            if _CACHE_TTL > 0 and (now - ts) > float(_CACHE_TTL):
                _CACHE.pop(k, None)
                continue
            return reasoning
    return None


def _model_matches(model: Optional[str]) -> bool:
    if not model:
        return False
    mn = normalize_model(model)
    for pat in _MODEL_PATTERNS:
        try:
            if fnmatch(mn, pat) or fnmatch(str(model), pat):
                return True
        except Exception:
            continue
    return False


def _suffix_matches(path: str) -> bool:
    p = (path or "").rstrip("/")
    return any(p.endswith(s.rstrip("/")) for s in _PATH_SUFFIXES)


def _req_summary(request: httpx.Request) -> str:
    try:
        url = request.url
        host = getattr(url, "host", "")
        path = getattr(url, "path", "")
        method = request.method.upper() if request.method else ""
        return f"{method} {host}{path}"
    except Exception:
        return "<request>"


def _tool_keys(tool_call: JsonDict, idx: int) -> List[str]:
    out: List[str] = []

    tc_idx = tool_call.get("index")
    if isinstance(tc_idx, int):
        idx = tc_idx

    tcid = tool_call.get("id")
    if isinstance(tcid, str) and tcid:
        out.append(tcid)

    fn_name: Optional[str] = None
    fn = tool_call.get("function")
    if isinstance(fn, dict):
        n = fn.get("name")
        if isinstance(n, str) and n:
            fn_name = n

    if fn_name:
        out.append(f"{fn_name}:{idx}")
    else:
        out.append(f"tool:{idx}")

    seen = set()
    uniq: List[str] = []
    for k in out:
        if k not in seen:
            uniq.append(k)
            seen.add(k)
    return uniq


def _is_blank_reasoning(val: Any) -> bool:
    return not isinstance(val, str) or val.strip() == ""


def _store_from_openai_chat_completion_json(obj: JsonDict) -> int:
    choices = obj.get("choices")
    if not isinstance(choices, list):
        return 0

    stored = 0
    for choice in choices:
        if not isinstance(choice, dict):
            continue
        message = choice.get("message")
        if not isinstance(message, dict):
            continue
        tool_calls = message.get("tool_calls")
        if not isinstance(tool_calls, list) or not tool_calls:
            continue
        reasoning = message.get("reasoning_content")
        if _is_blank_reasoning(reasoning):
            continue

        for idx, tc in enumerate(tool_calls):
            if not isinstance(tc, dict):
                continue
            keys = _tool_keys(tc, idx)
            _cache_put(keys, str(reasoning))
            stored += 1

    if stored:
        _LOG.debug(f"stored reasoning for {stored} tool_calls")
    return stored


def _maybe_get_cache_keys(msg: JsonDict, idx: int) -> List[str]:
    tool_calls = msg.get("tool_calls")
    if not isinstance(tool_calls, list):
        return []
    keys: List[str] = []
    for i, tc in enumerate(tool_calls):
        if not isinstance(tc, dict):
            continue
        keys.extend(_tool_keys(tc, i))
    return keys


def _inject_reasoning_into_messages(messages: List[JsonDict]) -> bool:
    changed = False
    for idx, msg in enumerate(messages):
        if not isinstance(msg, dict):
            continue
        if msg.get("role") != "assistant":
            continue
        if not msg.get("tool_calls"):
            continue
        if "reasoning_content" in msg and not _is_blank_reasoning(msg.get("reasoning_content")):
            continue

        keys = _maybe_get_cache_keys(msg, idx)
        reasoning = _cache_get(keys)
        fallback = False
        if reasoning is None:
            reasoning = _FALLBACK_REASONING
            fallback = True

        msg["reasoning_content"] = reasoning
        changed = True
        _LOG.debug(
            "injected reasoning_content "
            f"msg_index={idx} tool_calls={len(msg.get('tool_calls') or [])} "
            f"cache_hit={not fallback} fallback={fallback} reasoning_len={len(reasoning)} "
            f"keys={keys if keys else None}"
        )
    return changed


def _get_request_body_bytes(request: httpx.Request) -> Optional[bytes]:
    try:
        content = request.content
    except Exception:
        return None
    if isinstance(content, bytes):
        return content
    if isinstance(content, bytearray):
        return bytes(content)
    if isinstance(content, str):
        return content.encode("utf-8", errors="replace")
    return None


def _httpx_mutator(request: httpx.Request) -> Optional[httpx.Response]:
    if request.method.upper() != "POST":
        return None

    try:
        path = getattr(request.url, "path", "") or ""
    except Exception:
        path = ""
    if not _suffix_matches(path):
        return None

    body = _get_request_body_bytes(request)
    if body is None:
        return None
    try:
        obj = json.loads(body) if body else {}
    except Exception:
        return None
    if not isinstance(obj, dict):
        return None

    model = obj.get("model")
    if not _model_matches(model):
        return None

    messages = obj.get("messages")
    if not isinstance(messages, list):
        return None

    if not _inject_reasoning_into_messages(messages):
        return None

    obj["messages"] = messages
    new_body = json.dumps(obj, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    request.headers["content-type"] = "application/json"
    request.headers["content-length"] = str(len(new_body))

    if hasattr(request, "_content"):
        try:
            setattr(request, "_content", new_body)
        except Exception:
            pass
    if hasattr(request, "_stream"):
        try:
            request._stream = httpx.ByteStream(new_body)  # type: ignore[attr-defined]
        except Exception:
            pass

    return None


def _httpx_response_mutator_sync(request: httpx.Request, response: httpx.Response) -> Optional[httpx.Response]:
    try:
        if response.status_code != 200:
            return None
        ctype = (response.headers.get("content-type") or "").lower()
        if "application/json" not in ctype:
            return None
        obj = response.json()
        if not isinstance(obj, dict):
            return None

        model = obj.get("model")
        if not model:
            try:
                req_obj = json.loads(request.content) if request.content else {}
                model = req_obj.get("model")
            except Exception:
                model = None

        if not _model_matches(model):
            return None

        _store_from_openai_chat_completion_json(obj)
    except Exception as e:
        _LOG.debug(f"response mutator failed (sync) req={_req_summary(request)}: {type(e).__name__}: {e}")
    return None


def _httpx_response_mutator_async(request: httpx.Request, response: httpx.Response) -> Optional[httpx.Response]:
    return _httpx_response_mutator_sync(request, response)


class _SSEToolCallParser:
    def __init__(self) -> None:
        self._buffer = bytearray()
        self._current_reasoning: Dict[int, str] = {}
        self._pending_keys: Dict[int, List[str]] = {}

    def feed_bytes(self, data: bytes) -> None:
        self._buffer.extend(data)
        while True:
            idx = self._buffer.find(b"\n\n")
            if idx == -1:
                break
            chunk = bytes(self._buffer[:idx])
            del self._buffer[: idx + 2]
            self._process_chunk(chunk)

    def finish(self) -> None:
        if self._buffer:
            self._process_chunk(bytes(self._buffer))
            self._buffer.clear()

    def _process_chunk(self, chunk: bytes) -> None:
        text = chunk.decode("utf-8", errors="replace")
        for line in text.splitlines():
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                return
            try:
                obj = json.loads(payload)
            except Exception:
                continue
            self._process_event(obj)

    def _process_event(self, obj: JsonDict) -> None:
        choices = obj.get("choices")
        if not isinstance(choices, list):
            return
        for choice in choices:
            if not isinstance(choice, dict):
                continue
            choice_idx = choice.get("index")
            idx = choice_idx if isinstance(choice_idx, int) else 0
            delta = choice.get("delta")
            if not isinstance(delta, dict):
                continue
            reasoning = delta.get("reasoning_content")
            if isinstance(reasoning, str) and reasoning.strip() != "":
                self._current_reasoning[idx] = self._merge_reasoning(idx, reasoning)
                self._flush_pending(idx)
            tool_calls = delta.get("tool_calls")
            if not isinstance(tool_calls, list) or not tool_calls:
                continue
            keys: List[str] = []
            for tc_idx, tc in enumerate(tool_calls):
                if not isinstance(tc, dict):
                    continue
                keys.extend(_tool_keys(tc, tc_idx))
            if not keys:
                continue
            current = self._current_reasoning.get(idx)
            if current:
                _cache_put(keys, current)
            else:
                pending = self._pending_keys.setdefault(idx, [])
                pending.extend(keys)

    def _merge_reasoning(self, idx: int, delta: str) -> str:
        current = self._current_reasoning.get(idx, "")
        if not current:
            return delta
        if delta.startswith(current):
            return delta
        return current + delta

    def _flush_pending(self, idx: int) -> None:
        current = self._current_reasoning.get(idx)
        if not current:
            return
        keys = self._pending_keys.pop(idx, None)
        if keys:
            _cache_put(keys, current)


class _TeeAsyncStream(getattr(httpx, "AsyncByteStream", object)):
    def __init__(self, stream: Any, parser: _SSEToolCallParser, log) -> None:
        self._stream = stream
        self._parser = parser
        self._log = log
        self._closed = False

    async def __aiter__(self):
        try:
            if hasattr(self._stream, "__aiter__"):
                async for chunk in self._stream:
                    data = self._coerce_bytes(chunk)
                    if data:
                        self._feed(data)
                    yield data
            elif hasattr(self._stream, "aiter_bytes"):
                async for chunk in self._stream.aiter_bytes():  # type: ignore[attr-defined]
                    data = self._coerce_bytes(chunk)
                    if data:
                        self._feed(data)
                    yield data
        finally:
            try:
                self._parser.finish()
            except Exception as e:
                self._log.debug(f"stream parser finish failed: {type(e).__name__}: {e}")

    async def aclose(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            if hasattr(self._stream, "aclose"):
                await self._stream.aclose()  # type: ignore[attr-defined]
            elif hasattr(self._stream, "close"):
                self._stream.close()  # type: ignore[attr-defined]
        except Exception:
            pass

    def _feed(self, data: bytes) -> None:
        try:
            self._parser.feed_bytes(data)
        except Exception as e:
            self._log.debug(f"stream parser failed: {type(e).__name__}: {e}")

    @staticmethod
    def _coerce_bytes(chunk: Any) -> bytes:
        if isinstance(chunk, bytes):
            return chunk
        if isinstance(chunk, bytearray):
            return bytes(chunk)
        if isinstance(chunk, memoryview):
            return chunk.tobytes()
        return b""


async def _capture_streaming_response(req: httpx.Request, resp: httpx.Response) -> Optional[httpx.Response]:
    try:
        if resp.status_code != 200:
            return None
        ctype = (resp.headers.get("content-type") or "").lower()
        if "text/event-stream" not in ctype:
            return None

        model = None
        try:
            req_obj = json.loads(req.content) if req.content else {}
            model = req_obj.get("model")
        except Exception:
            model = None

        if not _model_matches(model):
            return None

        if getattr(resp, "_litellm_ext_reasoning_replay_stream_wrapped", False):
            return resp

        stream = getattr(resp, "_stream", None)
        if stream is None:
            return resp

        parser = _SSEToolCallParser()
        resp._stream = _TeeAsyncStream(stream, parser, _LOG)  # type: ignore[attr-defined]
        setattr(resp, "_litellm_ext_reasoning_replay_stream_wrapped", True)
        return resp
    except Exception as e:
        _LOG.debug(f"stream parser failed req={_req_summary(req)}: {type(e).__name__}: {e}")
    return None


def _patch_proxy_app(module) -> None:
    if getattr(module, _PATCH_FLAG, False):
        return

    app = getattr(module, "app", None)
    if app is None or not hasattr(app, "add_middleware"):
        setattr(module, _PATCH_FLAG, True)
        return

    if getattr(app, "_litellm_ext_reasoning_replay", False):
        setattr(module, _PATCH_FLAG, True)
        return

    class _ReplayCaptureMiddleware:
        def __init__(self, app_):
            self.app = app_

        async def __call__(self, scope, receive, send):
            await self.app(scope, receive, send)

    app.add_middleware(_ReplayCaptureMiddleware)  # type: ignore
    setattr(app, "_litellm_ext_reasoning_replay", True)
    setattr(module, _PATCH_FLAG, True)


def install() -> None:
    global _FINDER_INSTALLED

    if not SETTINGS.is_enabled():
        _LOG.debug("disabled (reasoning_replay patch not enabled)")
        return

    install_httpx_patch()
    register_request_mutator("reasoning_replay", _httpx_mutator, priority=30)
    register_response_mutator("reasoning_replay_capture", _httpx_response_mutator_sync, priority=30)
    register_async_response_mutator("reasoning_replay_capture_async", _httpx_response_mutator_async, priority=30)
    register_async_response_mutator("reasoning_replay_stream_capture", _capture_streaming_response, priority=40)

    mod = sys.modules.get(TARGET_MOD)
    if mod is not None:
        _patch_proxy_app(mod)
        return

    if _FINDER_INSTALLED:
        return

    for f in sys.meta_path:
        if isinstance(f, _PatchFinder):
            _FINDER_INSTALLED = True
            return

    sys.meta_path.insert(0, _PatchFinder())


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
                _patch_proxy_app(module)

        spec.loader = _PatchedLoader()
        return spec
