from __future__ import annotations

"""Rewrite Anthropic-style usage in SSE streams and JSON responses."""

import json
import math
import sys
from fnmatch import fnmatch
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ...core.config import get_bool, get_list
from ...core.patch import PatchSettings
from ...policy import autocompact_multiplier_for_model, estimate_input_tokens_best_effort, normalize_model
from .proxy_patch_registry import register as register_proxy_patch, install as install_proxy_registry
from .utils import MAX_SSE_BUFFER, make_replay_receive, parse_json, read_body_with_limit, suffix_matches

TARGET_MOD = "litellm.proxy.proxy_server"

SETTINGS = PatchSettings(
    "stream_usage_rewrite",
    enabled_default=True,
    debug_default=False,
    enabled_envs=("LITELLM_EXT_STREAM_USAGE_REWRITE",),
    debug_envs=("LITELLM_EXT_STREAM_USAGE_REWRITE_DEBUG",),
)
_LOG = SETTINGS.logger("litellm_ext.stream_usage_rewrite")

_PATCH_FLAG = "_litellm_ext_stream_usage_rewrite_applied"
_MODEL_PATTERNS = get_list("extensions", "stream_usage_rewrite", "model_patterns", default=["glm-*", "deepseek-*"])
_RESPECT_AUTOCOMPACT = get_bool("extensions", "stream_usage_rewrite", "respect_autocompact_tuning", default=True)
_PATH_SUFFIXES = tuple(
    get_list(
        "extensions",
        "stream_usage_rewrite",
        "path_suffixes",
        default=["/v1/messages", "/anthropic/v1/messages"],
    )
)


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


def _safe_int(x: Any) -> int:
    try:
        return int(x)
    except Exception:
        return 0


def _total_tokens(usage: Dict[str, Any]) -> int:
    return (
        _safe_int(usage.get("input_tokens"))
        + _safe_int(usage.get("output_tokens"))
        + _safe_int(usage.get("cache_read_input_tokens"))
        + _safe_int(usage.get("cache_creation_input_tokens"))
    )


def _parse_frame(frame: str) -> Tuple[Optional[str], Optional[Dict[str, Any]], List[str]]:
    lines = frame.replace("\r\n", "\n").split("\n")
    event_line = None
    data_lines: List[str] = []
    for line in lines:
        if line.startswith("event:"):
            event_line = line
        elif line.startswith("data:"):
            data_lines.append(line.split(":", 1)[1].lstrip())
    data = "\n".join(data_lines).strip()
    if not data:
        return event_line, None, data_lines
    try:
        obj = json.loads(data)
    except Exception:
        return event_line, None, data_lines
    return event_line, obj if isinstance(obj, dict) else None, data_lines


def _build_frame(event_line: Optional[str], obj: Dict[str, Any], *, trailing_newline: bool) -> str:
    payload = json.dumps(obj, ensure_ascii=False, separators=(",", ":"))
    lines = []
    if event_line:
        lines.append(event_line)
    lines.append(f"data: {payload}")
    out = "\n".join(lines)
    if trailing_newline:
        out += "\n"
    return out


def _patch_message_start_frame(
    frame: str,
    *,
    missing_input_tokens: Optional[int],
    multiplier: float,
) -> Tuple[bool, str]:
    trailing = frame.endswith("\n")
    event_line, obj, _ = _parse_frame(frame)
    if not isinstance(obj, dict):
        return False, frame

    evt = obj.get("type")
    if evt != "message_start":
        return False, frame

    msg = obj.get("message")
    if not isinstance(msg, dict):
        return False, frame

    usage = msg.get("usage")
    if not isinstance(usage, dict):
        return False, frame

    input_tokens = _safe_int(usage.get("input_tokens"))
    cache_read = _safe_int(usage.get("cache_read_input_tokens"))
    cache_creation = _safe_int(usage.get("cache_creation_input_tokens"))
    effective = input_tokens + cache_read + cache_creation

    patched = False

    if effective <= 0 and missing_input_tokens is not None:
        usage["input_tokens"] = int(missing_input_tokens)
        usage.setdefault("cache_read_input_tokens", 0)
        usage.setdefault("cache_creation_input_tokens", 0)
        patched = True

    if multiplier != 1.0:
        input_tokens = _safe_int(usage.get("input_tokens"))
        if input_tokens > 0:
            usage["input_tokens"] = int(math.ceil(input_tokens * multiplier))
            patched = True
        if cache_read > 0:
            usage["cache_read_input_tokens"] = int(math.ceil(cache_read * multiplier))
            patched = True
        if cache_creation > 0:
            usage["cache_creation_input_tokens"] = int(math.ceil(cache_creation * multiplier))
            patched = True

    if "total_tokens" in usage and patched:
        usage["total_tokens"] = _total_tokens(usage)

    if not patched:
        return False, frame

    return True, _build_frame(event_line, obj, trailing_newline=trailing)


def _patch_message_delta_frame(
    frame: str,
    *,
    missing_input_tokens: Optional[int],
    multiplier: float,
    baseline_input: Optional[int],
    baseline_cache_read: Optional[int],
    baseline_cache_creation: Optional[int],
) -> Tuple[bool, str, Optional[int], Optional[int], Optional[int]]:
    trailing = frame.endswith("\n")
    event_line, obj, _ = _parse_frame(frame)
    if not isinstance(obj, dict):
        return False, frame, baseline_input, baseline_cache_read, baseline_cache_creation

    evt = obj.get("type")
    if evt != "message_delta":
        return False, frame, baseline_input, baseline_cache_read, baseline_cache_creation

    usage = obj.get("usage")
    if not isinstance(usage, dict):
        return False, frame, baseline_input, baseline_cache_read, baseline_cache_creation

    input_tokens = _safe_int(usage.get("input_tokens"))
    cache_read = _safe_int(usage.get("cache_read_input_tokens"))
    cache_creation = _safe_int(usage.get("cache_creation_input_tokens"))

    patched = False
    fallback_input = baseline_input if baseline_input is not None else missing_input_tokens

    injected_missing = False

    if input_tokens > 0:
        if baseline_input is not None and input_tokens >= baseline_input:
            scaled_input = input_tokens
        else:
            scaled_input = int(math.ceil(input_tokens * multiplier)) if multiplier != 1.0 else input_tokens
        if baseline_input is not None:
            scaled_input = max(scaled_input, baseline_input)
        if scaled_input != input_tokens:
            usage["input_tokens"] = scaled_input
            patched = True
        baseline_input = baseline_input or scaled_input
    else:
        if fallback_input is not None:
            if input_tokens != fallback_input:
                usage["input_tokens"] = int(fallback_input)
                patched = True
            baseline_input = baseline_input or int(fallback_input)
            injected_missing = True

    if injected_missing:
        usage.setdefault("cache_read_input_tokens", 0)
        usage.setdefault("cache_creation_input_tokens", 0)

    if cache_read > 0:
        if baseline_cache_read is not None and cache_read >= baseline_cache_read:
            scaled_cache = cache_read
        else:
            scaled_cache = int(math.ceil(cache_read * multiplier)) if multiplier != 1.0 else cache_read
        if baseline_cache_read is not None:
            scaled_cache = max(scaled_cache, baseline_cache_read)
        if scaled_cache != cache_read:
            usage["cache_read_input_tokens"] = scaled_cache
            patched = True
        baseline_cache_read = baseline_cache_read or scaled_cache
    elif baseline_cache_read is not None:
        usage["cache_read_input_tokens"] = baseline_cache_read
        patched = True

    if cache_creation > 0:
        if baseline_cache_creation is not None and cache_creation >= baseline_cache_creation:
            scaled_cache = cache_creation
        else:
            scaled_cache = int(math.ceil(cache_creation * multiplier)) if multiplier != 1.0 else cache_creation
        if baseline_cache_creation is not None:
            scaled_cache = max(scaled_cache, baseline_cache_creation)
        if scaled_cache != cache_creation:
            usage["cache_creation_input_tokens"] = scaled_cache
            patched = True
        baseline_cache_creation = baseline_cache_creation or scaled_cache
    elif baseline_cache_creation is not None:
        usage["cache_creation_input_tokens"] = baseline_cache_creation
        patched = True

    if "total_tokens" in usage and patched:
        usage["total_tokens"] = _total_tokens(usage)

    if not patched:
        return False, frame, baseline_input, baseline_cache_read, baseline_cache_creation

    return True, _build_frame(event_line, obj, trailing_newline=trailing), baseline_input, baseline_cache_read, baseline_cache_creation


class _SSERewriter:
    def __init__(self, *, missing_input_tokens: Optional[int], multiplier: float) -> None:
        self._buffer = ""
        self._missing_input_tokens = missing_input_tokens
        self._multiplier = multiplier
        self._seen_message_start = False
        self._baseline_input: Optional[int] = None
        self._baseline_cache_read: Optional[int] = None
        self._baseline_cache_creation: Optional[int] = None

    def feed(self, data: bytes | str) -> List[bytes]:
        if isinstance(data, str):
            text = data
        else:
            text = data.decode("utf-8", errors="replace")
        text = text.replace("\r\n", "\n")
        self._buffer += text
        if len(self._buffer) > MAX_SSE_BUFFER:
            self._buffer = self._buffer[-MAX_SSE_BUFFER:]
        out: List[bytes] = []
        while "\n\n" in self._buffer:
            frame, self._buffer = self._buffer.split("\n\n", 1)
            out.append(self._process_frame(frame).encode("utf-8"))
        return out

    def flush(self) -> List[bytes]:
        if not self._buffer:
            return []
        frame = self._buffer
        self._buffer = ""
        return [self._process_frame(frame).encode("utf-8")]

    def _process_frame(self, frame: str) -> str:
        event_line, obj, _ = _parse_frame(frame)
        if not isinstance(obj, dict):
            return frame + "\n\n"
        evt = obj.get("type") or (event_line.split(":", 1)[1].strip() if event_line else None)

        if evt == "message_start":
            if not self._seen_message_start:
                patched, new_frame = _patch_message_start_frame(
                    frame,
                    missing_input_tokens=self._missing_input_tokens,
                    multiplier=self._multiplier,
                )
                frame = new_frame
                _, obj2, _ = _parse_frame(frame)
                if isinstance(obj2, dict):
                    usage = (obj2.get("message") or {}).get("usage")
                    if isinstance(usage, dict):
                        self._baseline_input = _safe_int(usage.get("input_tokens")) or self._baseline_input
                        self._baseline_cache_read = _safe_int(usage.get("cache_read_input_tokens")) or self._baseline_cache_read
                        self._baseline_cache_creation = _safe_int(usage.get("cache_creation_input_tokens")) or self._baseline_cache_creation
                self._seen_message_start = True
            return frame + "\n\n"

        if evt == "message_delta":
            patched, new_frame, bi, bcr, bcc = _patch_message_delta_frame(
                frame,
                missing_input_tokens=self._missing_input_tokens,
                multiplier=self._multiplier,
                baseline_input=self._baseline_input,
                baseline_cache_read=self._baseline_cache_read,
                baseline_cache_creation=self._baseline_cache_creation,
            )
            self._baseline_input = bi
            self._baseline_cache_read = bcr
            self._baseline_cache_creation = bcc
            frame = new_frame
            return frame + "\n\n"

        return frame + "\n\n"


# -----------------------------
# ASGI middleware
# -----------------------------

def _patch_proxy_app(module) -> None:
    if getattr(module, _PATCH_FLAG, False):
        return

    app = getattr(module, "app", None)
    if app is None or not hasattr(app, "add_middleware"):
        _LOG.debug(f"proxy_server imported but no app found; skipping module={TARGET_MOD}")
        setattr(module, _PATCH_FLAG, True)
        return

    if getattr(app, "_litellm_ext_stream_usage_rewrite", False):
        setattr(module, _PATCH_FLAG, True)
        return

    class _StreamUsageRewriteMiddleware:
        def __init__(self, app_):
            self.app = app_

        async def __call__(self, scope, receive, send):
            if scope.get("type") != "http":
                return await self.app(scope, receive, send)

            path = (scope.get("path") or "").rstrip("/")
            if not suffix_matches(path, _PATH_SUFFIXES):
                return await self.app(scope, receive, send)

            # Read request body (then replay to downstream)
            body_msgs, body, truncated = await read_body_with_limit(receive)
            replay_receive = make_replay_receive(body_msgs, receive)
            if truncated:
                _LOG.debug(f"skipping stream_usage_rewrite: request body too large path={path!r}")
                return await self.app(scope, replay_receive, send)

            model = None
            payload = None
            payload = parse_json(body)
            if isinstance(payload, dict):
                model = payload.get("model")
            _LOG.debug(f"request path={path!r} model={model!r} suffix_match=True")

            if not _model_matches(model):
                _LOG.debug(f"skipping stream_usage_rewrite: model {model!r} does not match patterns")
                return await self.app(scope, replay_receive, send)

            mult = autocompact_multiplier_for_model(model) if _RESPECT_AUTOCOMPACT else 1.0
            missing_input_tokens = None
            _LOG.debug(
                f"stream_usage_rewrite enabled model={model!r} multiplier={mult:g} respect_autocompact={_RESPECT_AUTOCOMPACT}"
            )
            if payload is not None:
                try:
                    missing_input_tokens = estimate_input_tokens_best_effort(str(model or ""), payload)
                except Exception:
                    missing_input_tokens = None

            rewriter = _SSERewriter(missing_input_tokens=missing_input_tokens, multiplier=mult)
            headers: Dict[str, str] = {}
            resp_bytes = 0
            json_buffer = bytearray()

            response_started = False
            response_complete = False
            send_failed = False

            async def _send_safe(msg: Dict[str, Any]):
                nonlocal send_failed
                if send_failed:
                    return
                try:
                    await send(msg)
                except Exception:
                    send_failed = True

            async def send_wrapper(message: Dict[str, Any]):
                nonlocal resp_bytes, response_started, response_complete, send_failed
                if send_failed or response_complete:
                    return
                final_body = False
                if message.get("type") == "http.response.start":
                    response_started = True
                    for k, v in message.get("headers", []) or []:
                        try:
                            headers[k.decode("utf-8").lower()] = v.decode("utf-8")
                        except Exception:
                            pass
                    ctype_header = headers.get("content-type", "")
                    if "application/json" in ctype_header or "text/event-stream" in ctype_header:
                        try:
                            filtered = [
                                (k, v)
                                for (k, v) in (message.get("headers", []) or [])
                                if k.lower() != b"content-length"
                            ]
                            message = dict(message)
                            message["headers"] = filtered
                        except Exception:
                            pass
                    try:
                        await send(message)
                    except Exception:
                        send_failed = True
                    return

                if message.get("type") == "http.response.body":
                    ctype = headers.get("content-type", "")
                    body_chunk = message.get("body", b"") or b""
                    if isinstance(body_chunk, str):
                        body_chunk = body_chunk.encode("utf-8")
                    elif isinstance(body_chunk, memoryview):
                        body_chunk = body_chunk.tobytes()
                    elif isinstance(body_chunk, bytearray):
                        body_chunk = bytes(body_chunk)
                    more_body = bool(message.get("more_body", False))
                    final_body = not more_body
                    resp_bytes += len(body_chunk)
                    if _LOG.enabled() and not more_body:
                        _LOG.debug(
                            f"response done ctype={ctype!r} total_len={resp_bytes} model={model!r}"
                        )

                    if "text/event-stream" in ctype:
                        out_chunks = rewriter.feed(body_chunk)
                        if not more_body:
                            tail = rewriter.flush()
                            final_chunks = out_chunks + tail
                            if not final_chunks:
                                await _send_safe({"type": "http.response.body", "body": b"", "more_body": False})
                                if not send_failed:
                                    response_complete = True
                                return
                            for i, chunk in enumerate(final_chunks):
                                await _send_safe(
                                    {
                                        "type": "http.response.body",
                                        "body": chunk,
                                        "more_body": i < len(final_chunks) - 1,
                                    }
                                )
                            if not send_failed:
                                response_complete = True
                            return

                        for chunk in out_chunks:
                            await _send_safe({"type": "http.response.body", "body": chunk, "more_body": True})
                        return

                    if "application/json" in ctype:
                        json_buffer.extend(body_chunk)
                        if more_body:
                            return
                        try:
                            obj = json.loads(json_buffer.decode("utf-8", errors="replace"))
                            if isinstance(obj, dict):
                                usage = obj.get("usage")
                                if isinstance(usage, dict):
                                    patched, patched_frame = _patch_message_start_frame(
                                        "data: " + json.dumps({"type": "message_start", "message": {"usage": usage}}, separators=(",", ":")),
                                        missing_input_tokens=missing_input_tokens,
                                        multiplier=mult,
                                    )
                                    if patched:
                                        _, patched_obj, _ = _parse_frame(patched_frame)
                                        if isinstance(patched_obj, dict):
                                            patched_usage = (patched_obj.get("message") or {}).get("usage")
                                            if isinstance(patched_usage, dict):
                                                obj["usage"] = patched_usage
                            body = json.dumps(obj, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
                            await _send_safe({"type": "http.response.body", "body": body, "more_body": False})
                            if not send_failed:
                                response_complete = True
                            return
                        except Exception:
                            await _send_safe(
                                {"type": "http.response.body", "body": bytes(json_buffer), "more_body": False}
                            )
                            if not send_failed:
                                response_complete = True
                            return

                await _send_safe(message)
                if final_body and not send_failed:
                    response_complete = True

            await self.app(scope, replay_receive, send_wrapper)
            if response_started and not response_complete and not send_failed:
                try:
                    await send({"type": "http.response.body", "body": b"", "more_body": False})
                except Exception:
                    pass
            return

    app.add_middleware(_StreamUsageRewriteMiddleware)  # type: ignore
    setattr(app, "_litellm_ext_stream_usage_rewrite", True)
    setattr(module, _PATCH_FLAG, True)
    _LOG.debug("installed stream_usage_rewrite middleware")


def install() -> None:
    if not SETTINGS.is_enabled():
        _LOG.debug("disabled (stream_usage_rewrite patch not enabled)")
        return

    register_proxy_patch("stream_usage_rewrite", _patch_proxy_app, order=40)
    install_proxy_registry()
