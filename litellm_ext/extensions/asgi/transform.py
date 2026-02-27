"""ASGI middleware: convert between Anthropic and OpenAI message schemas."""

from __future__ import annotations

import json
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ...adapters.anthropic_openai import (
    SCHEMA_ANTHROPIC,
    SCHEMA_OPENAI,
    anthropic_to_openai_messages,
    anthropic_response_to_openai,
    detect_schema,
    map_anthropic_stop_reason_to_openai,
    map_openai_finish_reason_to_anthropic,
    openai_to_anthropic_messages,
    openai_response_to_anthropic,
)
from ...core.config import get_list
from ...core.config import get_bool
from ...core.patch import PatchSettings
from .proxy_patch_registry import register as register_proxy_patch, install as install_proxy_registry
from .utils import (
    MAX_SSE_BUFFER,
    make_replay_receive,
    parse_json,
    read_body_with_limit,
    suffix_matches,
    update_content_length,
)

TARGET_MOD = "litellm.proxy.proxy_server"

SETTINGS = PatchSettings(
    "transform",
    enabled_default=True,
    debug_default=False,
    enabled_envs=("LITELLM_EXT_TRANSFORM",),
    debug_envs=("LITELLM_EXT_TRANSFORM_DEBUG",),
)
_LOG = SETTINGS.logger("litellm_ext.transform")

_PATCH_FLAG = "_litellm_ext_transform_applied"
_OPENAI_PATHS = tuple(
    get_list(
        "extensions",
        "transform",
        "openai_paths",
        default=["/v1/chat/completions", "/chat/completions"],
    )
)
_ANTHROPIC_PATHS = tuple(
    get_list(
        "extensions",
        "transform",
        "anthropic_paths",
        default=["/v1/messages", "/anthropic/v1/messages"],
    )
)

_STRICT_OPENAI = get_bool("extensions", "transform", "strict_openai_compliance", default=True)

JsonDict = Dict[str, Any]


def _parse_sse_frames(buf: str) -> Tuple[List[str], str]:
    frames: List[str] = []
    while "\n\n" in buf:
        frame, buf = buf.split("\n\n", 1)
        frames.append(frame)
    return frames, buf


def _extract_sse_data(frame: str) -> List[str]:
    data_lines: List[str] = []
    for line in frame.replace("\r\n", "\n").split("\n"):
        if line.startswith("data:"):
            data_lines.append(line.split(":", 1)[1].lstrip())
    return data_lines


def _sse_event(event_type: str, payload: JsonDict) -> bytes:
    data = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    return f"event: {event_type}\ndata: {data}\n\n".encode("utf-8")


class _OpenAIToAnthropicSSE:
    def __init__(self, *, strict: bool) -> None:
        self._buffer = ""
        self._message_id: Optional[str] = None
        self._model: Optional[str] = None
        self._content_index = 0
        self._current_block_type: Optional[str] = None
        self._tool_state: Dict[int, Dict[str, Any]] = {}
        self._sent_message_start = False
        self._strict = strict
        self._usage_prompt_tokens = 0
        self._usage_completion_tokens: Optional[int] = None
        self._pending_finish_reason: Optional[str] = None
        self._delta_sent = False

    def feed(self, data: bytes) -> List[bytes]:
        self._buffer += data.decode("utf-8", errors="replace")
        if len(self._buffer) > MAX_SSE_BUFFER:
            self._buffer = self._buffer[-MAX_SSE_BUFFER:]
        frames, self._buffer = _parse_sse_frames(self._buffer)
        out: List[bytes] = []
        for frame in frames:
            out.extend(self._process_frame(frame))
        return out

    def flush(self) -> List[bytes]:
        if not self._buffer:
            return []
        out = self._process_frame(self._buffer)
        self._buffer = ""
        return out

    def _ensure_message_start(self) -> List[bytes]:
        if self._sent_message_start:
            return []
        msg = {
            "type": "message_start",
            "message": {
                "id": self._message_id or "",
                "type": "message",
                "role": "assistant",
                "model": self._model or "",
                "usage": {"input_tokens": self._usage_prompt_tokens, "output_tokens": 0},
            },
        }
        self._sent_message_start = True
        return [_sse_event("message_start", msg)]

    def _close_block(self) -> List[bytes]:
        if self._current_block_type is None:
            return []
        evt = {"type": "content_block_stop", "index": self._content_index}
        self._current_block_type = None
        self._content_index += 1
        return [_sse_event("content_block_stop", evt)]

    def _open_block(self, block_type: str, payload: JsonDict) -> List[bytes]:
        self._current_block_type = block_type
        evt = {
            "type": "content_block_start",
            "index": self._content_index,
            "content_block": payload,
        }
        return [_sse_event("content_block_start", evt)]

    def _emit_message_delta(self) -> List[bytes]:
        if self._delta_sent:
            return []
        delta: JsonDict = {}
        if self._pending_finish_reason is not None:
            delta["stop_reason"] = map_openai_finish_reason_to_anthropic(
                self._pending_finish_reason,
                strict=self._strict,
            )
        usage: JsonDict = {}
        if self._usage_completion_tokens is not None:
            usage["output_tokens"] = self._usage_completion_tokens
        if not delta and not usage:
            return []
        payload: JsonDict = {
            "type": "message_delta",
            "delta": delta,
        }
        if usage:
            payload["usage"] = usage
        self._delta_sent = True
        return [_sse_event("message_delta", payload)]

    def _process_frame(self, frame: str) -> List[bytes]:
        out: List[bytes] = []
        for data in _extract_sse_data(frame):
            if not data:
                continue
            if data.strip() == "[DONE]":
                out.extend(self._close_block())
                out.extend(self._emit_message_delta())
                out.append(_sse_event("message_stop", {"type": "message_stop"}))
                continue

            try:
                chunk = json.loads(data)
            except Exception:
                continue

            if not isinstance(chunk, dict):
                continue

            if self._message_id is None and isinstance(chunk.get("id"), str):
                self._message_id = chunk.get("id")
            if self._model is None and isinstance(chunk.get("model"), str):
                self._model = chunk.get("model")

            usage = chunk.get("usage") if isinstance(chunk.get("usage"), dict) else {}
            if usage:
                prompt_tokens = usage.get("prompt_tokens")
                completion_tokens = usage.get("completion_tokens")
                if isinstance(prompt_tokens, int):
                    self._usage_prompt_tokens = prompt_tokens
                if isinstance(completion_tokens, int):
                    self._usage_completion_tokens = completion_tokens

            choices = chunk.get("choices")
            if not isinstance(choices, list) or not choices:
                continue
            choice = choices[0] if isinstance(choices[0], dict) else {}
            delta = choice.get("delta") if isinstance(choice.get("delta"), dict) else {}
            finish_reason = choice.get("finish_reason") if isinstance(choice.get("finish_reason"), str) else None

            out.extend(self._ensure_message_start())

            reasoning = delta.get("reasoning")
            if isinstance(reasoning, str) and reasoning:
                if self._current_block_type != "thinking":
                    out.extend(self._close_block())
                    out.extend(self._open_block("thinking", {"type": "thinking", "thinking": ""}))
                evt = {
                    "type": "content_block_delta",
                    "index": self._content_index,
                    "delta": {"type": "thinking_delta", "thinking": reasoning},
                }
                out.append(_sse_event("content_block_delta", evt))

            content = delta.get("content")
            if isinstance(content, str) and content:
                if self._current_block_type != "text":
                    out.extend(self._close_block())
                    out.extend(self._open_block("text", {"type": "text", "text": ""}))
                evt = {
                    "type": "content_block_delta",
                    "index": self._content_index,
                    "delta": {"type": "text_delta", "text": content},
                }
                out.append(_sse_event("content_block_delta", evt))

            tool_calls = delta.get("tool_calls")
            if not isinstance(tool_calls, list):
                fn_call = delta.get("function_call") if isinstance(delta.get("function_call"), dict) else None
                if isinstance(fn_call, dict):
                    tool_calls = [{"index": 0, "function": fn_call}]

            if isinstance(tool_calls, list):
                for tc in tool_calls:
                    if not isinstance(tc, dict):
                        continue
                    idx = int(tc.get("index") or 0)
                    state = self._tool_state.setdefault(idx, {"id": None, "name": None, "started": False})
                    if isinstance(tc.get("id"), str):
                        state["id"] = tc.get("id")
                    fn = tc.get("function") if isinstance(tc.get("function"), dict) else {}
                    if isinstance(fn.get("name"), str):
                        state["name"] = fn.get("name")

                    if not state["started"] and state.get("name") is not None:
                        out.extend(self._close_block())
                        out.extend(
                            self._open_block(
                                "tool_use",
                                {
                                    "type": "tool_use",
                                    "id": state.get("id") or f"tool_{idx}",
                                    "name": state.get("name") or "",
                                },
                            )
                        )
                        state["started"] = True

                    args = fn.get("arguments")
                    if isinstance(args, str) and args:
                        if not state.get("started"):
                            out.extend(self._close_block())
                            out.extend(
                                self._open_block(
                                    "tool_use",
                                    {
                                        "type": "tool_use",
                                        "id": state.get("id") or f"tool_{idx}",
                                        "name": state.get("name") or "",
                                    },
                                )
                            )
                            state["started"] = True
                        evt = {
                            "type": "content_block_delta",
                            "index": self._content_index,
                            "delta": {"type": "input_json_delta", "partial_json": args},
                        }
                        out.append(_sse_event("content_block_delta", evt))

            if finish_reason is not None:
                self._pending_finish_reason = finish_reason
                out.extend(self._close_block())
                out.extend(self._emit_message_delta())

        return out


class _AnthropicToOpenAISSE:
    def __init__(self, *, strict: bool) -> None:
        self._buffer = ""
        self._message_id: Optional[str] = None
        self._model: Optional[str] = None
        self._tool_index = -1
        self._usage_prompt_tokens = 0
        self._cache_read_tokens = 0
        self._cache_creation_tokens = 0
        self._strict = strict

    def feed(self, data: bytes) -> List[bytes]:
        self._buffer += data.decode("utf-8", errors="replace")
        if len(self._buffer) > MAX_SSE_BUFFER:
            self._buffer = self._buffer[-MAX_SSE_BUFFER:]
        frames, self._buffer = _parse_sse_frames(self._buffer)
        out: List[bytes] = []
        for frame in frames:
            out.extend(self._process_frame(frame))
        return out

    def flush(self) -> List[bytes]:
        if not self._buffer:
            return []
        out = self._process_frame(self._buffer)
        self._buffer = ""
        return out

    def _chunk(self, payload: JsonDict) -> bytes:
        return f"data: {json.dumps(payload, ensure_ascii=False, separators=(',', ':'))}\n\n".encode("utf-8")

    def _content_blocks(self, obj: JsonDict, delta: JsonDict) -> Optional[List[JsonDict]]:
        if self._strict:
            return None
        content_block_obj = {"index": obj.get("index"), "delta": dict(delta)}
        content_block_obj["delta"].pop("type", None)
        return [content_block_obj]

    def _process_frame(self, frame: str) -> List[bytes]:
        out: List[bytes] = []
        for data in _extract_sse_data(frame):
            if not data:
                continue
            if data.strip() == "[DONE]":
                out.append(b"data: [DONE]\n\n")
                continue
            try:
                obj = json.loads(data)
            except Exception:
                continue
            if not isinstance(obj, dict):
                continue

            evt = obj.get("type")
            if evt == "error":
                err = obj.get("error") if isinstance(obj.get("error"), dict) else {}
                out.append(
                    self._chunk(
                        {
                            "id": self._message_id or "",
                            "object": "chat.completion.chunk",
                            "created": int(time.time()),
                            "model": self._model or "",
                            "choices": [
                                {
                                    "delta": {"content": ""},
                                    "finish_reason": err.get("type") or "error",
                                }
                            ],
                        }
                    )
                )
                out.append(b"data: [DONE]\n\n")
                continue

            if evt == "message_start":
                msg = obj.get("message") if isinstance(obj.get("message"), dict) else {}
                if isinstance(msg.get("id"), str):
                    self._message_id = msg.get("id")
                if isinstance(msg.get("model"), str):
                    self._model = msg.get("model")
                usage = msg.get("usage") if isinstance(msg.get("usage"), dict) else {}
                self._usage_prompt_tokens = int(usage.get("input_tokens") or 0)
                self._cache_read_tokens = int(usage.get("cache_read_input_tokens") or 0)
                self._cache_creation_tokens = int(usage.get("cache_creation_input_tokens") or 0)
                out.append(
                    self._chunk(
                        {
                            "id": self._message_id or "",
                            "object": "chat.completion.chunk",
                            "created": int(time.time()),
                            "model": self._model or "",
                            "choices": [
                                {
                                    "delta": {"content": "", "role": "assistant"},
                                    "index": 0,
                                    "logprobs": None,
                                    "finish_reason": None,
                                }
                            ],
                        }
                    )
                )
                continue

            if evt == "message_delta":
                usage = obj.get("usage") if isinstance(obj.get("usage"), dict) else None
                completion_tokens = None
                total_tokens = None
                if isinstance(usage, dict):
                    completion_tokens = int(usage.get("output_tokens") or 0)
                    total_tokens = (
                        self._usage_prompt_tokens
                        + self._cache_read_tokens
                        + self._cache_creation_tokens
                        + completion_tokens
                    )
                finish_reason = map_anthropic_stop_reason_to_openai(
                    obj.get("delta", {}).get("stop_reason") if isinstance(obj.get("delta"), dict) else None,
                    strict=self._strict,
                )
                if finish_reason is None and not usage:
                    continue
                payload = {
                    "id": self._message_id or "",
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": self._model or "",
                    "choices": [
                        {
                            "index": 0,
                            "delta": {},
                            "finish_reason": finish_reason,
                        }
                    ],
                }
                if isinstance(usage, dict):
                    payload["usage"] = {
                        "prompt_tokens": self._usage_prompt_tokens,
                        "completion_tokens": completion_tokens or 0,
                        "total_tokens": total_tokens or 0,
                        "prompt_tokens_details": {
                            "cached_tokens": self._cache_read_tokens,
                        },
                        "cache_read_input_tokens": self._cache_read_tokens,
                        "cache_creation_input_tokens": self._cache_creation_tokens,
                    }
                out.append(self._chunk(payload))
                continue

            if evt == "content_block_start":
                block = obj.get("content_block") if isinstance(obj.get("content_block"), dict) else {}
                if block.get("type") == "tool_use":
                    self._tool_index += 1
                    out.append(
                        self._chunk(
                            {
                                "id": self._message_id or "",
                                "object": "chat.completion.chunk",
                                "created": int(time.time()),
                                "model": self._model or "",
                                "choices": [
                                    {
                                        "delta": {
                                            "tool_calls": [
                                                {
                                                    "index": self._tool_index,
                                                    "id": block.get("id"),
                                                    "type": "function",
                                                    "function": {"name": block.get("name"), "arguments": ""},
                                                }
                                            ]
                                        },
                                        "index": 0,
                                        "logprobs": None,
                                        "finish_reason": None,
                                    }
                                ],
                            }
                        )
                    )
                continue

            if evt == "content_block_delta":
                delta = obj.get("delta") if isinstance(obj.get("delta"), dict) else {}
                dtype = delta.get("type")
                if dtype == "input_json_delta":
                    payload = {
                        "id": self._message_id or "",
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "model": self._model or "",
                        "choices": [
                            {
                                "delta": {
                                    "tool_calls": [
                                        {
                                            "index": self._tool_index,
                                            "function": {"arguments": delta.get("partial_json")},
                                        }
                                    ]
                                },
                                "index": 0,
                                "logprobs": None,
                                "finish_reason": None,
                            }
                        ],
                    }
                    extra = self._content_blocks(obj, delta)
                    if extra is not None:
                        payload["choices"][0]["delta"]["content_blocks"] = extra
                    out.append(self._chunk(payload))
                elif dtype == "text_delta":
                    payload = {
                        "id": self._message_id or "",
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "model": self._model or "",
                        "choices": [
                            {
                                "delta": {"content": delta.get("text")},
                                "index": 0,
                                "logprobs": None,
                                "finish_reason": None,
                            }
                        ],
                    }
                    extra = self._content_blocks(obj, delta)
                    if extra is not None:
                        payload["choices"][0]["delta"]["content_blocks"] = extra
                    out.append(self._chunk(payload))
                elif dtype == "thinking_delta":
                    if not self._strict and isinstance(delta.get("thinking"), str):
                        payload = {
                            "id": self._message_id or "",
                            "object": "chat.completion.chunk",
                            "created": int(time.time()),
                            "model": self._model or "",
                            "choices": [
                                {
                                    "delta": {"reasoning": delta.get("thinking")},
                                    "index": 0,
                                    "logprobs": None,
                                    "finish_reason": None,
                                }
                            ],
                        }
                        extra = self._content_blocks(obj, delta)
                        if extra is not None:
                            payload["choices"][0]["delta"]["content_blocks"] = extra
                        out.append(self._chunk(payload))
                elif dtype in ("signature_delta", "citations_delta"):
                    extra = self._content_blocks(obj, delta)
                    if extra is not None:
                        payload = {
                            "id": self._message_id or "",
                            "object": "chat.completion.chunk",
                            "created": int(time.time()),
                            "model": self._model or "",
                            "choices": [
                                {
                                    "delta": {"content": "", "content_blocks": extra},
                                    "index": 0,
                                    "logprobs": None,
                                    "finish_reason": None,
                                }
                            ],
                        }
                        out.append(self._chunk(payload))
                continue

            if evt == "message_stop":
                out.append(b"data: [DONE]\n\n")

        return out
def _convert_payload(path: str, payload: JsonDict) -> Optional[JsonDict]:
    schema = detect_schema(payload, path=path)

    if suffix_matches(path, _OPENAI_PATHS) and schema == SCHEMA_ANTHROPIC:
        out = anthropic_to_openai_messages(payload)
        _LOG.debug(f"converted Anthropic->OpenAI path={path!r} model={payload.get('model')!r}")
        return out

    if suffix_matches(path, _ANTHROPIC_PATHS) and schema == SCHEMA_OPENAI:
        out = openai_to_anthropic_messages(payload)
        _LOG.debug(f"converted OpenAI->Anthropic path={path!r} model={payload.get('model')!r}")
        return out

    return None


def _response_mode(path: str, payload: JsonDict) -> Optional[str]:
    schema = detect_schema(payload, path=path)
    if suffix_matches(path, _OPENAI_PATHS) and schema == SCHEMA_ANTHROPIC:
        return "openai_to_anthropic"
    if suffix_matches(path, _ANTHROPIC_PATHS) and schema == SCHEMA_OPENAI:
        return "anthropic_to_openai"
    return None


def _patch_proxy_app(module) -> None:
    if getattr(module, _PATCH_FLAG, False):
        return

    app = getattr(module, "app", None)
    if app is None or not hasattr(app, "add_middleware"):
        _LOG.debug(f"proxy_server imported but no app found; skipping module={TARGET_MOD}")
        setattr(module, _PATCH_FLAG, True)
        return

    if getattr(app, "_litellm_ext_transform_mw", False):
        setattr(module, _PATCH_FLAG, True)
        return

    class _TransformASGIMiddleware:
        def __init__(self, app_):
            self.app = app_

        async def __call__(self, scope, receive, send):
            path = (scope.get("path") or "").rstrip("/")
            try:
                if scope.get("type") != "http":
                    return await self.app(scope, receive, send)

                if (scope.get("method") or "").upper() != "POST":
                    return await self.app(scope, receive, send)

                if not (suffix_matches(path, _OPENAI_PATHS) or suffix_matches(path, _ANTHROPIC_PATHS)):
                    return await self.app(scope, receive, send)

                body_msgs, body, truncated = await read_body_with_limit(receive)
                if truncated:
                    _LOG.debug(f"skipping transform: request body too large path={path!r}")
                    replay_receive = make_replay_receive(body_msgs, receive)
                    return await self.app(scope, replay_receive, send)

                payload = parse_json(body) or {}
                if not payload:
                    return await self._call_downstream(scope, body_msgs, receive, send)

                resp_mode = _response_mode(path, payload)
                new_payload = _convert_payload(path, payload)
                if new_payload is None:
                    return await self._call_downstream(scope, body_msgs, receive, send, response_mode=resp_mode)

                new_body = json.dumps(new_payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
                scope2 = update_content_length(scope, len(new_body))

                return await self._call_downstream(
                    scope2,
                    [{"type": "http.request", "body": new_body, "more_body": False}],
                    receive,
                    send,
                    response_mode=resp_mode,
                )
            except Exception as e:
                _LOG.debug(f"transform middleware failed path={path!r}: {type(e).__name__}: {e}")
                return await self.app(scope, receive, send)

        async def _call_downstream(self, scope, body_msgs, receive_fn, send_fn, response_mode: Optional[str] = None):
            headers: Dict[str, str] = {}
            json_buffer = bytearray()
            sse_rewriter: Optional[object] = None
            if response_mode == "openai_to_anthropic":
                sse_rewriter = _OpenAIToAnthropicSSE(strict=_STRICT_OPENAI)
            elif response_mode == "anthropic_to_openai":
                sse_rewriter = _AnthropicToOpenAISSE(strict=_STRICT_OPENAI)
            replay_receive = make_replay_receive(body_msgs, receive_fn)

            if response_mode is None:
                return await self.app(scope, replay_receive, send_fn)

            async def send_wrapper(message: Dict[str, Any]):
                if message.get("type") == "http.response.start":
                    for k, v in message.get("headers", []) or []:
                        try:
                            headers[k.decode("utf-8").lower()] = v.decode("utf-8")
                        except Exception:
                            pass
                    if response_mode is not None:
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
                    return await send_fn(message)

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

                    if "application/json" in ctype:
                        json_buffer.extend(body_chunk)
                        if more_body:
                            return
                        try:
                            obj = json.loads(json_buffer.decode("utf-8", errors="replace"))
                            if isinstance(obj, dict):
                                if response_mode == "openai_to_anthropic":
                                    obj = openai_response_to_anthropic(obj, strict=_STRICT_OPENAI)
                                elif response_mode == "anthropic_to_openai":
                                    obj = anthropic_response_to_openai(obj, strict=_STRICT_OPENAI)
                            body = json.dumps(obj, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
                            await send_fn({"type": "http.response.body", "body": body, "more_body": False})
                            return
                        except Exception:
                            await send_fn(
                                {"type": "http.response.body", "body": bytes(json_buffer), "more_body": False}
                            )
                            return
                    elif "text/event-stream" in ctype and sse_rewriter is not None:
                        out_chunks = sse_rewriter.feed(body_chunk)  # type: ignore[attr-defined]
                        if not more_body:
                            tail = sse_rewriter.flush()  # type: ignore[attr-defined]
                            final_chunks = out_chunks + tail
                            if not final_chunks:
                                await send_fn({"type": "http.response.body", "body": b"", "more_body": False})
                                return
                            for i, chunk in enumerate(final_chunks):
                                await send_fn(
                                    {"type": "http.response.body", "body": chunk, "more_body": i < len(final_chunks) - 1}
                                )
                            return

                        for chunk in out_chunks:
                            await send_fn({"type": "http.response.body", "body": chunk, "more_body": True})
                        return

                await send_fn(message)

            return await self.app(scope, replay_receive, send_wrapper)

    app.add_middleware(_TransformASGIMiddleware)  # type: ignore
    setattr(app, "_litellm_ext_transform_mw", True)
    setattr(module, _PATCH_FLAG, True)
    _LOG.debug(
        "installed transform middleware "
        f"openai_paths={list(_OPENAI_PATHS)} anthropic_paths={list(_ANTHROPIC_PATHS)}"
    )


def install() -> None:
    if not SETTINGS.is_enabled():
        _LOG.debug("disabled (transform patch not enabled)")
        return

    register_proxy_patch("transform", _patch_proxy_app, order=60)
    install_proxy_registry()
