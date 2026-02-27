"""Helpers for Anthropic ↔ OpenAI request/response conversion.

These utilities are best-effort and designed for patch-internal normalization
and tooling. They do not guarantee full semantic fidelity for every provider.
"""

from __future__ import annotations

import json
import time
from typing import Any, Dict, List, Optional, Tuple

JsonDict = Dict[str, Any]

SCHEMA_OPENAI = "openai"
SCHEMA_ANTHROPIC = "anthropic"
SCHEMA_UNKNOWN = "unknown"


def map_anthropic_stop_reason_to_openai(stop_reason: Optional[str], *, strict: bool = True) -> str:
    if not stop_reason:
        return "stop"
    if not strict:
        return stop_reason
    mapping = {
        "stop_sequence": "stop",
        "end_turn": "stop",
        "pause_turn": "stop",
        "tool_use": "tool_calls",
        "max_tokens": "length",
    }
    return mapping.get(stop_reason, stop_reason)


def map_openai_finish_reason_to_anthropic(finish_reason: Optional[str], *, strict: bool = True) -> str:
    if not finish_reason:
        return "end_turn"
    if not strict:
        return finish_reason
    mapping = {
        "stop": "end_turn",
        "length": "max_tokens",
        "tool_calls": "tool_use",
        "function_call": "tool_use",
        "content_filter": "stop_sequence",
    }
    return mapping.get(finish_reason, "end_turn")


def detect_schema(payload: JsonDict, *, path: Optional[str] = None) -> str:
    """Best-effort schema detection using path and payload shape."""
    if path:
        p = path.rstrip("/")
        if p.endswith("/v1/messages") or p.endswith("/messages") or p.endswith("/anthropic/v1/messages"):
            return SCHEMA_ANTHROPIC
        if p.endswith("/v1/chat/completions") or p.endswith("/chat/completions"):
            return SCHEMA_OPENAI

    # Heuristics based on top-level fields
    if isinstance(payload.get("system"), (str, list)) or "stop_sequences" in payload:
        return SCHEMA_ANTHROPIC

    tools = payload.get("tools")
    if isinstance(tools, list):
        for t in tools:
            if isinstance(t, dict) and "input_schema" in t:
                return SCHEMA_ANTHROPIC
            if isinstance(t, dict) and "function" in t:
                return SCHEMA_OPENAI

    msgs = payload.get("messages")
    if isinstance(msgs, list):
        for m in msgs:
            if isinstance(m, dict):
                if "tool_calls" in m or "function_call" in m:
                    return SCHEMA_OPENAI
                c = m.get("content")
                if isinstance(c, list):
                    for b in c:
                        if isinstance(b, dict) and "type" in b:
                            return SCHEMA_ANTHROPIC
                if isinstance(c, str):
                    return SCHEMA_OPENAI
    return SCHEMA_UNKNOWN


def _json_text(val: Any) -> str:
    try:
        return json.dumps(val, ensure_ascii=False, default=str)
    except Exception:
        return str(val)


def extract_text_blocks(blocks: Any) -> str:
    """Extract text from content blocks, dumping non-text blocks as JSON."""
    if not isinstance(blocks, list):
        return ""
    parts: List[str] = []
    for b in blocks:
        if isinstance(b, dict) and isinstance(b.get("text"), str):
            parts.append(b["text"])
        elif isinstance(b, str):
            parts.append(b)
        else:
            parts.append(_json_text(b))
    return "\n".join(parts)




def clean_schema(schema: Any) -> Any:
    """Remove unsupported JSON schema fields (e.g., format: uri)."""
    if not isinstance(schema, dict):
        return schema
    obj = dict(schema)
    if obj.get("format") == "uri":
        obj.pop("format", None)
    props = obj.get("properties")
    if isinstance(props, dict):
        obj["properties"] = {k: clean_schema(v) for k, v in props.items()}
    items = obj.get("items")
    if items is not None:
        obj["items"] = clean_schema(items)
    return obj


def _parse_tool_arguments(raw: Any) -> Any:
    if raw is None:
        return {}
    if isinstance(raw, (dict, list)):
        return raw
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except Exception:
            return raw
    return raw


def _stringify_tool_arguments(raw: Any) -> str:
    if raw is None:
        return "{}"
    if isinstance(raw, str):
        return raw
    try:
        return json.dumps(raw, ensure_ascii=False, default=str)
    except Exception:
        return str(raw)


def _parse_data_url(url: Any) -> Optional[Tuple[str, str]]:
    if not isinstance(url, str) or not url.startswith("data:"):
        return None
    try:
        header, data = url.split(",", 1)
    except ValueError:
        return None
    meta = header[5:]
    parts = [p for p in meta.split(";") if p]
    if not parts:
        media_type = "application/octet-stream"
    else:
        media_type = parts[0] if "/" in parts[0] else "application/octet-stream"
    if "base64" not in parts:
        return None
    return media_type, data


def _image_url_to_anthropic_block(image_url: Any) -> Optional[JsonDict]:
    url = None
    if isinstance(image_url, dict):
        url = image_url.get("url")
    elif isinstance(image_url, str):
        url = image_url
    parsed = _parse_data_url(url)
    if not parsed:
        return None
    media_type, data = parsed
    if not data:
        return None
    return {
        "type": "image",
        "source": {"type": "base64", "media_type": media_type, "data": data},
    }




def openai_messages_to_canonical(messages: Any) -> List[JsonDict]:
    out: List[JsonDict] = []
    if not isinstance(messages, list):
        return out

    for m in messages:
        if not isinstance(m, dict):
            out.append({"role": "user", "content": [{"type": "text", "text": _json_text(m)}]})
            continue

        role = str(m.get("role") or "user")
        blocks: List[JsonDict] = []

        if role == "tool":
            content = m.get("content", "")
            blocks.append(
                {
                    "type": "tool_result",
                    "tool_use_id": m.get("tool_call_id"),
                    "content": content if content is not None else "",
                }
            )
            out.append({"role": role, "content": blocks})
            continue

        content = m.get("content")
        if isinstance(content, str):
            blocks.append({"type": "text", "text": content})
        elif isinstance(content, list):
            for b in content:
                if isinstance(b, dict) and isinstance(b.get("type"), str):
                    btype = b.get("type")
                    if btype == "text" and isinstance(b.get("text"), str):
                        blocks.append({"type": "text", "text": b["text"]})
                    elif btype == "image_url":
                        blocks.append({"type": "image_url", "image_url": b.get("image_url")})
                    else:
                        blocks.append({"type": "text", "text": _json_text(b)})
                else:
                    blocks.append({"type": "text", "text": _json_text(b)})
        elif content is not None:
            blocks.append({"type": "text", "text": _json_text(content)})

        tool_calls = m.get("tool_calls")
        if role == "assistant" and isinstance(tool_calls, list):
            for tc in tool_calls:
                if not isinstance(tc, dict):
                    continue
                fn = tc.get("function") if isinstance(tc.get("function"), dict) else {}
                blocks.append(
                    {
                        "type": "tool_use",
                        "id": tc.get("id"),
                        "name": fn.get("name"),
                        "input": _parse_tool_arguments(fn.get("arguments")),
                    }
                )

        out.append({"role": role, "content": blocks})

    return out


def anthropic_messages_to_canonical(messages: Any) -> List[JsonDict]:
    out: List[JsonDict] = []
    if not isinstance(messages, list):
        return out

    for m in messages:
        if not isinstance(m, dict):
            out.append({"role": "user", "content": [{"type": "text", "text": _json_text(m)}]})
            continue
        role = str(m.get("role") or "user")
        content = m.get("content")
        blocks: List[JsonDict] = []
        if isinstance(content, str):
            blocks.append({"type": "text", "text": content})
        elif isinstance(content, list):
            for b in content:
                if isinstance(b, dict) and isinstance(b.get("type"), str):
                    btype = b.get("type")
                    if btype == "text" and isinstance(b.get("text"), str):
                        blocks.append({"type": "text", "text": b["text"]})
                    elif btype == "thinking":
                        # skip thinking blocks
                        continue
                    else:
                        blocks.append(b)
                elif isinstance(b, str):
                    blocks.append({"type": "text", "text": b})
                else:
                    blocks.append({"type": "text", "text": _json_text(b)})
        elif content is not None:
            blocks.append({"type": "text", "text": _json_text(content)})

        out.append({"role": role, "content": blocks})

    return out


def canonical_to_openai_messages(messages: List[JsonDict]) -> List[JsonDict]:
    out: List[JsonDict] = []
    for m in messages:
        role = str(m.get("role") or "user")
        blocks = m.get("content") if isinstance(m.get("content"), list) else []

        text_blocks: List[str] = []
        image_blocks: List[JsonDict] = []
        tool_use_blocks: List[JsonDict] = []
        tool_result_blocks: List[JsonDict] = []

        for b in blocks:
            if not isinstance(b, dict):
                text_blocks.append(_json_text(b))
                continue
            btype = b.get("type")
            if btype == "tool_use":
                tool_use_blocks.append(b)
            elif btype == "tool_result":
                tool_result_blocks.append(b)
            elif btype == "text" and isinstance(b.get("text"), str):
                text_blocks.append(b["text"])
            elif btype == "image":
                source = b.get("source") if isinstance(b.get("source"), dict) else {}
                media_type = source.get("media_type") if isinstance(source.get("media_type"), str) else "image/png"
                data = source.get("data") if isinstance(source.get("data"), str) else ""
                if data:
                    image_blocks.append(
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:{media_type};base64,{data}"},
                        }
                    )
            elif btype == "image_url":
                image_blocks.append({"type": "image_url", "image_url": b.get("image_url")})
            else:
                text_blocks.append(_json_text(b))

        if image_blocks:
            content_parts: List[JsonDict] = []
            if text_blocks:
                content_parts.append({"type": "text", "text": "\n".join(t for t in text_blocks if t)})
            content_parts.extend(image_blocks)
            base: JsonDict = {"role": role, "content": content_parts}
        else:
            joined = "\n".join(t for t in text_blocks if t)
            base = {"role": role, "content": joined if joined else None}

        if role == "assistant" and tool_use_blocks:
            tool_calls: List[JsonDict] = []
            for idx, b in enumerate(tool_use_blocks):
                tool_calls.append(
                    {
                        "id": b.get("id") or f"call_{idx}",
                        "type": "function",
                        "function": {
                            "name": b.get("name"),
                            "arguments": _stringify_tool_arguments(b.get("input")),
                        },
                    }
                )
            base["tool_calls"] = tool_calls
            if base.get("content") is None:
                base["content"] = ""

        if role == "tool":
            base["tool_call_id"] = m.get("tool_call_id")

        if base.get("content") is not None or base.get("tool_calls"):
            out.append(base)

        for b in tool_result_blocks:
            out.append(
                {
                    "role": "tool",
                    "tool_call_id": b.get("tool_use_id") or b.get("tool_call_id"),
                    "content": extract_text_blocks(b.get("content"))
                    if isinstance(b.get("content"), list)
                    else (b.get("content") or ""),
                }
            )

    return out


def canonical_to_anthropic_messages(messages: List[JsonDict]) -> List[JsonDict]:
    out: List[JsonDict] = []
    for m in messages:
        role = str(m.get("role") or "user")
        if role == "tool":
            role = "user"
        blocks = m.get("content") if isinstance(m.get("content"), list) else []
        new_blocks: List[JsonDict] = []
        for b in blocks:
            if not isinstance(b, dict):
                new_blocks.append({"type": "text", "text": _json_text(b)})
                continue
            btype = b.get("type")
            if btype == "tool_use":
                new_blocks.append(
                    {
                        "type": "tool_use",
                        "id": b.get("id"),
                        "name": b.get("name"),
                        "input": b.get("input") if b.get("input") is not None else {},
                    }
                )
            elif btype == "tool_result":
                new_blocks.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": b.get("tool_use_id") or b.get("tool_call_id"),
                        "content": b.get("content", ""),
                    }
                )
            elif btype == "image":
                source = b.get("source") if isinstance(b.get("source"), dict) else None
                if isinstance(source, dict) and isinstance(source.get("data"), str):
                    new_blocks.append({"type": "image", "source": source})
                else:
                    new_blocks.append({"type": "text", "text": _json_text(b)})
            elif btype == "image_url":
                block = _image_url_to_anthropic_block(b.get("image_url") if isinstance(b, dict) else None)
                if block is not None:
                    new_blocks.append(block)
                else:
                    new_blocks.append({"type": "text", "text": _json_text(b)})
            elif btype == "text" and isinstance(b.get("text"), str):
                new_blocks.append({"type": "text", "text": b["text"]})
            else:
                new_blocks.append({"type": "text", "text": _json_text(b)})
        out.append({"role": role, "content": new_blocks if new_blocks else ""})
    return out


def canonicalize_messages(
    payload_or_messages: Any, *, schema: Optional[str] = None, path: Optional[str] = None
) -> List[JsonDict]:
    payload = payload_or_messages if isinstance(payload_or_messages, dict) else None
    messages = payload_or_messages.get("messages") if isinstance(payload_or_messages, dict) else payload_or_messages
    if schema is None:
        schema = detect_schema(payload or {}, path=path) if payload is not None else SCHEMA_UNKNOWN
    if schema == SCHEMA_OPENAI:
        return openai_messages_to_canonical(messages)
    if schema == SCHEMA_ANTHROPIC:
        return anthropic_messages_to_canonical(messages)
    return anthropic_messages_to_canonical(messages)


def anthropic_to_openai_messages(payload: JsonDict) -> JsonDict:
    """Convert Anthropic /v1/messages payload into OpenAI /v1/chat/completions format (best-effort)."""
    out: JsonDict = {}

    # system -> system messages
    messages: List[JsonDict] = []
    system = payload.get("system")
    if isinstance(system, str):
        messages.append({"role": "system", "content": system})
    elif isinstance(system, list):
        for b in system:
            if isinstance(b, dict) and isinstance(b.get("text"), str):
                messages.append({"role": "system", "content": b["text"]})

    # messages
    messages.extend(canonical_to_openai_messages(anthropic_messages_to_canonical(payload.get("messages"))))
    out["messages"] = messages

    # model
    if payload.get("model") is not None:
        out["model"] = payload.get("model")

    # params
    for k in ("max_tokens", "temperature", "top_p", "stream"):
        if payload.get(k) is not None:
            out[k] = payload.get(k)
    if payload.get("stop_sequences") is not None:
        out["stop"] = payload.get("stop_sequences")
    metadata = payload.get("metadata")
    if isinstance(metadata, dict) and metadata.get("user_id"):
        out["user"] = metadata.get("user_id")

    tools = payload.get("tools")
    if isinstance(tools, list):
        openai_tools: List[JsonDict] = []
        for t in tools:
            if not isinstance(t, dict):
                continue
            if t.get("type") == "BatchTool":
                continue
            name = t.get("name")
            if not name and isinstance(t.get("function"), dict):
                name = t.get("function", {}).get("name")
            if not name:
                continue
            entry = {
                "type": "function",
                "function": {
                    "name": name,
                    "description": t.get("description") or "",
                    "parameters": clean_schema(t.get("input_schema") or {}),
                },
            }
            if t.get("cache_control") is not None:
                entry["cache_control"] = t.get("cache_control")
            openai_tools.append(entry)
        if openai_tools:
            out["tools"] = openai_tools

    if payload.get("tool_choice") is not None:
        out["tool_choice"] = payload.get("tool_choice")

    return out


def openai_to_anthropic_messages(payload: JsonDict) -> JsonDict:
    """Convert OpenAI /v1/chat/completions payload into Anthropic /v1/messages format (best-effort)."""
    out: JsonDict = {}

    messages = payload.get("messages") if isinstance(payload.get("messages"), list) else []
    system_parts: List[str] = []
    filtered_messages: List[JsonDict] = []
    for m in messages:
        if isinstance(m, dict) and m.get("role") == "system":
            if isinstance(m.get("content"), str):
                system_parts.append(m["content"])
            elif isinstance(m.get("content"), list):
                system_parts.append(extract_text_blocks(m["content"]))
            else:
                system_parts.append(_json_text(m.get("content")))
        else:
            filtered_messages.append(m)

    if system_parts:
        out["system"] = "\n\n".join(p for p in system_parts if p)

    out["messages"] = canonical_to_anthropic_messages(openai_messages_to_canonical(filtered_messages))

    # model + params
    if payload.get("model") is not None:
        out["model"] = payload.get("model")
    for k in ("max_tokens", "temperature", "top_p", "stream"):
        if payload.get(k) is not None:
            out[k] = payload.get(k)
    if out.get("max_tokens") is None:
        for k in ("max_completion_tokens", "max_output_tokens"):
            if payload.get(k) is not None:
                out["max_tokens"] = payload.get(k)
                break
    if payload.get("stop") is not None:
        out["stop_sequences"] = payload.get("stop")
    if payload.get("user") is not None:
        out.setdefault("metadata", {})["user_id"] = payload.get("user")
    if payload.get("thinking") is not None:
        out["thinking"] = payload.get("thinking")

    tools = payload.get("tools")
    if isinstance(tools, list):
        anthropic_tools: List[JsonDict] = []
        for t in tools:
            if not isinstance(t, dict):
                continue
            if t.get("function") is not None:
                fn = t.get("function") if isinstance(t.get("function"), dict) else {}
                name = fn.get("name")
                if not name:
                    continue
                entry = {
                    "name": name,
                    "description": fn.get("description") or "",
                    "input_schema": clean_schema(fn.get("parameters") or {}),
                }
                # advanced tool fields (if present)
                for extra_key in ("defer_loading", "allowed_callers", "input_examples"):
                    if extra_key in fn:
                        entry[extra_key] = fn.get(extra_key)
                if t.get("cache_control") is not None:
                    entry["cache_control"] = t.get("cache_control")
                anthropic_tools.append(entry)
            elif isinstance(t.get("type"), str):
                tool_type = t.get("type")
                tool_opts = t.get(tool_type) if isinstance(t.get(tool_type), dict) else {}
                entry = {
                    "name": tool_type,
                    "type": tool_opts.get("name") if isinstance(tool_opts, dict) else None,
                }
                if isinstance(tool_opts, dict):
                    cleaned_opts = dict(tool_opts)
                    cleaned_opts.pop("name", None)
                    entry.update(cleaned_opts)
                if isinstance(t.get("cache_control"), dict):
                    entry["cache_control"] = t.get("cache_control")
                anthropic_tools.append(entry)
        if anthropic_tools:
            out["tools"] = anthropic_tools

    tool_choice = payload.get("tool_choice")
    if tool_choice is not None:
        if isinstance(tool_choice, str):
            if tool_choice == "required":
                out["tool_choice"] = {"type": "any"}
            elif tool_choice == "auto":
                out["tool_choice"] = {"type": "auto"}
            elif tool_choice == "none":
                out["tool_choice"] = {"type": "none"}
            else:
                out["tool_choice"] = tool_choice
        elif isinstance(tool_choice, dict):
            fn = tool_choice.get("function") if isinstance(tool_choice.get("function"), dict) else {}
            name = fn.get("name")
            if name:
                out["tool_choice"] = {"type": "tool", "name": name}
            elif isinstance(tool_choice.get("name"), str) and tool_choice.get("type") == "tool":
                out["tool_choice"] = {"type": "tool", "name": tool_choice.get("name")}
            else:
                out["tool_choice"] = tool_choice
        else:
            out["tool_choice"] = tool_choice

    return out


def openai_response_to_anthropic(payload: JsonDict, *, strict: bool = True) -> JsonDict:
    """Best-effort OpenAI response -> Anthropic response."""
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        return {}
    choice = choices[0] if isinstance(choices[0], dict) else {}
    message = choice.get("message") if isinstance(choice.get("message"), dict) else {}

    content: List[JsonDict] = []
    if not strict:
        reasoning_text = None
        if isinstance(message.get("reasoning"), str):
            reasoning_text = message.get("reasoning")
        elif isinstance(message.get("reasoning_content"), str):
            reasoning_text = message.get("reasoning_content")
        if reasoning_text:
            content.append({"type": "thinking", "thinking": reasoning_text})
    if isinstance(message.get("content"), str) and message.get("content"):
        content.append({"type": "text", "text": message.get("content")})

    tool_calls = message.get("tool_calls")
    if isinstance(tool_calls, list):
        for tc in tool_calls:
            if not isinstance(tc, dict):
                continue
            func = tc.get("function") if isinstance(tc.get("function"), dict) else {}
            args_str = func.get("arguments") if isinstance(func.get("arguments"), str) else "{}"
            try:
                args = json.loads(args_str)
            except Exception:
                args = {}
            content.append(
                {
                    "type": "tool_use",
                    "id": tc.get("id"),
                    "name": func.get("name"),
                    "input": args,
                }
            )

    finish_reason = choice.get("finish_reason")
    stop_reason = map_openai_finish_reason_to_anthropic(
        finish_reason if isinstance(finish_reason, str) else None,
        strict=strict,
    )

    usage = payload.get("usage") if isinstance(payload.get("usage"), dict) else {}
    input_tokens = int(usage.get("prompt_tokens") or 0)
    output_tokens = int(usage.get("completion_tokens") or 0)

    return {
        "id": payload.get("id") or "",
        "type": "message",
        "role": "assistant",
        "content": content,
        "model": payload.get("model") or "",
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
        },
    }


def anthropic_response_to_openai(payload: JsonDict, *, strict: bool = True) -> JsonDict:
    """Best-effort Anthropic response -> OpenAI response."""
    if not isinstance(payload, dict):
        return {}

    model = payload.get("model") or ""
    content_blocks = payload.get("content")
    canon = anthropic_messages_to_canonical([{"role": "assistant", "content": content_blocks}])
    openai_msgs = canonical_to_openai_messages(canon)
    assistant_msg = next((m for m in openai_msgs if isinstance(m, dict) and m.get("role") == "assistant"), None)
    if assistant_msg is None:
        assistant_msg = {"role": "assistant", "content": ""}
    if not strict:
        try:
            non_tool_blocks = [
                b for b in (content_blocks or []) if isinstance(b, dict) and b.get("type") != "tool_use"
            ]
        except Exception:
            non_tool_blocks = []
        assistant_msg = dict(assistant_msg)
        if non_tool_blocks:
            assistant_msg["content_blocks"] = non_tool_blocks
        if isinstance(content_blocks, list):
            thinking_parts = []
            for b in content_blocks:
                if not isinstance(b, dict):
                    continue
                if b.get("type") == "thinking":
                    text = b.get("thinking") or b.get("summary")
                    if isinstance(text, str) and text:
                        thinking_parts.append(text)
            if thinking_parts:
                assistant_msg["reasoning"] = "\n".join(thinking_parts)

    stop_reason = payload.get("stop_reason")
    finish_reason = map_anthropic_stop_reason_to_openai(
        stop_reason if isinstance(stop_reason, str) else None,
        strict=strict,
    )

    usage = payload.get("usage") if isinstance(payload.get("usage"), dict) else {}
    input_tokens = int(usage.get("input_tokens") or 0)
    output_tokens = int(usage.get("output_tokens") or 0)
    cache_read = int(usage.get("cache_read_input_tokens") or 0)
    cache_creation = int(usage.get("cache_creation_input_tokens") or 0)
    usage_out: JsonDict = {}
    if input_tokens or output_tokens or cache_read or cache_creation:
        usage_out = {
            "prompt_tokens": input_tokens,
            "completion_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens + cache_read + cache_creation,
            "prompt_tokens_details": {"cached_tokens": cache_read},
        }
        if cache_read or cache_creation:
            usage_out["cache_read_input_tokens"] = cache_read
            usage_out["cache_creation_input_tokens"] = cache_creation

    return {
        "id": payload.get("id") or "",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": assistant_msg,
                "finish_reason": finish_reason,
            }
        ],
        "usage": usage_out or None,
    }


