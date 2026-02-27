from __future__ import annotations

from typing import Any, Dict, List

from ..adapters.anthropic_openai import extract_text_blocks
from .settings import (
    COMPACT_ROUTE_ENABLED,
    COMPACT_ROUTE_MODE,
    _COMPACT_EXPLICIT_PATTERNS,
    _COMPACT_PATTERNS,
)


def normalize_system(system: Any) -> str:
    if isinstance(system, str):
        return system
    if isinstance(system, list):
        return extract_text_blocks(system)
    return ""


def normalize_messages(messages: Any) -> str:
    if not isinstance(messages, list):
        return ""
    parts: List[str] = []
    for m in messages:
        if not isinstance(m, dict):
            parts.append(str(m))
            continue
        c = m.get("content", "")
        if isinstance(c, str):
            parts.append(c)
        elif isinstance(c, list):
            parts.append(extract_text_blocks(c))
        elif isinstance(c, dict):
            t = c.get("text")
            parts.append(t if isinstance(t, str) else str(c))
        else:
            parts.append(str(c))
    return "\n".join(x for x in parts if x)


def _normalize_message_content(message: Any) -> str:
    if not isinstance(message, dict):
        return str(message)
    content = message.get("content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return extract_text_blocks(content)
    if isinstance(content, dict):
        text = content.get("text")
        return text if isinstance(text, str) else str(content)
    return str(content)


def latest_user_message_text(messages: Any) -> str:
    if not isinstance(messages, list):
        return ""

    latest_nonempty = ""
    for message in reversed(messages):
        text = _normalize_message_content(message).strip()
        if not text:
            continue

        if not latest_nonempty:
            latest_nonempty = text

        if isinstance(message, dict) and str(message.get("role", "")).strip().lower() == "user":
            return text

    return latest_nonempty


def extract_all_text(payload: Dict[str, Any]) -> str:
    system_txt = normalize_system(payload.get("system"))
    latest_msg_txt = latest_user_message_text(payload.get("messages"))
    return "\n".join(x for x in (system_txt, latest_msg_txt) if x)


def _matches_patterns(text: str, patterns: List[Any]) -> bool:
    return any(p.search(text) for p in patterns)


def looks_like_compact(payload: Dict[str, Any]) -> bool:
    if not COMPACT_ROUTE_ENABLED:
        return False
    txt = extract_all_text(payload)
    if not txt:
        return False
    if COMPACT_ROUTE_MODE == "explicit":
        return _matches_patterns(txt, _COMPACT_EXPLICIT_PATTERNS)
    if COMPACT_ROUTE_MODE == "pattern":
        return _matches_patterns(txt, _COMPACT_PATTERNS)
    return False
