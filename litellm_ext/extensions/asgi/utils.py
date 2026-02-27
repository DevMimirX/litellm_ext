from __future__ import annotations

"""Shared ASGI middleware utilities."""

import json
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple


# Maximum request body size in bytes (100MB)
MAX_REQUEST_SIZE = 100 * 1024 * 1024

# Maximum SSE buffer size in bytes (10MB)
MAX_SSE_BUFFER = 10 * 1024 * 1024

ASGIReceive = Callable[[], Awaitable[Dict[str, Any]]]


def update_content_length(scope: Dict[str, Any], new_len: int) -> Dict[str, Any]:
    """Update content-length header in ASGI scope."""
    try:
        headers = list(scope.get("headers") or [])
    except Exception:
        return scope
    out: List[tuple[bytes, bytes]] = []
    found = False
    for k, v in headers:
        if k.lower() == b"content-length":
            out.append((k, str(new_len).encode("utf-8")))
            found = True
        else:
            out.append((k, v))
    if not found:
        out.append((b"content-length", str(new_len).encode("utf-8")))
    new_scope = dict(scope)
    new_scope["headers"] = out
    return new_scope


def parse_json(body: bytes, *, max_size: Optional[int] = None) -> Optional[Dict[str, Any]]:
    """Parse JSON bytes with optional size limit.

    Args:
        body: The bytes to parse
        max_size: Maximum allowed size in bytes (default: MAX_REQUEST_SIZE)

    Returns:
        Parsed JSON dict or None if invalid
    """
    max_size = MAX_REQUEST_SIZE if max_size is None else int(max_size)
    if not body:
        return None
    if len(body) > max_size:
        return None
    try:
        obj = json.loads(body.decode("utf-8", errors="replace"))
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def suffix_matches(path: str, suffixes: Tuple[str, ...]) -> bool:
    """Check if path ends with any of the suffixes."""
    p = (path or "").rstrip("/")
    return any(p.endswith(s.rstrip("/")) for s in suffixes)


async def read_body_with_limit(
    receive: ASGIReceive,
    *,
    max_size: Optional[int] = None,
) -> Tuple[List[Dict[str, Any]], bytes, bool]:
    """Read request body with size limit.

    Args:
        receive: ASGI receive function
        max_size: Maximum allowed size in bytes

    Returns:
        Tuple of (message list, body bytes, truncated)
    """
    max_size = MAX_REQUEST_SIZE if max_size is None else int(max_size)
    body_msgs: List[Dict[str, Any]] = []
    body_parts: List[bytes] = []
    total_size = 0
    truncated = False
    more = True

    while more:
        m = await receive()
        if m.get("type") == "http.request":
            if "more_body" not in m or m.get("more_body") is None:
                m["more_body"] = False
            if m.get("body") is None:
                m["body"] = b""

        body_msgs.append(m)

        if m.get("type") != "http.request":
            break

        chunk = m.get("body", b"") or b""
        next_size = total_size + len(chunk)
        if next_size > max_size:
            # Stop buffering body as soon as the request exceeds the limit.
            # The caller can replay buffered messages and delegate to the
            # original receive() to stream any remaining client body.
            truncated = True
            break

        body_parts.append(chunk)
        total_size = next_size
        more = bool(m.get("more_body", False))

    body = b"" if truncated else b"".join(body_parts)
    return body_msgs, body, truncated


def make_replay_receive(body_msgs: List[Dict[str, Any]], receive: ASGIReceive) -> ASGIReceive:
    """Build a replaying ASGI receive() wrapper with safe terminal behavior."""
    idx = 0
    terminal_sent = False
    has_request = False
    last_request_more_body = False
    for msg in reversed(body_msgs):
        if msg.get("type") == "http.request":
            has_request = True
            last_request_more_body = bool(msg.get("more_body", False))
            break
    needs_terminal = has_request and not last_request_more_body

    async def replay_receive() -> Dict[str, Any]:
        nonlocal idx, terminal_sent
        if idx < len(body_msgs):
            msg = body_msgs[idx]
            idx += 1
            return msg
        if needs_terminal and not terminal_sent:
            terminal_sent = True
            return {"type": "http.request", "body": b"", "more_body": False}
        return await receive()

    return replay_receive
