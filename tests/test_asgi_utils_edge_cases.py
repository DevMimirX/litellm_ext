from __future__ import annotations

import asyncio

from litellm_ext.extensions.asgi.utils import make_replay_receive, parse_json, read_body_with_limit


def test_parse_json_respects_max_size() -> None:
    body = b'{"a":1}'
    assert parse_json(body, max_size=6) is None
    assert parse_json(body, max_size=32) == {"a": 1}


def test_read_body_with_limit_truncates_without_consuming_entire_stream() -> None:
    queued = [
        {"type": "http.request", "body": b"aaaa", "more_body": True},
        {"type": "http.request", "body": b"bbbb", "more_body": True},
        {"type": "http.request", "body": b"cccc", "more_body": False},
    ]
    idx = {"n": 0}

    async def receive():
        i = idx["n"]
        idx["n"] += 1
        if i < len(queued):
            return queued[i]
        return {"type": "http.disconnect"}

    body_msgs, body, truncated = asyncio.run(read_body_with_limit(receive, max_size=6))

    assert truncated is True
    assert body == b""
    assert idx["n"] == 2
    assert len(body_msgs) == 2

    replay_receive = make_replay_receive(body_msgs, receive)
    first = asyncio.run(replay_receive())
    second = asyncio.run(replay_receive())
    delegated = asyncio.run(replay_receive())
    terminal = asyncio.run(replay_receive())

    assert first["body"] == b"aaaa"
    assert second["body"] == b"bbbb"
    assert delegated["body"] == b"cccc"
    assert terminal == {"type": "http.disconnect"}


def test_make_replay_receive_does_not_inject_request_after_disconnect_only() -> None:
    queued = [{"type": "http.disconnect"}]
    idx = {"n": 0}

    async def receive():
        i = idx["n"]
        idx["n"] += 1
        if i < len(queued):
            return queued[i]
        return {"type": "http.disconnect"}

    replay_receive = make_replay_receive([{"type": "http.disconnect"}], receive)
    first = asyncio.run(replay_receive())
    second = asyncio.run(replay_receive())

    assert first == {"type": "http.disconnect"}
    assert second == {"type": "http.disconnect"}
