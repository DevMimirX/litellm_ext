import json

import httpx

import litellm_ext.extensions.httpx.reasoning_replay as rr


def _make_payload(tool_call_id: str = "call_123"):
    return {
        "model": "kimi-k2.5",
        "messages": [
            {"role": "user", "content": "hi"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": tool_call_id,
                        "type": "function",
                        "function": {"name": "echo", "arguments": "{}"},
                    }
                ],
                # reasoning_content intentionally missing
            },
            {"role": "tool", "tool_call_id": tool_call_id, "content": "hi"},
        ],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "echo",
                    "description": "Echo",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ],
        "tool_choice": "auto",
        "max_tokens": 10,
    }


def test_store_caches_both_id_and_fallback_keys():
    rr._CACHE.clear()

    obj = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "",
                    "reasoning_content": "R",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "echo", "arguments": "{}"},
                        }
                    ],
                }
            }
        ]
    }

    n = rr._store_from_openai_chat_completion_json(obj)
    assert n == 1

    # id key
    assert rr._cache_get(["call_1"]) == "R"
    # fallback key
    assert rr._cache_get(["echo:0"]) == "R"


def test_httpx_mutator_injects_using_fallback_when_id_key_missing():
    rr._CACHE.clear()
    # only populate fallback (simulates layers stripping tool-call id internally)
    rr._cache_put(["echo:0"], "R2")

    payload = _make_payload(tool_call_id="call_999")
    req = httpx.Request(
        "POST",
        "https://api.moonshot.cn/v1/chat/completions",
        json=payload,
    )

    rr._httpx_mutator(req)

    body = req.read()
    obj = json.loads(body)
    assert obj["messages"][1]["reasoning_content"] == "R2"


def test_httpx_mutator_injects_fallback_on_cache_miss():
    rr._CACHE.clear()

    payload = _make_payload(tool_call_id="call_miss")
    req = httpx.Request(
        "POST",
        "https://api.moonshot.cn/v1/chat/completions",
        json=payload,
    )

    rr._httpx_mutator(req)

    obj = json.loads(req.read())
    assert "reasoning_content" in obj["messages"][1]
    assert obj["messages"][1]["reasoning_content"] == rr._FALLBACK_REASONING


def test_store_skips_blank_reasoning():
    rr._CACHE.clear()

    obj = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "",
                    "reasoning_content": "",
                    "tool_calls": [
                        {
                            "id": "call_blank",
                            "type": "function",
                            "function": {"name": "echo", "arguments": "{}"},
                        }
                    ],
                }
            }
        ]
    }

    n = rr._store_from_openai_chat_completion_json(obj)
    assert n == 0
    assert rr._cache_get(["call_blank", "echo:0"]) is None


def test_httpx_response_mutator_uses_request_model_when_response_missing():
    rr._CACHE.clear()

    req = httpx.Request(
        "POST",
        "https://api.moonshot.cn/v1/chat/completions",
        json={"model": "kimi-k2.5"},
    )
    resp = httpx.Response(
        status_code=200,
        json={
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "",
                        "reasoning_content": "R3",
                        "tool_calls": [
                            {
                                "id": "call_req_model",
                                "type": "function",
                                "function": {"name": "echo", "arguments": "{}"},
                            }
                        ],
                    }
                }
            ]
        },
        headers={"content-type": "application/json"},
        request=req,
    )

    rr._httpx_response_mutator_sync(req, resp)
    assert rr._cache_get(["call_req_model"]) == "R3"


def test_sse_parser_caches_tool_calls_with_reasoning():
    rr._CACHE.clear()

    parser = rr._SSEToolCallParser()
    sse = (
        "event: message\n"
        "data: {\"choices\":[{\"delta\":{\"reasoning_content\":\"R4\",\"tool_calls\":[{\"index\":0,\"id\":\"call_sse\",\"type\":\"function\",\"function\":{\"name\":\"echo\"}}]}}]}\n"
        "\n"
        "data: [DONE]\n\n"
    ).encode("utf-8")

    parser.feed_bytes(sse)
    parser.finish()

    assert rr._cache_get(["call_sse"]) == "R4"


def test_sse_parser_accumulates_reasoning_deltas():
    rr._CACHE.clear()

    parser = rr._SSEToolCallParser()
    sse = (
        "data: {\"choices\":[{\"delta\":{\"reasoning_content\":\"R\"}}]}\n\n"
        "data: {\"choices\":[{\"delta\":{\"reasoning_content\":\"2\"}}]}\n\n"
        "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call_acc\",\"type\":\"function\",\"function\":{\"name\":\"echo\"}}]}}]}\n\n"
        "data: [DONE]\n\n"
    ).encode("utf-8")

    parser.feed_bytes(sse)
    parser.finish()

    assert rr._cache_get(["call_acc"]) == "R2"


def test_sse_parser_buffers_tool_calls_until_reasoning():
    rr._CACHE.clear()

    parser = rr._SSEToolCallParser()
    sse = (
        "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call_late\",\"type\":\"function\",\"function\":{\"name\":\"echo\"}}]}}]}\n\n"
        "data: {\"choices\":[{\"delta\":{\"reasoning_content\":\"R5\"}}]}\n\n"
        "data: [DONE]\n\n"
    ).encode("utf-8")

    parser.feed_bytes(sse)
    parser.finish()

    assert rr._cache_get(["call_late"]) == "R5"


def test_httpx_mutator_does_not_consume_streaming_request_body():
    class CountingStream(httpx.SyncByteStream):
        def __init__(self) -> None:
            self.iter_calls = 0

        def __iter__(self):
            self.iter_calls += 1
            yield b'{"model":"kimi-k2.5","messages":[]}'

    stream = CountingStream()
    req = httpx.Request(
        "POST",
        "https://api.moonshot.cn/v1/chat/completions",
        stream=stream,
        headers={"content-type": "application/json"},
    )

    rr._httpx_mutator(req)

    assert stream.iter_calls == 0
