import json

from litellm_ext.extensions.asgi.transform import _AnthropicToOpenAISSE, _OpenAIToAnthropicSSE


def _collect_text(chunks):
    return b"".join(chunks).decode("utf-8", errors="replace")


def test_openai_sse_to_anthropic_text_stream():
    sse = _OpenAIToAnthropicSSE(strict=True)
    chunk = {
        "id": "chatcmpl-1",
        "model": "gpt-test",
        "choices": [{"delta": {"content": "Hi"}}],
    }
    data = f"data: {json.dumps(chunk)}\n\n"
    out = _collect_text(sse.feed(data.encode("utf-8")) + sse.feed(b"data: [DONE]\n\n") + sse.flush())

    assert "event: message_start" in out
    assert "\"type\":\"content_block_start\"" in out
    assert "\"type\":\"text_delta\"" in out
    assert "event: message_stop" in out


def test_openai_sse_to_anthropic_tool_calls():
    sse = _OpenAIToAnthropicSSE(strict=True)
    first = {
        "id": "chatcmpl-2",
        "model": "gpt-test",
        "choices": [
            {
                "delta": {
                    "tool_calls": [
                        {"index": 0, "id": "call_1", "function": {"name": "echo"}}
                    ]
                }
            }
        ],
    }
    second = {
        "id": "chatcmpl-2",
        "model": "gpt-test",
        "choices": [
            {
                "delta": {
                    "tool_calls": [
                        {"index": 0, "function": {"arguments": "{\"text\":\"hi\"}"}}
                    ]
                }
            }
        ],
    }
    out = _collect_text(
        sse.feed(f"data: {json.dumps(first)}\n\n".encode("utf-8"))
        + sse.feed(f"data: {json.dumps(second)}\n\n".encode("utf-8"))
        + sse.feed(b"data: [DONE]\n\n")
        + sse.flush()
    )

    assert "\"type\":\"tool_use\"" in out
    assert "\"type\":\"input_json_delta\"" in out
    assert "event: message_stop" in out




def test_openai_sse_to_anthropic_finish_reason_usage():
    sse = _OpenAIToAnthropicSSE(strict=True)
    first = {
        "id": "chatcmpl-3",
        "model": "gpt-test",
        "choices": [{"delta": {"content": "Hello"}}],
    }
    last = {
        "id": "chatcmpl-3",
        "model": "gpt-test",
        "choices": [{"delta": {}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 4, "completion_tokens": 2},
    }
    out = _collect_text(
        sse.feed(f"data: {json.dumps(first)}\n\n".encode("utf-8"))
        + sse.feed(f"data: {json.dumps(last)}\n\n".encode("utf-8"))
        + sse.feed(b"data: [DONE]\n\n")
        + sse.flush()
    )

    assert "\"type\":\"message_delta\"" in out
    assert "\"stop_reason\":\"end_turn\"" in out
    assert "\"output_tokens\":2" in out




def test_anthropic_sse_to_openai_message_delta_without_usage():
    sse = _AnthropicToOpenAISSE(strict=True)
    start = {
        "type": "message_start",
        "message": {
            "id": "msg_3",
            "type": "message",
            "role": "assistant",
            "model": "claude-test",
            "usage": {"input_tokens": 3, "output_tokens": 0},
        },
    }
    msg_delta = {"type": "message_delta", "delta": {"stop_reason": "end_turn"}}
    done = {"type": "message_stop"}

    def frame(obj):
        return f"event: {obj['type']}\ndata: {json.dumps(obj)}\n\n".encode("utf-8")

    out = _collect_text(
        sse.feed(frame(start)) + sse.feed(frame(msg_delta)) + sse.feed(frame(done)) + sse.flush()
    )

    assert "\"finish_reason\":\"stop\"" in out


def test_anthropic_sse_to_openai_thinking_delta_non_strict():
    sse = _AnthropicToOpenAISSE(strict=False)
    start = {
        "type": "message_start",
        "message": {
            "id": "msg_4",
            "type": "message",
            "role": "assistant",
            "model": "claude-test",
            "usage": {"input_tokens": 1, "output_tokens": 0},
        },
    }
    thinking = {
        "type": "content_block_delta",
        "index": 0,
        "delta": {"type": "thinking_delta", "thinking": "step"},
    }
    done = {"type": "message_stop"}

    def frame(obj):
        return f"event: {obj['type']}\ndata: {json.dumps(obj)}\n\n".encode("utf-8")

    out = _collect_text(sse.feed(frame(start)) + sse.feed(frame(thinking)) + sse.feed(frame(done)) + sse.flush())

    assert "\"reasoning\":\"step\"" in out
    assert "\"content_blocks\"" in out


def test_anthropic_sse_to_openai_text_and_tool_calls():
    sse = _AnthropicToOpenAISSE(strict=True)
    start = {
        "type": "message_start",
        "message": {
            "id": "msg_1",
            "type": "message",
            "role": "assistant",
            "model": "claude-test",
            "usage": {"input_tokens": 5, "output_tokens": 0},
        },
    }
    tool_start = {
        "type": "content_block_start",
        "index": 0,
        "content_block": {"type": "tool_use", "id": "call_1", "name": "echo"},
    }
    tool_delta = {
        "type": "content_block_delta",
        "index": 0,
        "delta": {"type": "input_json_delta", "partial_json": "{\"text\":\"hi\"}"},
    }
    text_delta = {
        "type": "content_block_delta",
        "index": 1,
        "delta": {"type": "text_delta", "text": "Hello"},
    }
    msg_delta = {
        "type": "message_delta",
        "delta": {"stop_reason": "end_turn"},
        "usage": {"output_tokens": 2},
    }
    done = {"type": "message_stop"}

    def frame(obj):
        return f"event: {obj['type']}\ndata: {json.dumps(obj)}\n\n".encode("utf-8")

    out = _collect_text(
        sse.feed(frame(start))
        + sse.feed(frame(tool_start))
        + sse.feed(frame(tool_delta))
        + sse.feed(frame(text_delta))
        + sse.feed(frame(msg_delta))
        + sse.feed(frame(done))
        + sse.flush()
    )

    assert "\"tool_calls\"" in out
    assert "\"content\":\"Hello\"" in out
    assert "data: [DONE]" in out


def test_anthropic_sse_to_openai_non_strict_includes_content_blocks():
    sse = _AnthropicToOpenAISSE(strict=False)
    start = {
        "type": "message_start",
        "message": {
            "id": "msg_2",
            "type": "message",
            "role": "assistant",
            "model": "claude-test",
            "usage": {"input_tokens": 1, "output_tokens": 0},
        },
    }
    text_delta = {
        "type": "content_block_delta",
        "index": 0,
        "delta": {"type": "text_delta", "text": "Hi"},
    }
    done = {"type": "message_stop"}

    def frame(obj):
        return f"event: {obj['type']}\ndata: {json.dumps(obj)}\n\n".encode("utf-8")

    out = _collect_text(
        sse.feed(frame(start)) + sse.feed(frame(text_delta)) + sse.feed(frame(done)) + sse.flush()
    )

    assert "\"content_blocks\"" in out
