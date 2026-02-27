import json

from litellm_ext.adapters.anthropic_openai import (
    anthropic_to_openai_messages,
    anthropic_response_to_openai,
    map_anthropic_stop_reason_to_openai,
    map_openai_finish_reason_to_anthropic,
    openai_to_anthropic_messages,
    openai_response_to_anthropic,
)


def test_anthropic_to_openai_system_and_tools():
    payload = {
        "model": "claude-3-opus",
        "system": "You are helpful",
        "max_tokens": 10,
        "messages": [{"role": "user", "content": "hi"}],
        "tools": [
            {
                "name": "echo",
                "description": "Echo",
                "input_schema": {"type": "object", "properties": {"text": {"type": "string"}}},
            }
        ],
        "stop_sequences": ["\n\n"],
    }
    out = anthropic_to_openai_messages(payload)
    assert out["model"] == "claude-3-opus"
    assert out["messages"][0]["role"] == "system"
    assert out["messages"][0]["content"] == "You are helpful"
    assert out["messages"][1]["role"] == "user"
    assert out["messages"][1]["content"] == "hi"
    assert out["tools"][0]["type"] == "function"
    assert out["tools"][0]["function"]["name"] == "echo"
    assert out["stop"] == ["\n\n"]


def test_anthropic_to_openai_metadata_user():
    payload = {
        "model": "claude-3-opus",
        "messages": [{"role": "user", "content": "hi"}],
        "metadata": {"user_id": "user-123"},
    }
    out = anthropic_to_openai_messages(payload)
    assert out["user"] == "user-123"


def test_anthropic_to_openai_tool_use_and_result():
    payload = {
        "model": "claude-3-opus",
        "messages": [
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "check"},
                    {"type": "tool_use", "id": "call_1", "name": "echo", "input": {"text": "hi"}},
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "call_1", "content": "ok"}
                ],
            },
        ],
    }
    out = anthropic_to_openai_messages(payload)
    msg0 = out["messages"][0]
    assert msg0["role"] == "assistant"
    assert msg0.get("tool_calls")
    assert msg0["tool_calls"][0]["id"] == "call_1"
    # tool_result becomes tool role message
    msg1 = out["messages"][1]
    assert msg1["role"] == "tool"
    assert msg1["tool_call_id"] == "call_1"
    assert msg1["content"] == "ok"


def test_openai_to_anthropic_system_and_tools():
    payload = {
        "model": "gpt-4",
        "messages": [
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "hi"},
        ],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "echo",
                    "description": "Echo",
                    "parameters": {"type": "object", "properties": {"text": {"type": "string"}}},
                },
            }
        ],
        "stop": ["\n\n"],
    }
    out = openai_to_anthropic_messages(payload)
    assert out["system"] == "You are helpful"
    assert out["messages"][0]["role"] == "user"
    assert out["tools"][0]["name"] == "echo"
    assert out["stop_sequences"] == ["\n\n"]




def test_openai_to_anthropic_image_url_data():
    payload = {
        "model": "gpt-4",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/png;base64,AAA"},
                    }
                ],
            }
        ],
    }
    out = openai_to_anthropic_messages(payload)
    block = out["messages"][0]["content"][0]
    assert block["type"] == "image"
    assert block["source"]["media_type"] == "image/png"
    assert block["source"]["data"] == "AAA"


def test_openai_to_anthropic_tool_choice_mapping():
    payload = {
        "model": "gpt-4",
        "messages": [{"role": "user", "content": "hi"}],
        "tool_choice": "required",
    }
    out = openai_to_anthropic_messages(payload)
    assert out["tool_choice"] == {"type": "any"}


def test_openai_to_anthropic_tool_type_passthrough():
    payload = {
        "model": "gpt-4",
        "messages": [{"role": "user", "content": "hi"}],
        "tools": [
            {
                "type": "mcp_toolset",
                "mcp_toolset": {"name": "mcp", "server": "local"},
            }
        ],
    }
    out = openai_to_anthropic_messages(payload)
    assert out["tools"][0]["name"] == "mcp_toolset"


def test_openai_response_to_anthropic():
    payload = {
        "id": "chatcmpl-1",
        "model": "gpt-4",
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "Hello",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "echo", "arguments": json.dumps({"text": "hi"})},
                        }
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5},
    }
    out = openai_response_to_anthropic(payload)
    assert out["role"] == "assistant"
    assert out["stop_reason"] == "tool_use"
    assert out["usage"]["input_tokens"] == 10
    assert out["usage"]["output_tokens"] == 5




def test_openai_response_to_anthropic_reasoning_non_strict():
    payload = {
        "id": "chatcmpl-9",
        "model": "gpt-4",
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "Hello",
                    "reasoning": "step 1",
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1},
    }
    out = openai_response_to_anthropic(payload, strict=False)
    assert out["content"][0]["type"] == "thinking"
    assert out["content"][0]["thinking"] == "step 1"


def test_anthropic_response_to_openai_reasoning_non_strict():
    payload = {
        "id": "msg_9",
        "type": "message",
        "role": "assistant",
        "model": "claude-sonnet-4-5",
        "content": [
            {"type": "thinking", "thinking": "step a"},
            {"type": "text", "text": "Hello"},
        ],
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 1, "output_tokens": 1},
    }
    out = anthropic_response_to_openai(payload, strict=False)
    msg = out["choices"][0]["message"]
    assert msg["reasoning"] == "step a"


def test_anthropic_response_to_openai():
    payload = {
        "id": "msg_1",
        "type": "message",
        "role": "assistant",
        "model": "claude-sonnet-4-5",
        "content": [
            {"type": "text", "text": "Hello"},
            {"type": "tool_use", "id": "call_1", "name": "echo", "input": {"text": "hi"}},
        ],
        "stop_reason": "tool_use",
        "usage": {"input_tokens": 12, "output_tokens": 3},
    }
    out = anthropic_response_to_openai(payload)
    assert out["model"] == "claude-sonnet-4-5"
    assert out["choices"][0]["message"]["role"] == "assistant"
    assert out["choices"][0]["finish_reason"] == "tool_calls"
    assert out["usage"]["prompt_tokens"] == 12
    assert out["usage"]["completion_tokens"] == 3


def test_finish_reason_maps():
    assert map_anthropic_stop_reason_to_openai("tool_use") == "tool_calls"
    assert map_anthropic_stop_reason_to_openai("end_turn") == "stop"
    assert map_anthropic_stop_reason_to_openai("tool_use", strict=False) == "tool_use"
    assert map_openai_finish_reason_to_anthropic("tool_calls") == "tool_use"
    assert map_openai_finish_reason_to_anthropic("length") == "max_tokens"
    assert map_openai_finish_reason_to_anthropic("tool_calls", strict=False) == "tool_calls"
