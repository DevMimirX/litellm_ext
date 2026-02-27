import json


from litellm_ext.extensions.asgi.stream_usage_rewrite import _SSERewriter


def test_patches_first_message_start_only_when_zero():
    s = (
        "event: message_start\n"
        "data: {\"type\":\"message_start\",\"message\":{\"id\":\"msg\",\"usage\":{\"input_tokens\":0,\"output_tokens\":0}}}\n\n"
        "event: content_block_start\n"
        "data: {\"type\":\"content_block_start\"}\n\n"
        "event: message_start\n"
        "data: {\"type\":\"message_start\",\"message\":{\"id\":\"msg2\",\"usage\":{\"input_tokens\":0,\"output_tokens\":0}}}\n\n"
    )

    r = _SSERewriter(missing_input_tokens=123, multiplier=1.0)
    out = b"".join(r.feed(s[:80]) + r.feed(s[80:]) + r.flush())
    out_s = out.decode("utf-8")

    # First message_start patched
    first_frame = out_s.split("\n\n", 1)[0]
    data_line = [l for l in first_frame.split("\n") if l.startswith("data:")][0]
    obj = json.loads(data_line.split(":", 1)[1].strip())
    assert obj["message"]["usage"]["input_tokens"] == 123

    # Second message_start should remain 0 (only patch first)
    assert "msg2" in out_s
    assert "\"input_tokens\":0" in out_s


def test_does_not_patch_when_nonzero():
    s = (
        "event: message_start\n"
        "data: {\"type\":\"message_start\",\"message\":{\"usage\":{\"input_tokens\":7,\"output_tokens\":0}}}\n\n"
    )
    r = _SSERewriter(missing_input_tokens=999, multiplier=1.0)
    out = b"".join(r.feed(s) + r.flush())
    out_s = out.decode("utf-8")
    assert "\"input_tokens\":7" in out_s
    assert "999" not in out_s


def test_patches_message_delta_usage_and_scales_cache_read():
    s = (
        "event: message_start\n"
        "data: {\"type\":\"message_start\",\"message\":{\"usage\":{\"input_tokens\":7,\"output_tokens\":0}}}\n\n"
        "event: message_delta\n"
        "data: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"},"
        "\"usage\":{\"input_tokens\":10,\"cache_read_input_tokens\":90,\"output_tokens\":1}}\n\n"
    )

    r = _SSERewriter(missing_input_tokens=999, multiplier=2.0)
    out = b"".join(r.feed(s) + r.flush())
    out_s = out.decode("utf-8")

    # message_delta should have scaled input + cache_read
    frames = out_s.split("\n\n")
    delta_frame = [f for f in frames if "message_delta" in f][0]
    data_line = [l for l in delta_frame.split("\n") if l.startswith("data:")][0]
    obj = json.loads(data_line.split(":", 1)[1].strip())
    assert obj["usage"]["input_tokens"] == 20
    assert obj["usage"]["cache_read_input_tokens"] == 180
    assert obj["usage"]["output_tokens"] == 1


def test_message_delta_injects_estimate_when_all_zero():
    s = (
        "event: message_delta\n"
        "data: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"},"
        "\"usage\":{\"input_tokens\":0,\"cache_read_input_tokens\":0,\"output_tokens\":0}}\n\n"
    )
    r = _SSERewriter(missing_input_tokens=1234, multiplier=1.6)
    out = b"".join(r.feed(s) + r.flush())
    out_s = out.decode("utf-8")
    data_line = [l for l in out_s.split("\n") if l.startswith("data:")][0]
    obj = json.loads(data_line.split(":", 1)[1].strip())
    assert obj["usage"]["input_tokens"] == 1234
    assert obj["usage"]["cache_read_input_tokens"] == 0
