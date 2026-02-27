import json

import litellm_ext.extensions.asgi.stream_usage_rewrite as m


def _extract_data(frame: str) -> dict:
    """Return JSON from the first data: line in a frame."""
    for line in frame.replace("\r\n", "\n").split("\n"):
        if line.lstrip().startswith("data:"):
            payload = line.split(":", 1)[1].strip()
            return json.loads(payload)
    raise AssertionError("no data: line found")


def test_patch_message_start_frame_patches_when_zero():
    frame = (
        "event: message_start\n"
        "data: {\"type\":\"message_start\",\"message\":{\"usage\":{\"input_tokens\":0,\"output_tokens\":0}}}\n"
    )
    patched, new_frame = m._patch_message_start_frame(frame, missing_input_tokens=123, multiplier=1.0)
    assert patched is True
    obj = _extract_data(new_frame)
    assert obj["message"]["usage"]["input_tokens"] == 123
    assert obj["message"]["usage"]["output_tokens"] == 0


def test_patch_message_start_frame_does_not_overwrite_nonzero():
    frame = (
        "event: message_start\n"
        "data: {\"type\":\"message_start\",\"message\":{\"usage\":{\"input_tokens\":7,\"output_tokens\":0}}}\n"
    )
    patched, new_frame = m._patch_message_start_frame(frame, missing_input_tokens=999, multiplier=1.0)
    assert patched is False
    assert new_frame == frame


def test_patch_message_start_frame_accepts_json_type_without_event_line():
    frame = "data: {\"type\":\"message_start\",\"message\":{\"usage\":{\"input_tokens\":0}}}\n"
    patched, new_frame = m._patch_message_start_frame(frame, missing_input_tokens=55, multiplier=1.0)
    assert patched is True
    obj = _extract_data(new_frame)
    assert obj["message"]["usage"]["input_tokens"] == 55


def test_patch_message_start_frame_multiline_data_is_supported():
    # Split JSON across multiple data: lines; newline is valid JSON whitespace.
    frame = (
        "event: message_start\n"
        "data: {\"type\":\"message_start\",\n"
        "data: \"message\":{\"usage\":{\"input_tokens\":0}}}\n"
    )
    patched, new_frame = m._patch_message_start_frame(frame, missing_input_tokens=88, multiplier=1.0)
    assert patched is True
    # We normalize to a single data: line.
    assert new_frame.count("data:") == 1
    obj = _extract_data(new_frame)
    assert obj["message"]["usage"]["input_tokens"] == 88


def test_rewriter_patches_only_first_message_start():
    s = (
        "event: message_start\n"
        "data: {\"type\":\"message_start\",\"message\":{\"usage\":{\"input_tokens\":0,\"output_tokens\":0}}}\n\n"
        "event: message_start\n"
        "data: {\"type\":\"message_start\",\"message\":{\"usage\":{\"input_tokens\":0,\"output_tokens\":0}}}\n\n"
    )
    r = m._SSERewriter(missing_input_tokens=321, multiplier=1.0)
    out = b"".join(r.feed(s.encode("utf-8")))
    out += b"".join(r.flush())

    text = out.decode("utf-8")
    frames = [f for f in text.split("\n\n") if f.strip()]
    assert len(frames) == 2
    obj1 = _extract_data(frames[0])
    obj2 = _extract_data(frames[1])
    assert obj1["message"]["usage"]["input_tokens"] == 321
    # second remains unpatched (still 0)
    assert obj2["message"]["usage"]["input_tokens"] == 0


def test_rewriter_handles_chunk_boundaries():
    s = (
        "event: message_start\n"
        "data: {\"type\":\"message_start\",\"message\":{\"usage\":{\"input_tokens\":0,\"output_tokens\":0}}}\n\n"
        "event: content_block_delta\n"
        "data: {\"type\":\"content_block_delta\"}\n\n"
    )
    # split in the middle of the JSON and delimiter
    chunks = [s[:40].encode(), s[40:73].encode(), s[73:].encode()]

    r = m._SSERewriter(missing_input_tokens=11, multiplier=1.0)
    out = b"".join(sum((r.feed(c) for c in chunks), []))
    out += b"".join(r.flush())
    text = out.decode("utf-8")
    assert "event: message_start" in text
    # ensure patched
    first_frame = text.split("\n\n", 1)[0]
    obj = _extract_data(first_frame)
    assert obj["message"]["usage"]["input_tokens"] == 11


def test_rewriter_supports_crlf_newlines():
    s = (
        "event: message_start\r\n"
        "data: {\"type\":\"message_start\",\"message\":{\"usage\":{\"input_tokens\":0,\"output_tokens\":0}}}\r\n\r\n"
    )
    r = m._SSERewriter(missing_input_tokens=9, multiplier=1.0)
    out = b"".join(r.feed(s.encode("utf-8"))) + b"".join(r.flush())
    obj = _extract_data(out.decode("utf-8").split("\n\n", 1)[0])
    assert obj["message"]["usage"]["input_tokens"] == 9


def test_model_matching_defaults_glm_and_deepseek():
    assert m._model_matches("glm-4.7") is True
    assert m._model_matches("glm-4.7-turbo") is True
    assert m._model_matches("deepseek-chat") is True


def test_patch_message_start_frame_scales_when_nonzero_and_multiplier_not_one():
    frame = (
        "event: message_start\n"
        "data: {\"type\":\"message_start\",\"message\":{\"usage\":{\"input_tokens\":7,\"output_tokens\":0}}}\n"
    )
    patched, new_frame = m._patch_message_start_frame(frame, missing_input_tokens=None, multiplier=1.5)
    assert patched is True
    obj = _extract_data(new_frame)
    # ceil(7*1.5)=11
    assert obj["message"]["usage"]["input_tokens"] == 11


def test_rewriter_scales_nonzero_when_multiplier_not_one():
    s = (
        "event: message_start\n"
        "data: {\"type\":\"message_start\",\"message\":{\"usage\":{\"input_tokens\":10,\"output_tokens\":0}}}\n\n"
        "event: message_stop\n"
        "data: {\"type\":\"message_stop\"}\n\n"
    )
    r = m._SSERewriter(missing_input_tokens=None, multiplier=1.2)
    out = b"".join(r.feed(s.encode("utf-8"))) + b"".join(r.flush())
    text = out.decode("utf-8")
    first_frame = text.split("\n\n", 1)[0]
    obj = _extract_data(first_frame)
    assert obj["message"]["usage"]["input_tokens"] == 12
