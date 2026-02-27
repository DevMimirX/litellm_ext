import json
import unittest


from litellm_ext.extensions.asgi.stream_usage_rewrite import _SSERewriter


def _collect_frames(sse_bytes: bytes, *, missing_input_tokens=None, multiplier=1.0):
    """Feed a full SSE transcript into the rewriter and return list[str] frames."""
    r = _SSERewriter(missing_input_tokens=missing_input_tokens, multiplier=multiplier)
    outs = []
    for chunk in r.feed(sse_bytes):
        outs.append(chunk)
    outs.extend(r.flush())
    merged = b"".join(outs).decode("utf-8", errors="replace")
    # Frames are separated by a blank line.
    frames = [f for f in merged.split("\n\n") if f.strip()]
    return frames


def _get_event_and_data(frame: str):
    event = None
    data_lines = []
    for line in frame.split("\n"):
        if line.startswith("event:"):
            event = line.split(":", 1)[1].strip()
        if line.startswith("data:"):
            data_lines.append(line.split(":", 1)[1].lstrip())
    data = "\n".join(data_lines).strip()
    obj = json.loads(data) if data else None
    return event, obj


class TestSSEUsageRewrite(unittest.TestCase):
    def test_glm_message_delta_is_not_under_message_start(self):
        """GLM example: message_start input_tokens=11, message_delta input_tokens=7.

        After patching with multiplier=1.05:
          - message_start input_tokens becomes ceil(11*1.05)=12
          - message_delta input_tokens is brought up to >= 12 (never under)
        """
        sse = (
            "event: message_start\n"
            'data: {"type":"message_start","message":{"id":"x","type":"message","role":"assistant","model":"glm-4.7","content":[],"stop_reason":null,"stop_sequence":null,"usage":{"input_tokens":11,"output_tokens":0}}}\n\n'
            "event: message_delta\n"
            'data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"input_tokens":7,"output_tokens":10}}\n\n'
            "event: message_stop\n"
            'data: {"type":"message_stop"}\n\n'
        ).encode("utf-8")

        frames = _collect_frames(sse, multiplier=1.05)

        ev0, obj0 = _get_event_and_data(frames[0])
        self.assertEqual(ev0, "message_start")
        self.assertEqual(obj0["message"]["usage"]["input_tokens"], 12)

        ev1, obj1 = _get_event_and_data(frames[1])
        self.assertEqual(ev1, "message_delta")
        self.assertEqual(obj1["usage"]["input_tokens"], 12)
        self.assertEqual(obj1["usage"]["output_tokens"], 10)

    def test_deepseek_chat_injects_prompt_usage_into_message_delta(self):
        """DeepSeek-chat example: message_delta only includes output_tokens.

        We inject scaled prompt usage into message_delta so downstream consumers
        can rely on it.
        """
        sse = (
            "event: message_start\n"
            'data: {"type":"message_start","message":{"id":"x","type":"message","role":"assistant","model":"deepseek-chat","content":[],"stop_reason":null,"stop_sequence":null,"usage":{"input_tokens":10,"cache_creation_input_tokens":0,"cache_read_input_tokens":0,"output_tokens":0,"service_tier":"standard"}}}\n\n'
            "event: message_delta\n"
            'data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":9}}\n\n'
        ).encode("utf-8")

        frames = _collect_frames(sse, multiplier=1.6)
        ev0, obj0 = _get_event_and_data(frames[0])
        self.assertEqual(obj0["message"]["usage"]["input_tokens"], 16)

        ev1, obj1 = _get_event_and_data(frames[1])
        self.assertEqual(obj1["usage"]["input_tokens"], 16)
        self.assertEqual(obj1["usage"].get("cache_read_input_tokens", 0), 0)
        self.assertEqual(obj1["usage"].get("cache_creation_input_tokens", 0), 0)
        self.assertEqual(obj1["usage"]["output_tokens"], 9)

    def test_deepseek_reasoner_scales_cache_read_and_propagates(self):
        """DeepSeek-reasoner can include large cache_read_input_tokens.

        We scale cache_read_input_tokens once and propagate into message_delta.
        """
        sse = (
            "event: message_start\n"
            'data: {"type":"message_start","message":{"id":"x","type":"message","role":"assistant","model":"deepseek-reasoner","content":[],"stop_reason":null,"stop_sequence":null,"usage":{"input_tokens":2063,"cache_creation_input_tokens":0,"cache_read_input_tokens":112704,"output_tokens":0,"service_tier":"standard"}}}\n\n'
            "event: message_delta\n"
            'data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":50}}\n\n'
        ).encode("utf-8")

        frames = _collect_frames(sse, multiplier=1.6)
        _, obj0 = _get_event_and_data(frames[0])
        self.assertEqual(obj0["message"]["usage"]["input_tokens"], 3301)
        self.assertEqual(obj0["message"]["usage"]["cache_read_input_tokens"], 180327)

        _, obj1 = _get_event_and_data(frames[1])
        self.assertEqual(obj1["usage"]["input_tokens"], 3301)
        self.assertEqual(obj1["usage"]["cache_read_input_tokens"], 180327)

    def test_idempotent_across_multiple_message_delta_frames(self):
        """If a backend emits multiple message_delta frames with prompt usage,
        we must not compound the multiplier.
        """
        sse = (
            "event: message_start\n"
            'data: {"type":"message_start","message":{"id":"x","type":"message","role":"assistant","model":"glm-4.7","content":[],"stop_reason":null,"stop_sequence":null,"usage":{"input_tokens":10,"output_tokens":0}}}\n\n'
            "event: message_delta\n"
            'data: {"type":"message_delta","delta":{"stop_reason":null},"usage":{"input_tokens":10,"output_tokens":1}}\n\n'
            "event: message_delta\n"
            'data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"input_tokens":10,"output_tokens":2}}\n\n'
        ).encode("utf-8")

        frames = _collect_frames(sse, multiplier=1.6)
        # message_start scales 10 -> 16
        _, obj0 = _get_event_and_data(frames[0])
        self.assertEqual(obj0["message"]["usage"]["input_tokens"], 16)

        # Both deltas should be 16 (not 26, etc.)
        _, obj1 = _get_event_and_data(frames[1])
        _, obj2 = _get_event_and_data(frames[2])
        self.assertEqual(obj1["usage"]["input_tokens"], 16)
        self.assertEqual(obj2["usage"]["input_tokens"], 16)


if __name__ == "__main__":
    unittest.main()
