from __future__ import annotations

import sys
import types
import asyncio


def test_streaming_sse_filters_only_data_and_raw_json_lines_and_handles_partial_tail():
    # Create fake target module already loaded
    target_name = "litellm.proxy.pass_through_endpoints.streaming_handler"
    fake_mod = types.ModuleType(target_name)

    class PassThroughStreamingHandler:
        _convert_raw_bytes_to_str_lines = staticmethod(lambda raw_bytes: ["BAD"])  # will be replaced

        @staticmethod
        async def _route_streaming_logging_to_handler(*args, **kwargs):
            raise ValueError("boom")

    fake_mod.PassThroughStreamingHandler = PassThroughStreamingHandler
    sys.modules[target_name] = fake_mod

    import litellm_ext.extensions.asgi.streaming_sse as sse
    sse.install()

    raw = [
        b": keepalive\n",
        b"event: ping\n",
        b"data: {\"a\":1}\n",
        b"\n",
        b"{\"b\":2}\n",
        b"data: [DONE]\n",
        b"id: 1\n",
        b"data: {\"c\":3}",  # no newline tail
    ]

    # Call as class attribute (no instance)
    lines1 = PassThroughStreamingHandler._convert_raw_bytes_to_str_lines(raw)
    # Call via instance (bound method edge-case)
    lines2 = PassThroughStreamingHandler()._convert_raw_bytes_to_str_lines(raw)

    assert lines1 == ["data: {\"a\":1}", "data: {\"b\":2}", "data: {\"c\":3}"]
    assert lines2 == lines1


def test_streaming_sse_suppresses_logging_exceptions():
    target_name = "litellm.proxy.pass_through_endpoints.streaming_handler"
    fake_mod = types.ModuleType(target_name)

    class PassThroughStreamingHandler:
        @staticmethod
        async def _route_streaming_logging_to_handler(*args, **kwargs):
            raise RuntimeError("boom")

    fake_mod.PassThroughStreamingHandler = PassThroughStreamingHandler
    sys.modules[target_name] = fake_mod

    import litellm_ext.extensions.asgi.streaming_sse as sse
    sse.install()

    # Should not raise
    out = asyncio.run(PassThroughStreamingHandler._route_streaming_logging_to_handler())
    assert out is None
