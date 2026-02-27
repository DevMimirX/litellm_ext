from __future__ import annotations

from typing import Callable, Iterable, List, Tuple

from ..core.patch import PatchSettings

ExtensionInstall = Callable[[], None]


def _entry(name: str, install: ExtensionInstall, settings: PatchSettings, order: int) -> Tuple[int, str, ExtensionInstall, PatchSettings]:
    return (order, name, install, settings)


def iter_extensions() -> Iterable[Tuple[int, str, ExtensionInstall, PatchSettings]]:
    from .httpx.count_tokens_stub import install as install_count_tokens_stub, SETTINGS as COUNT_TOKENS_STUB
    from .httpx.hard_caps import install as install_hard_caps, SETTINGS as HARD_CAPS
    from .httpx.reasoning_replay import install as install_reasoning, SETTINGS as REASONING_REPLAY
    from .litellm.local_token_counter import install as install_local_token_counter, SETTINGS as LOCAL_TOKEN_COUNTER
    from .asgi.stream_usage_rewrite import install as install_stream_usage, SETTINGS as STREAM_USAGE_REWRITE
    from .asgi.hard_caps_messages import install as install_hard_caps_messages, SETTINGS as HARD_CAPS_MESSAGES
    from .asgi.streaming_sse import install as install_streaming_sse, SETTINGS as STREAMING_SSE
    from .asgi.transform import install as install_transform, SETTINGS as TRANSFORM
    from .suppress_warnings import install as install_suppress_warnings, SETTINGS as SUPPRESS_WARNINGS

    entries = [
        _entry("suppress_warnings", install_suppress_warnings, SUPPRESS_WARNINGS, 5),
        _entry("count_tokens_stub", install_count_tokens_stub, COUNT_TOKENS_STUB, 10),
        _entry("hard_caps", install_hard_caps, HARD_CAPS, 20),
        _entry("local_token_counter", install_local_token_counter, LOCAL_TOKEN_COUNTER, 25),
        _entry("reasoning_replay", install_reasoning, REASONING_REPLAY, 30),
        _entry("hard_caps_messages", install_hard_caps_messages, HARD_CAPS_MESSAGES, 35),
        _entry("stream_usage_rewrite", install_stream_usage, STREAM_USAGE_REWRITE, 40),
        _entry("streaming_sse", install_streaming_sse, STREAMING_SSE, 50),
        _entry("transform", install_transform, TRANSFORM, 60),
    ]
    return sorted(entries, key=lambda e: (e[0], e[1]))


def install_all() -> List[str]:
    installed: List[str] = []
    for _order, name, install, settings in iter_extensions():
        if not settings.is_enabled():
            continue
        install()
        installed.append(name)
    return installed
