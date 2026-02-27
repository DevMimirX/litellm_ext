from __future__ import annotations

import asyncio
import sys
import types


def _install_fake_anthropic_count_tokens_handler(
    monkeypatch,
    *,
    remote_tokens: int | None = None,
    raise_remote: bool = False,
    remote_exc: Exception | None = None,
):
    class FakeAnthropicCountTokensHandler:
        calls = 0

        async def handle_count_tokens_request(self, model, messages, api_key, api_base=None, timeout=None):
            type(self).calls += 1
            if remote_exc is not None:
                raise remote_exc
            if raise_remote:
                raise RuntimeError("remote unavailable")
            out = {"source": "remote"}
            if remote_tokens is not None:
                out["input_tokens"] = remote_tokens
            return out

    litellm_mod = types.ModuleType("litellm")
    llms_mod = types.ModuleType("litellm.llms")
    anthropic_mod = types.ModuleType("litellm.llms.anthropic")
    count_tokens_mod = types.ModuleType("litellm.llms.anthropic.count_tokens")
    handler_mod = types.ModuleType("litellm.llms.anthropic.count_tokens.handler")

    handler_mod.AnthropicCountTokensHandler = FakeAnthropicCountTokensHandler
    count_tokens_mod.handler = handler_mod
    anthropic_mod.count_tokens = count_tokens_mod
    llms_mod.anthropic = anthropic_mod
    litellm_mod.llms = llms_mod

    monkeypatch.setitem(sys.modules, "litellm", litellm_mod)
    monkeypatch.setitem(sys.modules, "litellm.llms", llms_mod)
    monkeypatch.setitem(sys.modules, "litellm.llms.anthropic", anthropic_mod)
    monkeypatch.setitem(sys.modules, "litellm.llms.anthropic.count_tokens", count_tokens_mod)
    monkeypatch.setitem(sys.modules, "litellm.llms.anthropic.count_tokens.handler", handler_mod)

    return FakeAnthropicCountTokensHandler


def _install_fake_litellm_token_counter(monkeypatch, *, remote_tokens: int | None = None, raise_remote: bool = False):
    litellm_mod = types.ModuleType("litellm")

    def token_counter(*args, **kwargs):
        if raise_remote:
            raise RuntimeError("remote token_counter unavailable")
        if remote_tokens is None:
            return None
        return remote_tokens

    litellm_mod.token_counter = token_counter
    monkeypatch.setitem(sys.modules, "litellm", litellm_mod)
    return litellm_mod


def test_anthropic_count_tokens_falls_back_when_model_pattern_misses(monkeypatch):
    import litellm_ext.extensions.litellm.local_token_counter as ltc

    handler_cls = _install_fake_anthropic_count_tokens_handler(monkeypatch, remote_tokens=424242)

    monkeypatch.setattr(ltc, "_FORCE_ANTHROPIC_COUNT_TOKENS", True)
    monkeypatch.setattr(ltc, "_PROVIDERS", ("anthropic",))
    monkeypatch.setattr(ltc, "_MODEL_PATTERNS", ("claude-*",))
    monkeypatch.setattr(ltc, "_HOST_SUBSTRS", ("api.anthropic.com",))

    ltc._patch_anthropic_count_tokens()

    out = asyncio.run(
        handler_cls().handle_count_tokens_request(
            model="glm-5",
            messages=[{"role": "user", "content": "hello"}],
            api_key="sk-test",
            api_base="https://api.z.ai/api/anthropic",
        )
    )
    assert out == {"input_tokens": 424242, "source": "remote"}


def test_anthropic_count_tokens_prefers_remote_when_model_pattern_matches(monkeypatch):
    import litellm_ext.extensions.litellm.local_token_counter as ltc

    handler_cls = _install_fake_anthropic_count_tokens_handler(monkeypatch, remote_tokens=424242)

    monkeypatch.setattr(ltc, "_FORCE_ANTHROPIC_COUNT_TOKENS", True)
    monkeypatch.setattr(ltc, "_PROVIDERS", ("anthropic",))
    monkeypatch.setattr(ltc, "_MODEL_PATTERNS", ("claude-*",))
    monkeypatch.setattr(ltc, "_HOST_SUBSTRS", ("api.anthropic.com",))

    ltc._patch_anthropic_count_tokens()

    out = asyncio.run(
        handler_cls().handle_count_tokens_request(
            model="claude-3-5-sonnet",
            messages=[{"role": "user", "content": "hello"}],
            api_key="sk-test",
            api_base="https://api.anthropic.com",
        )
    )
    assert out == {"input_tokens": 424242, "source": "remote"}


def test_anthropic_count_tokens_returns_remote_when_remote_missing(monkeypatch):
    import litellm_ext.extensions.litellm.local_token_counter as ltc

    handler_cls = _install_fake_anthropic_count_tokens_handler(monkeypatch, remote_tokens=None)

    monkeypatch.setattr(ltc, "_FORCE_ANTHROPIC_COUNT_TOKENS", True)
    monkeypatch.setattr(ltc, "_PROVIDERS", ("anthropic",))
    monkeypatch.setattr(ltc, "_MODEL_PATTERNS", ("claude-*",))
    monkeypatch.setattr(ltc, "_HOST_SUBSTRS", ("api.anthropic.com",))

    ltc._patch_anthropic_count_tokens()

    out = asyncio.run(
        handler_cls().handle_count_tokens_request(
            model="claude-3-5-sonnet",
            messages=[{"role": "user", "content": "hello"}],
            api_key="sk-test",
            api_base="https://api.anthropic.com",
        )
    )
    assert out == {"source": "remote"}


def test_anthropic_count_tokens_injects_local_usage_when_remote_errors(monkeypatch):
    import litellm_ext.extensions.litellm.local_token_counter as ltc

    handler_cls = _install_fake_anthropic_count_tokens_handler(monkeypatch, raise_remote=True)

    monkeypatch.setattr(ltc, "_FORCE_ANTHROPIC_COUNT_TOKENS", True)
    monkeypatch.setattr(ltc, "_PROVIDERS", ("anthropic",))
    monkeypatch.setattr(ltc, "_MODEL_PATTERNS", ("claude-*",))
    monkeypatch.setattr(ltc, "_HOST_SUBSTRS", ("api.anthropic.com",))

    ltc._patch_anthropic_count_tokens()

    out = asyncio.run(
        handler_cls().handle_count_tokens_request(
            model="claude-3-5-sonnet",
            messages=[{"role": "user", "content": "hello"}],
            api_key="sk-test",
            api_base="https://api.anthropic.com",
        )
    )
    assert isinstance(out.get("input_tokens"), int)
    assert out["input_tokens"] > 0
    assert out.get("source") != "remote"


def test_anthropic_count_tokens_is_remote_first_on_each_401(monkeypatch):
    import litellm_ext.extensions.litellm.local_token_counter as ltc

    class AuthError(RuntimeError):
        status_code = 401

    handler_cls = _install_fake_anthropic_count_tokens_handler(
        monkeypatch,
        remote_exc=AuthError("401 Unauthorized for https://api.anthropic.com/v1/messages/count_tokens"),
    )

    monkeypatch.setattr(ltc, "_FORCE_ANTHROPIC_COUNT_TOKENS", True)
    monkeypatch.setattr(ltc, "_PROVIDERS", ("anthropic",))
    monkeypatch.setattr(ltc, "_MODEL_PATTERNS", ("doubao-*",))
    monkeypatch.setattr(ltc, "_HOST_SUBSTRS", ("api.anthropic.com",))

    ltc._patch_anthropic_count_tokens()

    out1 = asyncio.run(
        handler_cls().handle_count_tokens_request(
            model="doubao-seed-2.0-code",
            messages=[{"role": "user", "content": "hello"}],
            api_key="sk-test",
            api_base="https://api.anthropic.com",
        )
    )
    out2 = asyncio.run(
        handler_cls().handle_count_tokens_request(
            model="doubao-seed-2.0-code",
            messages=[{"role": "user", "content": "hello"}],
            api_key="sk-test",
            api_base="https://api.anthropic.com",
        )
    )

    assert isinstance(out1.get("input_tokens"), int) and out1["input_tokens"] > 0
    assert isinstance(out2.get("input_tokens"), int) and out2["input_tokens"] > 0
    assert handler_cls.calls == 2


def test_anthropic_count_tokens_patch_accepts_new_kwargs(monkeypatch):
    import litellm_ext.extensions.litellm.local_token_counter as ltc

    class FakeAnthropicCountTokensHandler:
        seen_extra_headers = None

        async def handle_count_tokens_request(
            self,
            model,
            messages,
            api_key,
            api_base=None,
            timeout=None,
            extra_headers=None,
        ):
            type(self).seen_extra_headers = extra_headers
            return {"input_tokens": 321, "source": "remote"}

    litellm_mod = types.ModuleType("litellm")
    llms_mod = types.ModuleType("litellm.llms")
    anthropic_mod = types.ModuleType("litellm.llms.anthropic")
    count_tokens_mod = types.ModuleType("litellm.llms.anthropic.count_tokens")
    handler_mod = types.ModuleType("litellm.llms.anthropic.count_tokens.handler")

    handler_mod.AnthropicCountTokensHandler = FakeAnthropicCountTokensHandler
    count_tokens_mod.handler = handler_mod
    anthropic_mod.count_tokens = count_tokens_mod
    llms_mod.anthropic = anthropic_mod
    litellm_mod.llms = llms_mod

    monkeypatch.setitem(sys.modules, "litellm", litellm_mod)
    monkeypatch.setitem(sys.modules, "litellm.llms", llms_mod)
    monkeypatch.setitem(sys.modules, "litellm.llms.anthropic", anthropic_mod)
    monkeypatch.setitem(sys.modules, "litellm.llms.anthropic.count_tokens", count_tokens_mod)
    monkeypatch.setitem(sys.modules, "litellm.llms.anthropic.count_tokens.handler", handler_mod)

    monkeypatch.setattr(ltc, "_FORCE_ANTHROPIC_COUNT_TOKENS", True)
    monkeypatch.setattr(ltc, "_PROVIDERS", ("anthropic",))
    monkeypatch.setattr(ltc, "_MODEL_PATTERNS", ("claude-*",))
    monkeypatch.setattr(ltc, "_HOST_SUBSTRS", ("api.anthropic.com",))

    ltc._patch_anthropic_count_tokens()

    out = asyncio.run(
        FakeAnthropicCountTokensHandler().handle_count_tokens_request(
            model="claude-3-5-sonnet",
            messages=[{"role": "user", "content": "hello"}],
            api_key="sk-test",
            api_base="https://api.anthropic.com",
            extra_headers={"x-test": "1"},
        )
    )
    assert out == {"input_tokens": 321, "source": "remote"}
    assert FakeAnthropicCountTokensHandler.seen_extra_headers == {"x-test": "1"}


def test_anthropic_count_tokens_requires_provider_or_host_match(monkeypatch):
    import litellm_ext.extensions.litellm.local_token_counter as ltc

    handler_cls = _install_fake_anthropic_count_tokens_handler(monkeypatch, remote_tokens=424242)

    monkeypatch.setattr(ltc, "_FORCE_ANTHROPIC_COUNT_TOKENS", True)
    monkeypatch.setattr(ltc, "_PROVIDERS", ("openai",))
    monkeypatch.setattr(ltc, "_MODEL_PATTERNS", ("claude-*",))
    monkeypatch.setattr(ltc, "_HOST_SUBSTRS", ("api.anthropic.com",))

    ltc._patch_anthropic_count_tokens()

    out = asyncio.run(
        handler_cls().handle_count_tokens_request(
            model="claude-3-5-sonnet",
            messages=[{"role": "user", "content": "hello"}],
            api_key="sk-test",
            api_base="https://api.z.ai/api/anthropic",
        )
    )
    assert out == {"input_tokens": 424242, "source": "remote"}


def test_matches_model_is_case_insensitive(monkeypatch):
    import litellm_ext.extensions.litellm.local_token_counter as ltc

    monkeypatch.setattr(ltc, "_MODEL_PATTERNS", ("MiniMax-*",))
    assert ltc._matches_model("minimax-m2.5")


def test_token_counter_prefers_remote_usage_when_available(monkeypatch):
    import litellm_ext.extensions.litellm.local_token_counter as ltc

    litellm_mod = _install_fake_litellm_token_counter(monkeypatch, remote_tokens=777)

    monkeypatch.setattr(ltc, "_FORCE_ANTHROPIC_COUNT_TOKENS", False)
    monkeypatch.setattr(ltc, "_PROVIDERS", ())
    monkeypatch.setattr(ltc, "_MODEL_PATTERNS", ("doubao-*",))
    monkeypatch.setattr(ltc, "_HOST_SUBSTRS", ())

    ltc.install()

    out = litellm_mod.token_counter(
        model="doubao-seed-2.0-code",
        messages=[{"role": "user", "content": "hello"}],
    )
    assert out == 777


def test_token_counter_injects_local_usage_when_remote_missing(monkeypatch):
    import litellm_ext.extensions.litellm.local_token_counter as ltc

    litellm_mod = _install_fake_litellm_token_counter(monkeypatch, remote_tokens=None)

    monkeypatch.setattr(ltc, "_FORCE_ANTHROPIC_COUNT_TOKENS", False)
    monkeypatch.setattr(ltc, "_PROVIDERS", ())
    monkeypatch.setattr(ltc, "_MODEL_PATTERNS", ("doubao-*",))
    monkeypatch.setattr(ltc, "_HOST_SUBSTRS", ())

    ltc.install()

    out = litellm_mod.token_counter(
        model="doubao-seed-2.0-code",
        messages=[{"role": "user", "content": "hello"}],
    )
    assert isinstance(out, int)
    assert out > 0


def test_token_counter_injects_local_usage_when_remote_errors(monkeypatch):
    import litellm_ext.extensions.litellm.local_token_counter as ltc

    litellm_mod = _install_fake_litellm_token_counter(monkeypatch, raise_remote=True)

    monkeypatch.setattr(ltc, "_FORCE_ANTHROPIC_COUNT_TOKENS", False)
    monkeypatch.setattr(ltc, "_PROVIDERS", ())
    monkeypatch.setattr(ltc, "_MODEL_PATTERNS", ("doubao-*",))
    monkeypatch.setattr(ltc, "_HOST_SUBSTRS", ())

    ltc.install()

    out = litellm_mod.token_counter(
        model="doubao-seed-2.0-code",
        messages=[{"role": "user", "content": "hello"}],
    )
    assert isinstance(out, int)
    assert out > 0


def test_calibration_uses_scope_default_when_no_history(monkeypatch):
    import litellm_ext.extensions.litellm.local_token_counter as ltc

    monkeypatch.setattr(ltc, "_CALIBRATION_FACTORS", {})
    got = ltc._get_calibration_factor("doubao-seed-2.0-code", "messages")
    assert got == ltc._CALIBRATION_DEFAULT_BY_SCOPE["messages"]


def test_calibration_corrects_overestimation_faster_than_underestimation(monkeypatch):
    import litellm_ext.extensions.litellm.local_token_counter as ltc

    key = ltc._calibration_key("doubao-seed-2.0-code", "messages")

    # Overestimation case: observed factor is lower than previous.
    monkeypatch.setattr(ltc, "_CALIBRATION_FACTORS", {key: 1.0})
    down = ltc._update_calibration_factor("doubao-seed-2.0-code", "messages", local_raw=100, remote_tokens=50)

    # Underestimation case: observed factor is higher than previous.
    monkeypatch.setattr(ltc, "_CALIBRATION_FACTORS", {key: 1.0})
    up = ltc._update_calibration_factor("doubao-seed-2.0-code", "messages", local_raw=100, remote_tokens=150)

    assert (1.0 - down) > (up - 1.0)


def test_log_compare_uses_alias_when_api_base_matches(monkeypatch, tmp_path):
    import litellm_ext.extensions.litellm.local_token_counter as ltc

    cfg = tmp_path / "litellm.yaml"
    cfg.write_text(
        """
model_list:
  - model_name: glm-5
    litellm_params:
      model: anthropic/glm-5
      api_base: "https://api.z.ai/api/anthropic"
  - model_name: glm-5-ali
    litellm_params:
      model: anthropic/glm-5
      api_base: "https://coding.dashscope.aliyuncs.com/apps/anthropic"
""".strip()
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("LITELLM_CONFIG", str(cfg))

    logs = []

    class _FakeLogger:
        def debug(self, msg):
            logs.append(msg)

    monkeypatch.setattr(ltc, "_LOG", _FakeLogger())

    ltc._log_compare(
        kind="token_counter",
        model="glm-5",
        scope="messages",
        used="remote",
        remote=42,
        local_raw=56,
        local_adj=42,
        factor=0.75,
        api_base="https://coding.dashscope.aliyuncs.com/apps/anthropic",
    )

    assert logs
    assert "model='glm-5-ali' resolved_model='glm-5'" in logs[0]
