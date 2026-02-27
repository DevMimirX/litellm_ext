import importlib
import json

import httpx
import pytest

from litellm_ext.core.model_alias import reset_model_alias_cache


def _reload_policy(monkeypatch, **env):
    """
    litellm_ext.policy reads env at import time, so we must reload after setting env.
    """
    controlled_envs = (
        "LITELLM_EXT_CONFIG_PATH",
        "LITELLM_EXT_COMPACT_ROUTE",
        "LITELLM_COMPACT_MODEL",
        "LITELLM_EXT_COMPACT_ROUTE_MODE",
    )
    for key in controlled_envs:
        if key not in env:
            monkeypatch.delenv(key, raising=False)
    for k, v in env.items():
        if v is None:
            monkeypatch.delenv(k, raising=False)
        else:
            monkeypatch.setenv(k, str(v))
    import litellm_ext.policy as policy  # noqa
    return importlib.reload(policy)


def _reload_hard_caps(monkeypatch, **env):
    """
    hard caps imports enforce() from policy at import time, so reload both.
    """
    policy = _reload_policy(monkeypatch, **env)
    import litellm_ext.extensions.httpx.hard_caps as hard_caps  # noqa
    hard_caps = importlib.reload(hard_caps)
    reset_model_alias_cache()
    return policy, hard_caps


# -----------------------------
# Compact routing: detection
# -----------------------------

def test_compact_routes_when_marker_in_messages_string(monkeypatch):
    policy = _reload_policy(
        monkeypatch,
        LITELLM_EXT_COMPACT_ROUTE="1",
        LITELLM_COMPACT_MODEL="glm-4.7",
    )

    payload = {
        "model": "deepseek-chat",
        "max_tokens": 50,
        "messages": [{"role": "user", "content": "<command-name>/compact</command-name>"}],
    }

    changed = policy.enforce(payload, allow_raise=True)

    assert changed is True
    assert payload["model"] == "glm-4.7"


def test_compact_routes_when_plain_slash_compact_command(monkeypatch):
    policy = _reload_policy(
        monkeypatch,
        LITELLM_EXT_COMPACT_ROUTE="1",
        LITELLM_COMPACT_MODEL="glm-4.7",
        LITELLM_EXT_COMPACT_ROUTE_MODE="explicit",
    )

    payload = {
        "model": "deepseek-chat",
        "max_tokens": 50,
        "messages": [{"role": "user", "content": "/compact"}],
    }

    changed = policy.enforce(payload, allow_raise=True)

    assert changed is True
    assert payload["model"] == "glm-4.7"


def test_compact_routes_on_strict_autocompact_signature_in_explicit_mode(monkeypatch):
    policy = _reload_policy(
        monkeypatch,
        LITELLM_EXT_COMPACT_ROUTE="1",
        LITELLM_COMPACT_MODEL="glm-4.7",
        LITELLM_EXT_COMPACT_ROUTE_MODE="explicit",
    )

    payload = {
        "model": "deepseek-chat",
        "max_tokens": 50,
        "messages": [
            {
                "role": "user",
                "content": (
                    "Our task is to create a detailed summary of the conversation so far.\n"
                    "IMPORTANT: Do NOT use any tools.\n"
                    "You MUST respond with ONLY the <summary>...</summary> block as your text output."
                ),
            }
        ],
    }

    changed = policy.enforce(payload, allow_raise=True)

    assert changed is True
    assert payload["model"] == "glm-4.7"


def test_compact_does_not_route_on_summary_phrase_in_explicit_mode(monkeypatch):
    policy = _reload_policy(
        monkeypatch,
        LITELLM_EXT_COMPACT_ROUTE="1",
        LITELLM_COMPACT_MODEL="glm-4.7",
        LITELLM_EXT_COMPACT_ROUTE_MODE="explicit",
    )

    payload = {
        "model": "deepseek-chat",
        "max_tokens": 50,
        "system": [{"type": "text", "text": "Your task is to create a detailed summary of the conversation so far"}],
        "messages": [{"role": "user", "content": "hello"}],
    }

    changed = policy.enforce(payload, allow_raise=True)

    assert changed is False
    assert payload["model"] == "deepseek-chat"


def test_compact_routes_on_phrase_when_pattern_mode_enabled(monkeypatch, tmp_path):
    cfg_path = tmp_path / "extensions.yaml"
    cfg_path.write_text(
        """version: 1
policy:
  compact_routing:
    enabled: true
    mode: pattern
    target_model: glm-4.7
    patterns:
      - "Your task is to create a detailed summary of the conversation so far"
  model_limits:
    '*':
      max_output: 20000
      max_context: 180000
""",
        encoding="utf-8",
    )
    policy = _reload_policy(
        monkeypatch,
        LITELLM_EXT_CONFIG_PATH=str(cfg_path),
        LITELLM_EXT_COMPACT_ROUTE="1",
    )

    payload = {
        "model": "deepseek-chat",
        "max_tokens": 50,
        "system": [{"type": "text", "text": "Your task is to create a detailed summary of the conversation so far"}],
        "messages": [{"role": "user", "content": "hello"}],
    }

    changed = policy.enforce(payload, allow_raise=True)

    assert changed is True
    assert payload["model"] == "glm-4.7"


def test_compact_routes_when_anthropic_content_blocks(monkeypatch):
    policy = _reload_policy(
        monkeypatch,
        LITELLM_EXT_COMPACT_ROUTE="1",
        LITELLM_COMPACT_MODEL="glm-4.7",
    )

    payload = {
        "model": "deepseek-chat",
        "max_tokens": 50,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Some preface..."},
                    {"type": "text", "text": "<command-message>compact</command-message>"},
                ],
            }
        ],
    }

    changed = policy.enforce(payload, allow_raise=True)

    assert changed is True
    assert payload["model"] == "glm-4.7"


def test_compact_does_not_retrigger_from_old_history_marker(monkeypatch):
    policy = _reload_policy(
        monkeypatch,
        LITELLM_EXT_COMPACT_ROUTE="1",
        LITELLM_COMPACT_MODEL="glm-4.7",
    )

    payload = {
        "model": "deepseek-chat",
        "max_tokens": 50,
        "messages": [
            {"role": "user", "content": "<command-name>/compact</command-name>"},
            {"role": "assistant", "content": "summary done"},
            {"role": "user", "content": "normal follow-up question"},
        ],
    }

    changed = policy.enforce(payload, allow_raise=True)

    assert changed is False
    assert payload["model"] == "deepseek-chat"


def test_compact_does_not_route_when_disabled(monkeypatch):
    policy = _reload_policy(
        monkeypatch,
        LITELLM_EXT_COMPACT_ROUTE="0",
        LITELLM_COMPACT_MODEL="glm-4.7",
    )

    payload = {
        "model": "deepseek-chat",
        "max_tokens": 50,
        "messages": [{"role": "user", "content": "<command-name>/compact</command-name>"}],
    }

    changed = policy.enforce(payload, allow_raise=True)

    # With route disabled, only caps could change. Here max_tokens=50 is below cap, so no change.
    assert changed is False
    assert payload["model"] == "deepseek-chat"


def test_compact_no_rewrite_if_already_target_model(monkeypatch):
    policy = _reload_policy(
        monkeypatch,
        LITELLM_EXT_COMPACT_ROUTE="1",
        LITELLM_COMPACT_MODEL="glm-4.7",
    )

    payload = {
        "model": "GLM-4.7",  # mixed case to test normalize_model()
        "max_tokens": 50,
        "messages": [{"role": "user", "content": "<command-name>/compact</command-name>"}],
    }

    changed = policy.enforce(payload, allow_raise=True)

    # No routing rewrite needed; max_tokens already within * cap => no change
    assert changed is False
    assert payload["model"] == "GLM-4.7"


def test_compact_logs_when_already_target_model(monkeypatch):
    policy = _reload_policy(
        monkeypatch,
        LITELLM_EXT_COMPACT_ROUTE="1",
        LITELLM_COMPACT_MODEL="glm-4.7",
    )

    import litellm_ext.policy.engine as engine

    logs = []
    monkeypatch.setattr(engine, "dbg", lambda msg: logs.append(msg))

    payload = {
        "model": "GLM-4.7",
        "max_tokens": 50,
        "messages": [{"role": "user", "content": "<command-name>/compact</command-name>"}],
    }

    changed = policy.enforce(payload, allow_raise=True)

    assert changed is False
    assert any("route /compact 'GLM-4.7' -> 'glm-4.7' (already target)" in m for m in logs)


# -----------------------------
# Compact routing + caps behavior
# -----------------------------

def test_compact_routing_affects_which_cap_applies(monkeypatch):
    policy = _reload_policy(
        monkeypatch,
        LITELLM_EXT_COMPACT_ROUTE="1",
        LITELLM_COMPACT_MODEL="glm-4.7",
    )

    payload = {
        "model": "deepseek-chat",
        "max_tokens": 20000,
        "messages": [{"role": "user", "content": "<command-name>/compact</command-name>"}],
    }

    changed = policy.enforce(payload, allow_raise=True)

    assert changed is True
    # Routed to glm -> cap should come from "*" (20000 by default in template) unless you add a glm-* entry
    assert policy.normalize_model(payload["model"]) == "glm-4.7"
    assert payload["max_tokens"] == 20000


def test_non_compact_deepseek_chat_uses_deepseek_cap(monkeypatch):
    policy = _reload_policy(
        monkeypatch,
        LITELLM_EXT_COMPACT_ROUTE="1",
        LITELLM_COMPACT_MODEL="glm-4.7",
    )

    payload = {
        "model": "deepseek-chat",
        "max_tokens": 20000,
        "messages": [{"role": "user", "content": "normal message"}],
    }

    changed = policy.enforce(payload, allow_raise=True)

    assert changed is True
    assert policy.normalize_model(payload["model"]) == "deepseek-chat"
    assert payload["max_tokens"] == 8192


# -----------------------------
# Positional model argument edge case
# -----------------------------

def test_compact_rewrites_positional_model_arg_via_enforce_from_args_kwargs(monkeypatch):
    policy = _reload_policy(
        monkeypatch,
        LITELLM_EXT_COMPACT_ROUTE="1",
        LITELLM_COMPACT_MODEL="glm-4.7",
    )

    args = ("deepseek-chat",)
    kwargs = {
        "max_tokens": 20000,
        "messages": [{"role": "user", "content": "<command-message>compact</command-message>"}],
    }

    new_args, new_kwargs = policy.enforce_from_args_kwargs(args, kwargs, allow_raise=True)

    assert new_args[0] == "glm-4.7"
    assert new_kwargs["max_tokens"] == 20000


def test_compact_does_not_duplicate_model_kwarg(monkeypatch):
    policy = _reload_policy(
        monkeypatch,
        LITELLM_EXT_COMPACT_ROUTE="1",
        LITELLM_COMPACT_MODEL="glm-4.7",
    )

    args = ("deepseek-chat",)
    kwargs = {
        "model": "deepseek-chat",  # explicit kwarg takes precedence; args are ignored by policy wrapper
        "max_tokens": 20000,
        "messages": [{"role": "user", "content": "<command-name>/compact</command-name>"}],
    }

    new_args, new_kwargs = policy.enforce_from_args_kwargs(args, kwargs, allow_raise=True)

    # model is in kwargs so args remain unchanged; kwargs model is rewritten
    assert new_args == args
    assert policy.normalize_model(new_kwargs["model"]) == "glm-4.7"
    assert new_kwargs["max_tokens"] == 20000


# -----------------------------
# httpx mutator edge cases
# -----------------------------

def test_httpx_mutator_routes_compact_and_updates_content_length(monkeypatch):
    policy, hard_caps = _reload_hard_caps(
        monkeypatch,
        LITELLM_EXT_COMPACT_ROUTE="1",
        LITELLM_COMPACT_MODEL="glm-4.7",
    )

    payload = {
        "model": "deepseek-chat",
        "max_tokens": 20000,
        "messages": [{"role": "user", "content": "<command-name>/compact</command-name>"}],
    }
    raw = json.dumps(payload).encode("utf-8")

    req = httpx.Request(
        "POST",
        "https://example.com/v1/messages",
        headers={"content-type": "application/json"},
        content=raw,
    )

    # mutator should not return a response (it mutates and passes through)
    resp = hard_caps._hard_caps_mutator(req)
    assert resp is None

    mutated = json.loads(req.content.decode("utf-8"))
    assert policy.normalize_model(mutated["model"]) == "glm-4.7"
    assert mutated["max_tokens"] == 20000

    # content-length must match new body size
    assert int(req.headers["content-length"]) == len(req.content)


def test_httpx_mutator_skips_count_tokens_endpoints(monkeypatch):
    _, hard_caps = _reload_hard_caps(
        monkeypatch,
        LITELLM_EXT_COMPACT_ROUTE="1",
        LITELLM_COMPACT_MODEL="glm-4.7",
    )

    payload = {
        "model": "deepseek-chat",
        "max_tokens": 20000,
        "messages": [{"role": "user", "content": "<command-name>/compact</command-name>"}],
    }
    raw = json.dumps(payload).encode("utf-8")

    req = httpx.Request(
        "POST",
        "https://example.com/v1/messages/count_tokens",
        headers={"content-type": "application/json"},
        content=raw,
    )

    before = req.content
    resp = hard_caps._hard_caps_mutator(req)
    assert resp is None

    # Should not mutate
    assert req.content == before


def test_httpx_mutator_rewrites_alias_model_on_messages_using_litellm_config(monkeypatch, tmp_path):
    _, hard_caps = _reload_hard_caps(
        monkeypatch,
        LITELLM_EXT_COMPACT_ROUTE="1",
        LITELLM_COMPACT_MODEL="MiniMax-M2.5-ali",
    )

    cfg = tmp_path / "litellm.yaml"
    cfg.write_text(
        """
model_list:
  - model_name: MiniMax-M2.5-ali
    litellm_params:
      model: anthropic/MiniMax-M2.5
""".strip()
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("LITELLM_CONFIG", str(cfg))

    payload = {
        "model": "MiniMax-M2.5-ali",
        "max_tokens": 20000,
        "messages": [{"role": "user", "content": "hello"}],
    }
    req = httpx.Request(
        "POST",
        "https://coding.dashscope.aliyuncs.com/apps/anthropic/v1/messages",
        headers={"content-type": "application/json"},
        content=json.dumps(payload).encode("utf-8"),
    )

    resp = hard_caps._hard_caps_mutator(req)
    assert resp is None

    mutated = json.loads(req.content.decode("utf-8"))
    assert mutated["model"] == "MiniMax-M2.5"


def test_httpx_mutator_does_not_rewrite_alias_model_on_non_messages_path(monkeypatch, tmp_path):
    _, hard_caps = _reload_hard_caps(
        monkeypatch,
        LITELLM_EXT_COMPACT_ROUTE="1",
        LITELLM_COMPACT_MODEL="MiniMax-M2.5-ali",
    )

    cfg = tmp_path / "litellm.yaml"
    cfg.write_text(
        """
model_list:
  - model_name: MiniMax-M2.5-ali
    litellm_params:
      model: anthropic/MiniMax-M2.5
""".strip()
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("LITELLM_CONFIG", str(cfg))

    payload = {
        "model": "MiniMax-M2.5-ali",
        "max_tokens": 20000,
        "messages": [{"role": "user", "content": "hello"}],
    }
    req = httpx.Request(
        "POST",
        "https://example.com/v1/chat/completions",
        headers={"content-type": "application/json"},
        content=json.dumps(payload).encode("utf-8"),
    )

    resp = hard_caps._hard_caps_mutator(req)
    assert resp is None

    mutated = json.loads(req.content.decode("utf-8"))
    assert mutated["model"] == "MiniMax-M2.5-ali"
