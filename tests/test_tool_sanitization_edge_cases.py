from __future__ import annotations

from litellm_ext.policy import enforce


def test_tool_result_in_first_message_is_sanitized_strict_adjacency_convert_to_text():
    payload = {
        "model": "deepseek/deepseek-reasoner",
        "max_tokens": 10,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "call_1", "content": "ok"},
                    {"type": "text", "text": "hi"},
                ],
            }
        ],
    }
    enforce(payload, allow_raise=False)
    blocks = payload["messages"][0]["content"]
    assert all(b.get("type") != "tool_result" for b in blocks)
    # We convert invalid tool_result blocks into text blocks instead of dropping by default.
    assert any(b.get("type") == "text" and "tool_result call_1" in (b.get("text") or "") for b in blocks)


def test_tool_result_without_prev_tool_use_is_sanitized_strict_adjacency_convert_to_text():
    payload = {
        "model": "deepseek/deepseek-reasoner",
        "max_tokens": 10,
        "messages": [
            {"role": "assistant", "content": [{"type": "text", "text": "no tool_use here"}]},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "call_2", "content": "oops"}]},
        ],
    }
    enforce(payload, allow_raise=False)
    blocks = payload["messages"][1]["content"]
    assert all(b.get("type") != "tool_result" for b in blocks)
    assert any(b.get("type") == "text" and "tool_result call_2" in (b.get("text") or "") for b in blocks)


def test_valid_tool_result_is_kept_strict_adjacency():
    payload = {
        "model": "deepseek/deepseek-reasoner",
        "max_tokens": 10,
        "messages": [
            {"role": "assistant", "content": [{"type": "tool_use", "id": "call_3", "name": "x", "input": {}}]},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "call_3", "content": "ok"}]},
        ],
    }
    enforce(payload, allow_raise=False)
    blocks = payload["messages"][1]["content"]
    assert any(b.get("type") == "tool_result" and b.get("tool_use_id") == "call_3" for b in blocks)


def test_tool_result_when_content_is_single_dict_is_sanitized():
    payload = {
        "model": "deepseek/deepseek-reasoner",
        "max_tokens": 10,
        "messages": [
            {"role": "assistant", "content": {"type": "text", "text": "no tool_use here"}},
            {"role": "user", "content": {"type": "tool_result", "tool_use_id": "call_9", "content": "oops"}},
        ],
    }
    enforce(payload, allow_raise=False)
    # after sanitization, content may be dict or list; accept both
    c = payload["messages"][1]["content"]
    if isinstance(c, dict):
        assert c.get("type") == "text"
        assert "tool_result call_9" in (c.get("text") or "")
    else:
        assert all(b.get("type") != "tool_result" for b in c)
        assert any("tool_result call_9" in (b.get("text") or "") for b in c if isinstance(b, dict))


def test_strict_adjacency_ignores_tool_use_two_messages_back():
    payload = {
        "model": "deepseek/deepseek-reasoner",
        "max_tokens": 10,
        "messages": [
            {"role": "assistant", "content": [{"type": "tool_use", "id": "call_old", "name": "x", "input": {}}]},
            {"role": "assistant", "content": [{"type": "text", "text": "intermediate"}]},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "call_old", "content": "oops"}]},
        ],
    }
    enforce(payload, allow_raise=False)
    blocks = payload["messages"][2]["content"]
    assert all(b.get("type") != "tool_result" for b in blocks)


def test_mixed_tool_results_only_keeps_ids_from_previous_message_and_converts_rest():
    payload = {
        "model": "deepseek/deepseek-reasoner",
        "max_tokens": 10,
        "messages": [
            {
                "role": "assistant",
                "content": [
                    {"type": "tool_use", "id": "call_ok", "name": "x", "input": {}},
                    {"type": "tool_use", "id": "call_ok2", "name": "y", "input": {}},
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "call_ok", "content": "good"},
                    {"type": "tool_result", "tool_use_id": "call_bad", "content": "bad"},
                    {"type": "text", "text": "tail"},
                ],
            },
        ],
    }
    enforce(payload, allow_raise=False)
    blocks = payload["messages"][1]["content"]
    assert any(b.get("type") == "tool_result" and b.get("tool_use_id") == "call_ok" for b in blocks)
    assert all(not (b.get("type") == "tool_result" and b.get("tool_use_id") == "call_bad") for b in blocks)
    assert any(b.get("type") == "text" and "tool_result call_bad" in (b.get("text") or "") for b in blocks)


def test_tool_use_block_can_use_tool_use_id_field():
    # Some providers/proxies may serialize tool_use blocks with `tool_use_id`.
    payload = {
        "model": "deepseek/deepseek-reasoner",
        "max_tokens": 10,
        "messages": [
            {"role": "assistant", "content": [{"type": "tool_use", "tool_use_id": "call_t", "name": "x", "input": {}}]},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "call_t", "content": "ok"}]},
        ],
    }
    enforce(payload, allow_raise=False)
    blocks = payload["messages"][1]["content"]
    assert any(b.get("type") == "tool_result" and b.get("tool_use_id") == "call_t" for b in blocks)


def test_tool_result_block_can_use_id_field_and_still_be_valid():
    payload = {
        "model": "deepseek/deepseek-reasoner",
        "max_tokens": 10,
        "messages": [
            {"role": "assistant", "content": [{"type": "tool_use", "id": "call_x", "name": "x", "input": {}}]},
            {"role": "user", "content": [{"type": "tool_result", "id": "call_x", "content": "ok"}]},
        ],
    }
    enforce(payload, allow_raise=False)
    blocks = payload["messages"][1]["content"]
    assert any(b.get("type") == "tool_result" and (b.get("tool_use_id") == "call_x" or b.get("id") == "call_x") for b in blocks)


def test_on_invalid_drop_mode_removes_invalid_tool_result(monkeypatch, tmp_path):
    # Force on_invalid=drop to ensure invalid tool_result blocks are removed (not converted to text).
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(
        """
policy:
    tool_sanitizer:
      enabled: true
      on_invalid: drop
""".lstrip(),
        encoding="utf-8",
    )
    monkeypatch.setenv("LITELLM_EXT_CONFIG_PATH", str(cfg_path))

    import litellm_ext.core.config as _cfg

    # Make sure config reloads from new path.
    # Force reload so this test can verify `on_invalid: drop` behavior.
    _cfg.load_config(force=True)

    payload = {
        "model": "deepseek/deepseek-reasoner",
        "max_tokens": 10,
        "messages": [
            {"role": "assistant", "content": [{"type": "text", "text": "no tool_use"}]},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "call_bad", "content": "bad"}]},
        ],
    }
    enforce(payload, allow_raise=False)
    blocks = payload["messages"][1]["content"]
    assert all(b.get("type") != "tool_result" for b in blocks)
    assert not any(b.get("type") == "text" and "tool_result" in (b.get("text") or "") for b in blocks)
