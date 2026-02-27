from __future__ import annotations


def test_best_effort_does_not_double_apply_context_multiplier():
    """Regression test.

    When LiteLLM's `token_counter` is unavailable (common in these patch bundles),
    `estimate_input_tokens_best_effort()` falls back to the heuristic.

    The heuristic already applies CONTEXT_ESTIMATE_MULTIPLIER internally, so the
    best-effort wrapper must *not* apply it again.
    """

    # Import fresh (tests reset modules between runs)
    import litellm_ext.policy as pol

    payload = {
        "model": "glm-4.7",
        "messages": [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
            {"role": "user", "content": "say hello"},
        ],
    }

    h = pol.estimate_input_tokens_heuristic(payload)
    b = pol.estimate_input_tokens_best_effort("glm-4.7", payload)

    assert b is not None
    assert b == h
