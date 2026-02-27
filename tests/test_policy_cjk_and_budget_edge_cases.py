
from __future__ import annotations

import os
import sys
from pathlib import Path

import yaml


def _reset_patches():
    for name in list(sys.modules.keys()):
        if name.startswith("litellm_ext"):
            sys.modules.pop(name, None)


def _write_cfg(text: str) -> Path:
    p = Path(os.environ["LITELLM_EXT_CONFIG_PATH"])
    p.write_text(text, encoding="utf-8")
    _reset_patches()
    return p


def test_cjk_heuristic_is_more_conservative_than_english():
    _write_cfg(
        """version: 1
policy:
    enabled: true
    safety_buffer_tokens: 1024
    context_estimate_multiplier: 1.0
    overflow_policy: reduce_output
    model_limits:
      "*":
        max_output: 2000
        max_context: 8000
"""
    )

    from litellm_ext.policy import estimate_input_tokens_heuristic

    cjk_text = "汉" * 9000
    eng_text = "a" * 9000

    cjk_est = estimate_input_tokens_heuristic({"messages": [{"role": "user", "content": cjk_text}]})
    eng_est = estimate_input_tokens_heuristic({"messages": [{"role": "user", "content": eng_text}]})

    assert cjk_est > eng_est


def test_enforce_reduces_output_when_cjk_input_would_overflow_context():
    _write_cfg(
        """version: 1
policy:
    enabled: true
    safety_buffer_tokens: 1024
    context_estimate_multiplier: 1.0
    overflow_policy: reduce_output
    model_limits:
      glm-4.7:
        max_output: 2000
        max_context: 8000
"""
    )

    from litellm_ext.policy import enforce

    payload = {
        "model": "glm-4.7",
        "max_tokens": 2000,
        "messages": [{"role": "user", "content": "汉" * 9000}],
    }

    changed = enforce(payload, allow_raise=True)
    assert changed is True
    assert int(payload["max_tokens"]) < 2000
    assert int(payload["max_tokens"]) >= 1
