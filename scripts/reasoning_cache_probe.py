#!/usr/bin/env python3
"""Manual probe for reasoning_content capture + replay.

This script runs in-process (no proxy) so it can directly inspect the
litellm_ext.extensions.httpx.reasoning_replay cache. It validates:
  1) A real upstream response includes reasoning_content for tool calls.
  2) The reasoning_replay cache is populated.
  3) A follow-up request without reasoning_content gets it injected.

Usage examples:
  export MOONSHOT_API_KEY=...  # required for direct mode
  python scripts/reasoning_cache_probe.py --direct

  # If you need to force thinking mode, pass extra_body JSON:
  python scripts/reasoning_cache_probe.py --direct --extra-body '{"thinking": {"type": "enabled"}}'

Notes:
- This does not test the LiteLLM proxy path. For proxy verification, enable
  reasoning_replay debug logs and look for:
    "stored reasoning for ..." and "cache_hit=True".
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, Optional

import httpx

import litellm_ext.extensions.httpx.reasoning_replay as rr
from litellm_ext.core.registry import install_httpx_patch


def _parse_json(s: Optional[str]) -> Optional[Dict[str, Any]]:
    if not s:
        return None
    try:
        v = json.loads(s)
        return v if isinstance(v, dict) else None
    except Exception:
        return None


def _build_tool() -> Dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": "echo",
            "description": "Echo text",
            "parameters": {
                "type": "object",
                "properties": {"text": {"type": "string"}},
                "required": ["text"],
            },
        },
    }


def _prompt_message() -> str:
    return "You MUST call the echo tool once with {\"text\":\"hi\"}."


def _direct_call(base_url: str, model: str, extra_body: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    tools = [_build_tool()]
    payload: Dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": _prompt_message()}],
        "tools": tools,
        "tool_choice": "auto",
        "temperature": 1,
        "stream": False,
        "max_tokens": 200,
    }
    if extra_body:
        payload["extra_body"] = extra_body

    headers = {
        "Authorization": f"Bearer {os.environ.get('MOONSHOT_API_KEY', '')}",
        "Content-Type": "application/json",
    }

    with httpx.Client(timeout=120) as client:
        r = client.post(base_url, json=payload, headers=headers)
        r.raise_for_status()
        return r.json()


def main() -> int:
    parser = argparse.ArgumentParser(description="Probe reasoning_content caching.")
    parser.add_argument("--direct", action="store_true", help="Call Moonshot API directly (requires MOONSHOT_API_KEY).")
    parser.add_argument("--model", default="kimi-k2.5")
    parser.add_argument("--base-url", default="https://api.moonshot.cn/v1/chat/completions")
    parser.add_argument("--extra-body", default=None, help="JSON string for extra_body.")
    args = parser.parse_args()

    if not args.direct:
        print("This probe currently supports direct mode only. Use --direct.")
        return 2

    if not os.environ.get("MOONSHOT_API_KEY"):
        print("MOONSHOT_API_KEY is required for --direct.")
        return 2

    extra_body = _parse_json(args.extra_body)

    # Ensure httpx send wrappers + reasoning replay patch are installed in-process.
    install_httpx_patch()
    rr.install()

    rr._CACHE.clear()

    print("Calling model...")
    obj = _direct_call(args.base_url, args.model, extra_body)

    msg = obj.get("choices", [{}])[0].get("message", {})
    tool_calls = msg.get("tool_calls") or []
    reasoning = msg.get("reasoning_content")

    print(f"tool_calls={len(tool_calls)} reasoning_present={isinstance(reasoning, str)}")
    if isinstance(reasoning, str):
        print(f"reasoning_len={len(reasoning)}")

    if not tool_calls:
        print("No tool_calls in response; cannot test caching.")
        return 1

    tcid = tool_calls[0].get("id")
    fn_name = (tool_calls[0].get("function") or {}).get("name")
    fallback_key = f"{fn_name}:0" if fn_name else "tool:0"

    # Verify cache populated by response mutator
    cached = rr._cache_get([tcid, fallback_key]) if tcid else rr._cache_get([fallback_key])
    print(f"cache_hit={cached is not None} cached_len={len(cached) if cached else 0}")

    # Build a follow-up request and verify injection
    followup = {
        "model": args.model,
        "messages": [
            {"role": "user", "content": _prompt_message()},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": tool_calls,
                # reasoning_content intentionally omitted
            },
            {"role": "tool", "tool_call_id": tcid, "content": "hi"},
        ],
        "tools": [_build_tool()],
        "tool_choice": "auto",
        "temperature": 1,
        "stream": False,
        "max_tokens": 200,
    }

    req = httpx.Request("POST", args.base_url, json=followup)
    rr._httpx_mutator(req)
    payload_after = json.loads(req.read())
    injected = payload_after["messages"][1].get("reasoning_content")
    print(f"injected_present={injected is not None} injected_len={len(injected) if isinstance(injected, str) else 0}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
