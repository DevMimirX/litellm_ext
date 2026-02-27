from __future__ import annotations

import math
from fnmatch import fnmatch
from typing import Any, Dict, List, Optional, Tuple

from ..adapters.anthropic_openai import extract_text_blocks
from .settings import (
    CONTEXT_ESTIMATE_MULTIPLIER,
    COMPACT_MODEL,
    MODEL_MAPPING_DEFAULT_MODEL,
    MODEL_MAPPING_ENABLED,
    MODEL_MAPPING_FAMILY,
    MODEL_MAPPING_MATCH_SUBSTRINGS,
    MODEL_MAPPING_PATTERNS,
    MODEL_MAPPING_REASONING_MODEL,
    MAX_TRIM_STEPS,
    MIN_TAIL_MESSAGES,
    MODEL_LIMITS,
    OUTPUT_KEYS,
    OVERFLOW_POLICY,
    SAFETY_BUFFER_TOKENS,
    SKIP_PATH_SUFFIXES,
    dbg,
    normalize_model,
    tool_sanitizer_settings,
)
from .text import looks_like_compact, normalize_messages, normalize_system


# -----------------------------
# Helpers
# -----------------------------

def safe_int(x: Any) -> Optional[int]:
    try:
        if x is None:
            return None
        return int(x)
    except Exception:
        return None


def match_limits(model_norm: str) -> Dict[str, Optional[int]]:
    model_norm = (model_norm or "").strip().lower()
    if model_norm in MODEL_LIMITS:
        return MODEL_LIMITS[model_norm]

    for pat, lim in MODEL_LIMITS.items():
        if pat in ("*", model_norm):
            continue
        if fnmatch(model_norm, pat):
            return lim

    return MODEL_LIMITS.get("*", {"max_output": None, "max_context": None})


def get_requested_output(d: Dict[str, Any]) -> Optional[int]:
    for k in OUTPUT_KEYS:
        if k in d:
            v = safe_int(d.get(k))
            if v is not None:
                return v
    return None


def set_output(d: Dict[str, Any], val: int) -> None:
    v = int(max(1, val))
    d["max_tokens"] = v
    for k in ("max_output_tokens", "max_completion_tokens"):
        if k in d:
            d[k] = v


def _matches_model_mapping_patterns(model: str) -> bool:
    if not MODEL_MAPPING_PATTERNS:
        return True
    m_norm = normalize_model(model)
    for pat in MODEL_MAPPING_PATTERNS:
        try:
            if fnmatch(m_norm, pat) or fnmatch(model, pat):
                return True
        except Exception:
            continue
    return False


def has_thinking_enabled(payload: Dict[str, Any]) -> bool:
    thinking = payload.get("thinking")
    if isinstance(thinking, dict):
        t = thinking.get("type")
        if isinstance(t, str):
            return t.strip().lower() == "enabled"
        enabled = thinking.get("enabled")
        if isinstance(enabled, bool):
            return enabled
        if isinstance(enabled, str):
            return enabled.strip().lower() in {"1", "true", "yes", "y", "on", "enabled"}
        return False
    if isinstance(thinking, bool):
        return thinking
    if isinstance(thinking, str):
        return thinking.strip().lower() in {"1", "true", "yes", "y", "on", "enabled"}
    return False


def apply_model_mapping(payload: Dict[str, Any]) -> bool:
    if not MODEL_MAPPING_ENABLED:
        return False
    model = payload.get("model")
    if not model:
        return False
    model_str = str(model)
    if not _matches_model_mapping_patterns(model_str):
        return False

    mapped: Optional[str] = None
    if MODEL_MAPPING_REASONING_MODEL and has_thinking_enabled(payload):
        mapped = MODEL_MAPPING_REASONING_MODEL
    else:
        lower = model_str.lower()
        if MODEL_MAPPING_MATCH_SUBSTRINGS:
            if "haiku" in lower and MODEL_MAPPING_FAMILY.get("haiku"):
                mapped = MODEL_MAPPING_FAMILY.get("haiku")
            elif "opus" in lower and MODEL_MAPPING_FAMILY.get("opus"):
                mapped = MODEL_MAPPING_FAMILY.get("opus")
            elif "sonnet" in lower and MODEL_MAPPING_FAMILY.get("sonnet"):
                mapped = MODEL_MAPPING_FAMILY.get("sonnet")
        if not mapped and MODEL_MAPPING_DEFAULT_MODEL:
            mapped = MODEL_MAPPING_DEFAULT_MODEL

    if mapped and mapped != model_str:
        payload["model"] = mapped
        dbg(f"model mapping {model_str!r} -> {mapped!r}")
        return True
    return False


def _sanitize_tool_results_in_messages(messages: Any) -> bool:
    enabled, on_invalid = tool_sanitizer_settings()
    if not enabled:
        return False

    if not isinstance(messages, list) or not messages:
        return False

    def _content_to_blocks(msg: Dict[str, Any]) -> Tuple[List[Any], str]:
        c = msg.get("content")
        if isinstance(c, list):
            return c, "list"
        if isinstance(c, dict):
            return [c], "dict"
        return [], "other"

    def _set_blocks(msg: Dict[str, Any], blocks: List[Any], kind: str) -> None:
        if kind == "dict":
            if len(blocks) == 1 and isinstance(blocks[0], dict):
                msg["content"] = blocks[0]
            else:
                msg["content"] = blocks
            return
        if kind == "list":
            msg["content"] = blocks
            return

    def _is_tool_use_block(b: Any) -> Optional[str]:
        if not isinstance(b, dict):
            return None
        if b.get("type") != "tool_use":
            return None
        tid = b.get("id")
        if not isinstance(tid, str) or not tid.strip():
            tid = b.get("tool_use_id")
        if isinstance(tid, str) and tid.strip():
            return tid.strip()
        return None

    def _is_tool_result_block(b: Any) -> Optional[str]:
        if not isinstance(b, dict):
            return None
        if b.get("type") != "tool_result":
            return None
        tid = b.get("tool_use_id")
        if not isinstance(tid, str) or not tid.strip():
            tid = b.get("id")
        if isinstance(tid, str) and tid.strip():
            return tid.strip()
        return None

    def _tool_result_to_text_block(b: Dict[str, Any]) -> Dict[str, Any]:
        tid = str(b.get("tool_use_id") or b.get("id") or "").strip()
        content = b.get("content", "")
        if isinstance(content, str):
            body = content
        elif isinstance(content, list):
            body = extract_text_blocks(content)
        elif isinstance(content, dict):
            body = extract_text_blocks([content])
        else:
            body = str(content)
        body = body.strip()
        prefix = f"[tool_result {tid}]".strip()
        txt = f"{prefix} {body}".strip() if body else prefix
        return {"type": "text", "text": txt}

    changed = False

    for i in range(len(messages)):
        msg = messages[i]
        if not isinstance(msg, dict):
            continue

        blocks, kind = _content_to_blocks(msg)
        if not blocks:
            continue

        if not any(isinstance(b, dict) and b.get("type") == "tool_result" for b in blocks):
            continue

        prev_tool_use_ids: set[str] = set()
        if i > 0 and isinstance(messages[i - 1], dict):
            prev_blocks, _ = _content_to_blocks(messages[i - 1])
            for pb in prev_blocks:
                tid = _is_tool_use_block(pb)
                if tid:
                    prev_tool_use_ids.add(tid)

        new_blocks: List[Any] = []
        invalid = 0
        kept = 0

        for b in blocks:
            tid = _is_tool_result_block(b)
            if not tid:
                new_blocks.append(b)
                continue

            if tid in prev_tool_use_ids:
                new_blocks.append(b)
                kept += 1
                continue

            invalid += 1
            if on_invalid == "drop":
                continue
            new_blocks.append(_tool_result_to_text_block(b))

        if invalid:
            changed = True
            dbg(
                f"tool_sanitizer: msg[{i}] invalid_tool_results={invalid} kept={kept} "
                f"mode={on_invalid} prev_tool_use_ids={len(prev_tool_use_ids)}"
            )

            if kind in ("list", "dict") and not new_blocks:
                new_blocks = [{"type": "text", "text": ""}]

            _set_blocks(msg, new_blocks, kind)

    return changed


# -----------------------------
# Token estimation
# -----------------------------

def _estimate_chars_per_token(text: str) -> float:
    if not text:
        return 3.2
    n = len(text)
    cjk = 0
    for ch in text:
        o = ord(ch)
        if (0x4E00 <= o <= 0x9FFF) or (0x3040 <= o <= 0x30FF) or (0xAC00 <= o <= 0xD7AF):
            cjk += 1
    ratio = cjk / n if n else 0.0
    if ratio >= 0.50:
        return 1.3
    if ratio >= 0.20:
        return 1.8
    return 3.2


def estimate_input_tokens_heuristic(data: Dict[str, Any]) -> int:
    system_txt = normalize_system(data.get("system"))
    msgs = data.get("messages")
    combined = "\n".join(x for x in (system_txt, normalize_messages(msgs)) if x)
    msg_count = len(msgs) if isinstance(msgs, list) else 0

    cpt = _estimate_chars_per_token(combined)
    base = int(math.ceil(len(combined) / cpt)) if combined else 0
    overhead = (8 * msg_count) + 16
    est = base + overhead

    mult = float(CONTEXT_ESTIMATE_MULTIPLIER or 1.0)
    est = int(math.ceil(est * mult))

    return max(1, est)


def estimate_input_tokens_best_effort(model_for_counter: str, data: Dict[str, Any]) -> Optional[int]:
    system_txt = normalize_system(data.get("system"))
    msgs = data.get("messages")

    try:
        import litellm  # lazy import

        total = 0
        if isinstance(msgs, list):
            total += int(litellm.token_counter(model=model_for_counter, messages=msgs))
        if system_txt:
            total += int(litellm.token_counter(model=model_for_counter, text=system_txt))
        if total > 0:
            return int(math.ceil(total * CONTEXT_ESTIMATE_MULTIPLIER))
    except Exception as e:
        dbg(f"token_counter failed model={model_for_counter!r}: {type(e).__name__}: {e}")

    try:
        est = estimate_input_tokens_heuristic(data)
        return int(est)
    except Exception:
        return None


def trim_oldest_messages_to_budget(
    model_for_counter: str,
    data: Dict[str, Any],
    budget_in: int,
) -> Tuple[bool, Optional[int]]:
    messages = data.get("messages")
    if not isinstance(messages, list) or not messages:
        return False, None

    # Work with a copy to avoid mutating original list
    current_messages = list(messages)
    changed = False
    steps = 0
    min_keep = min(MIN_TAIL_MESSAGES, len(current_messages))

    while len(current_messages) > min_keep and steps < MAX_TRIM_STEPS:
        # Create temp data with current messages for estimation
        temp_data = dict(data)
        temp_data["messages"] = current_messages

        n = estimate_input_tokens_best_effort(model_for_counter, temp_data)
        if n is not None and n <= budget_in:
            break

        # Create new list without first message (immutable pattern)
        current_messages = current_messages[1:]
        changed = True

        # Sanitize tool results on the new list
        try:
            if _sanitize_tool_results_in_messages(current_messages):
                changed = True
        except Exception:
            pass
        steps += 1

    if changed:
        data["messages"] = current_messages
        return changed, estimate_input_tokens_best_effort(model_for_counter, data)

    return changed, estimate_input_tokens_best_effort(model_for_counter, data)


# -----------------------------
# Enforcement
# -----------------------------

def enforce(payload: Dict[str, Any], *, allow_raise: bool = True) -> bool:
    changed_any = False
    compact_routed = False

    if looks_like_compact(payload):
        old = payload.get("model")
        target = COMPACT_MODEL
        if target:
            old_norm = normalize_model(old)
            target_norm = normalize_model(target)
            if old_norm != target_norm:
                payload["model"] = target
                changed_any = True
                dbg(f"route /compact {old!r} -> {target!r}")
            else:
                dbg(f"route /compact {old!r} -> {target!r} (already target)")
            compact_routed = True
        else:
            dbg("route /compact ignored (target model is empty)")

    if not compact_routed:
        if apply_model_mapping(payload):
            changed_any = True

    try:
        if _sanitize_tool_results_in_messages(payload.get("messages")):
            changed_any = True
    except Exception as e:
        dbg(f"tool_result sanitization failed: {type(e).__name__}: {e}")

    model_raw = payload.get("model")
    model_norm = normalize_model(model_raw)
    lim = match_limits(model_norm)

    max_output = safe_int(lim.get("max_output"))
    max_context = safe_int(lim.get("max_context"))

    requested = get_requested_output(payload)
    if max_output and max_output > 0:
        if requested is None:
            set_output(payload, max_output)
            changed_any = True
            dbg(f"set max_tokens={max_output} model={model_norm} (missing)")
        elif requested > max_output:
            set_output(payload, max_output)
            changed_any = True
            dbg(f"clamp max_tokens {requested}->{max_output} model={model_norm}")

    out_now = get_requested_output(payload) or (max_output or 0)
    if out_now <= 0:
        return changed_any

    if not max_context or max_context <= 0:
        return changed_any

    model_for_counter = str(model_raw or model_norm or "")
    in_tokens = estimate_input_tokens_best_effort(model_for_counter, payload)
    if in_tokens is None:
        dbg(f"cannot count input tokens model={model_norm}; skip context enforcement")
        return changed_any

    def _fail(msg: str) -> bool:
        if allow_raise:
            raise ValueError(msg)
        dbg(f"(noraise) {msg}")
        return changed_any

    input_only_limit = max(1, max_context - SAFETY_BUFFER_TOKENS - 1)
    if in_tokens > input_only_limit:
        if OVERFLOW_POLICY in ("trim_messages", "reduce_then_trim"):
            ch, n2 = trim_oldest_messages_to_budget(model_for_counter, payload, input_only_limit)
            if ch:
                changed_any = True
            if ch and n2 is not None and n2 <= input_only_limit:
                dbg(f"trim input-only overflow model={model_norm} input {in_tokens}->{n2}")
                return True
        return _fail(
            f"Input too large for model={model_norm}: input≈{in_tokens} exceeds allowed≈{input_only_limit} "
            f"(ctx={max_context}, buf={SAFETY_BUFFER_TOKENS})."
        )

    budget_in = max(1, max_context - out_now - SAFETY_BUFFER_TOKENS)
    if in_tokens <= budget_in:
        return changed_any

    dbg(
        f"context overflow model={model_norm} input≈{in_tokens} budget_in≈{budget_in} "
        f"(ctx={max_context} out={out_now} buf={SAFETY_BUFFER_TOKENS}) policy={OVERFLOW_POLICY}"
    )

    if OVERFLOW_POLICY in ("reduce_output", "reduce_then_trim"):
        new_out = max(1, max_context - in_tokens - SAFETY_BUFFER_TOKENS)
        if max_output and max_output > 0:
            new_out = min(new_out, max_output)
        if new_out < out_now:
            set_output(payload, new_out)
            changed_any = True
            dbg(f"reduced max_tokens {out_now}->{new_out} model={model_norm}")
            out_now = new_out
            budget_in = max(1, max_context - out_now - SAFETY_BUFFER_TOKENS)
            if in_tokens <= budget_in:
                return True

    if OVERFLOW_POLICY in ("trim_messages", "reduce_then_trim"):
        ch, n2 = trim_oldest_messages_to_budget(model_for_counter, payload, budget_in)
        if ch:
            changed_any = True
        if ch and n2 is not None and n2 <= budget_in:
            dbg(f"trimmed messages model={model_norm} input {in_tokens}->{n2}")
            return True

    return _fail(
        f"Context overflow for model={model_norm}: input≈{in_tokens} exceeds budget_in≈{budget_in} "
        f"(ctx={max_context}, out={out_now}, buf={SAFETY_BUFFER_TOKENS})."
    )


_MUTATED_KEYS = ("model", "max_tokens", "max_output_tokens", "max_completion_tokens", "messages", "system")


def enforce_call(
    args: Tuple[Any, ...],
    kwargs: Dict[str, Any],
    *,
    allow_raise: bool,
) -> Tuple[Tuple[Any, ...], Dict[str, Any]]:
    if "model" in kwargs or not args:
        enforce(kwargs, allow_raise=allow_raise)
        return args, kwargs

    tmp = dict(kwargs)
    tmp["model"] = args[0]
    old_model = tmp.get("model")

    enforce(tmp, allow_raise=allow_raise)

    new_args = args
    new_model = tmp.get("model")
    if new_model is not None and new_model != old_model:
        new_args = (new_model,) + args[1:]

    for k in _MUTATED_KEYS:
        if k == "model":
            continue
        if k in tmp:
            kwargs[k] = tmp[k]

    return new_args, kwargs


def enforce_from_args_kwargs(args: Tuple[Any, ...], kwargs: Dict[str, Any], *, allow_raise: bool) -> Tuple[Tuple[Any, ...], Dict[str, Any]]:
    return enforce_call(args, kwargs, allow_raise=allow_raise)
