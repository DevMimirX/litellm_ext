from __future__ import annotations

import os
import re
import sys
from fnmatch import fnmatch
from typing import Any, Dict, List, Optional, Tuple

from ..core.config import get, get_bool, get_int, get_str
from ..core.logging import env_flag_any, format_log_line

# -----------------------------
# Debug
# -----------------------------
_DEBUG = env_flag_any(("LITELLM_EXT_POLICY_DEBUG",))
if _DEBUG is None:
    _DEBUG = get_bool("debug", "policy", default=False)


def dbg(msg: str) -> None:
    if _DEBUG:
        print(format_log_line("litellm_ext.policy", msg), file=sys.stderr, flush=True)


# -----------------------------
# Config namespace
# -----------------------------

def _pget(*keys, default=None):
    return get("policy", *keys, default=default)


def _pint(*keys, default=0):
    v = _pget(*keys, default=None)
    try:
        return int(v) if v is not None else default
    except Exception:
        return default


def _pstr(*keys, default=""):
    v = _pget(*keys, default=None)
    return str(v) if v is not None else default


def _pbool(*keys, default=False):
    v = _pget(*keys, default=None)
    if isinstance(v, bool):
        return v
    if isinstance(v, int):
        return bool(v)
    if isinstance(v, str):
        s = v.strip().lower()
        if s in {"1", "true", "yes", "y", "on"}:
            return True
        if s in {"0", "false", "no", "n", "off"}:
            return False
    return default


# -----------------------------
# POLICY (edit via YAML)
# -----------------------------
_DEFAULT_MODEL_LIMITS: Dict[str, Dict[str, Optional[int]]] = {
    "*": {"max_output": 16000, "max_context": None},
    "deepseek-chat": {"max_output": 8192, "max_context": 131072},
    "deepseek-reasoner": {"max_output": 16384, "max_context": 131072},
}

_RAW_MODEL_LIMITS = _pget("model_limits", default=None)
MODEL_LIMITS: Dict[str, Dict[str, Optional[int]]] = {}

if isinstance(_RAW_MODEL_LIMITS, dict) and _RAW_MODEL_LIMITS:
    for k, v in _RAW_MODEL_LIMITS.items():
        if not isinstance(k, str) or not isinstance(v, dict):
            continue
        key = k.strip().lower()
        mo = v.get("max_output")
        mc = v.get("max_context")
        try:
            mo_i = int(mo) if mo is not None else None
        except Exception:
            mo_i = None
        try:
            mc_i = int(mc) if mc is not None else None
        except Exception:
            mc_i = None
        MODEL_LIMITS[key] = {"max_output": mo_i, "max_context": mc_i}

if not MODEL_LIMITS:
    MODEL_LIMITS = _DEFAULT_MODEL_LIMITS

# safety buffer / trimming knobs
_env_buf = os.environ.get("LITELLM_EXT_SAFETY_BUFFER_TOKENS")
SAFETY_BUFFER_TOKENS = int(_env_buf) if (_env_buf and _env_buf.isdigit()) else _pint("safety_buffer_tokens", default=1024)

# Multiply token estimates to be conservative (helps avoid surprise context-limit 400s).
_env_mul = os.environ.get("LITELLM_EXT_CONTEXT_ESTIMATE_MULTIPLIER")
try:
    _cfg_mul = _pget("context_estimate_multiplier", default=1.0)
    CONTEXT_ESTIMATE_MULTIPLIER = float(_env_mul) if _env_mul is not None else float(_cfg_mul if _cfg_mul is not None else 1.0)
except Exception:
    CONTEXT_ESTIMATE_MULTIPLIER = 1.0
if CONTEXT_ESTIMATE_MULTIPLIER < 1.0:
    CONTEXT_ESTIMATE_MULTIPLIER = 1.0


_ALLOWED_OVERFLOW_POLICIES = {"reduce_output", "trim_messages", "reduce_then_trim"}
_env_overflow = os.environ.get("LITELLM_EXT_OVERFLOW_POLICY")
_OVERFLOW_POLICY_RAW = (_env_overflow.strip() if _env_overflow else _pstr("overflow_policy", default="reduce_then_trim")).strip()
OVERFLOW_POLICY = _OVERFLOW_POLICY_RAW if _OVERFLOW_POLICY_RAW in _ALLOWED_OVERFLOW_POLICIES else "reduce_then_trim"
if OVERFLOW_POLICY != _OVERFLOW_POLICY_RAW:
    dbg(f"invalid overflow_policy={_OVERFLOW_POLICY_RAW!r}; using {OVERFLOW_POLICY!r}")

_env_tail = os.environ.get("LITELLM_EXT_MIN_TAIL_MESSAGES")
MIN_TAIL_MESSAGES = int(_env_tail) if (_env_tail and _env_tail.isdigit()) else _pint("min_tail_messages", default=6)

_env_steps = os.environ.get("LITELLM_EXT_MAX_TRIM_STEPS")
MAX_TRIM_STEPS = int(_env_steps) if (_env_steps and _env_steps.isdigit()) else _pint("max_trim_steps", default=200)

# Used by httpx mutators to avoid mutating token-count endpoints.
_SKIP_SUFFIXES_RAW = _pget("skip_path_suffixes", default=None)
if isinstance(_SKIP_SUFFIXES_RAW, list) and all(isinstance(x, str) for x in _SKIP_SUFFIXES_RAW):
    SKIP_PATH_SUFFIXES: Tuple[str, ...] = tuple(_SKIP_SUFFIXES_RAW)
else:
    SKIP_PATH_SUFFIXES = (
        "/messages/count_tokens",
        "/v1/messages/count_tokens",
        "/count_tokens",
    )

OUTPUT_KEYS = ("max_tokens", "max_output_tokens", "max_completion_tokens")


# -----------------------------
# /compact routing (Claude Code)
# -----------------------------
_env_compact = os.environ.get("LITELLM_EXT_COMPACT_ROUTE")
COMPACT_ROUTE_ENABLED = (_env_compact != "0") if _env_compact is not None else _pbool("compact_routing", "enabled", default=True)

_env_compact_model = os.environ.get("LITELLM_COMPACT_MODEL")
COMPACT_MODEL = (_env_compact_model.strip() if _env_compact_model else _pstr("compact_routing", "target_model", default="glm-4.7")).strip()

_ALLOWED_COMPACT_ROUTE_MODES = {"explicit", "pattern"}
_env_compact_mode = (os.environ.get("LITELLM_EXT_COMPACT_ROUTE_MODE") or "").strip().lower()
_cfg_compact_mode = _pstr("compact_routing", "mode", default="explicit").strip().lower()
_COMPACT_ROUTE_MODE_RAW = _env_compact_mode or _cfg_compact_mode or "explicit"
COMPACT_ROUTE_MODE = (
    _COMPACT_ROUTE_MODE_RAW if _COMPACT_ROUTE_MODE_RAW in _ALLOWED_COMPACT_ROUTE_MODES else "explicit"
)
if COMPACT_ROUTE_MODE != _COMPACT_ROUTE_MODE_RAW:
    dbg(f"invalid compact_routing.mode={_COMPACT_ROUTE_MODE_RAW!r}; using {COMPACT_ROUTE_MODE!r}")

_DEFAULT_COMPACT_EXPLICIT_PATTERNS = [
    r"<command-name>\s*/compact\s*</command-name>",
    r"<command-message>\s*compact\s*</command-message>",
]

_RAW_COMPACT_EXPLICIT_PATTERNS = _pget("compact_routing", "explicit_patterns", default=None)
if (
    isinstance(_RAW_COMPACT_EXPLICIT_PATTERNS, list)
    and all(isinstance(x, str) for x in _RAW_COMPACT_EXPLICIT_PATTERNS)
    and _RAW_COMPACT_EXPLICIT_PATTERNS
):
    _COMPACT_EXPLICIT_PATTERNS_SRC = _RAW_COMPACT_EXPLICIT_PATTERNS
else:
    _COMPACT_EXPLICIT_PATTERNS_SRC = _DEFAULT_COMPACT_EXPLICIT_PATTERNS

_COMPACT_EXPLICIT_PATTERNS: List[re.Pattern[str]] = []
for pat in _COMPACT_EXPLICIT_PATTERNS_SRC:
    try:
        _COMPACT_EXPLICIT_PATTERNS.append(re.compile(pat, re.IGNORECASE))
    except re.error as e:
        dbg(f"invalid compact explicit regex {pat!r}: {e}")

_DEFAULT_COMPACT_PATTERNS = [
    r"<command-name>\s*/compact\s*</command-name>",
    r"<command-message>\s*compact\s*</command-message>",
    r"Your task is to create a detailed summary of the conversation so far",
]

_RAW_COMPACT_PATTERNS = _pget("compact_routing", "patterns", default=None)
if isinstance(_RAW_COMPACT_PATTERNS, list) and all(isinstance(x, str) for x in _RAW_COMPACT_PATTERNS) and _RAW_COMPACT_PATTERNS:
    _COMPACT_PATTERNS_SRC = _RAW_COMPACT_PATTERNS
else:
    _COMPACT_PATTERNS_SRC = _DEFAULT_COMPACT_PATTERNS

_COMPACT_PATTERNS: List[re.Pattern[str]] = []
for pat in _COMPACT_PATTERNS_SRC:
    try:
        _COMPACT_PATTERNS.append(re.compile(pat, re.IGNORECASE))
    except re.error as e:
        dbg(f"invalid compact regex {pat!r}: {e}")


# -----------------------------
# Auto-compact tuning (per-model)
# -----------------------------
_AUTOCOMPACT_ENABLED = _pbool("autocompact_tuning", "enabled", default=False)
_AUTOCOMPACT_ALLOW_UNDERREPORT = _pbool("autocompact_tuning", "allow_underreport", default=True)

_AUTOCOMPACT_DEFAULT_MULTIPLIER_RAW = _pget("autocompact_tuning", "default_multiplier", default=1.0)
try:
    _AUTOCOMPACT_DEFAULT_MULTIPLIER = float(_AUTOCOMPACT_DEFAULT_MULTIPLIER_RAW)
except Exception:
    _AUTOCOMPACT_DEFAULT_MULTIPLIER = 1.0

_AUTOCOMPACT_MULTIPLIERS_RAW = _pget("autocompact_tuning", "multipliers", default=None)
if not isinstance(_AUTOCOMPACT_MULTIPLIERS_RAW, dict):
    _AUTOCOMPACT_MULTIPLIERS_RAW = {}


def normalize_model(model: Any) -> str:
    s = str(model or "").strip().lower()
    if "/" in s:
        s = s.split("/")[-1].strip()
    return s


def autocompact_multiplier_for_model(model: Any) -> float:
    """Return the configured multiplier for the model (default 1.0)."""

    if not _AUTOCOMPACT_ENABLED:
        return 1.0

    m = normalize_model(model)

    # Exact match first
    raw = _AUTOCOMPACT_MULTIPLIERS_RAW.get(m)

    # Wildcard match next (fnmatch)
    if raw is None:
        for pat, val in _AUTOCOMPACT_MULTIPLIERS_RAW.items():
            if pat in ("*", m):
                continue
            try:
                if fnmatch(m, str(pat)):
                    raw = val
                    break
            except Exception:
                continue

    if raw is None:
        raw = _AUTOCOMPACT_MULTIPLIERS_RAW.get("*", _AUTOCOMPACT_DEFAULT_MULTIPLIER)

    try:
        mult = float(raw)
    except Exception:
        mult = 1.0

    if mult <= 0:
        mult = 1.0
    if not _AUTOCOMPACT_ALLOW_UNDERREPORT:
        mult = max(1.0, mult)

    return mult


# -----------------------------
# Model mapping (thinking-aware)
# -----------------------------
MODEL_MAPPING_ENABLED = _pbool("model_mapping", "enabled", default=False)
MODEL_MAPPING_MATCH_SUBSTRINGS = _pbool("model_mapping", "match_substrings", default=True)

_MODEL_MAPPING_PATTERNS_RAW = _pget("model_mapping", "apply_to_patterns", default=None)
if isinstance(_MODEL_MAPPING_PATTERNS_RAW, list) and all(isinstance(x, str) for x in _MODEL_MAPPING_PATTERNS_RAW):
    MODEL_MAPPING_PATTERNS: List[str] = [x for x in _MODEL_MAPPING_PATTERNS_RAW if x.strip()]
else:
    MODEL_MAPPING_PATTERNS = []

MODEL_MAPPING_DEFAULT_MODEL = _pstr("model_mapping", "default_model", default="").strip()
MODEL_MAPPING_REASONING_MODEL = _pstr("model_mapping", "reasoning_model", default="").strip()

_MODEL_MAPPING_FAMILY_RAW = _pget("model_mapping", "family_overrides", default=None)
if not isinstance(_MODEL_MAPPING_FAMILY_RAW, dict):
    _MODEL_MAPPING_FAMILY_RAW = {}

MODEL_MAPPING_FAMILY: Dict[str, str] = {}
for k, v in _MODEL_MAPPING_FAMILY_RAW.items():
    if not isinstance(k, str):
        continue
    if not isinstance(v, str):
        continue
    k_norm = k.strip().lower()
    v_norm = v.strip()
    if not k_norm or not v_norm:
        continue
    MODEL_MAPPING_FAMILY[k_norm] = v_norm


# Tool-result sanitizer settings
_ALLOWED_TOOL_SANITIZER_MODES = {"convert_to_text", "drop"}


def tool_sanitizer_settings() -> tuple[bool, str]:
    enabled = _pbool("tool_sanitizer", "enabled", default=True)
    raw = _pstr("tool_sanitizer", "on_invalid", default="convert_to_text").strip().lower()
    mode = raw if raw in _ALLOWED_TOOL_SANITIZER_MODES else "convert_to_text"
    if mode != raw:
        dbg(f"invalid policy.tool_sanitizer.on_invalid={raw!r}; using {mode!r}")
    return enabled, mode
