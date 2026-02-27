from __future__ import annotations

"""ASGI middleware to enforce hard caps on /v1/messages requests."""

import json
import sys
from fnmatch import fnmatch
from typing import Any, Optional, Sequence

from ...core.config import get_list
from ...core.logging import format_log_line
from ...core.model_alias import format_model_log_fields
from ...core.patch import PatchSettings
from .proxy_patch_registry import register as register_proxy_patch, install as install_proxy_registry
from ...policy import enforce, get_requested_output, match_limits, normalize_model
from ...policy.text import looks_like_compact
from .utils import make_replay_receive, parse_json, read_body_with_limit, suffix_matches, update_content_length


TARGET_MOD = "litellm.proxy.proxy_server"

SETTINGS = PatchSettings(
    "hard_caps_messages",
    enabled_default=True,
    debug_default=False,
    enabled_envs=("LITELLM_EXT_HARD_CAPS_MESSAGES",),
    debug_envs=("LITELLM_EXT_HARD_CAPS_MESSAGES_DEBUG",),
)
_LOG = SETTINGS.logger("litellm_ext.hard_caps_messages")

def _log_always(msg: str) -> None:
    print(format_log_line("litellm_ext.hard_caps_messages", msg), file=sys.stderr, flush=True)

_PATCH_FLAG = "_litellm_ext_hard_caps_messages_applied"

_MODEL_PATTERNS = get_list(
    "extensions", "hard_caps_messages", "model_patterns", default=["*"]
)
_PATH_SUFFIXES = tuple(
    get_list(
        "extensions",
        "hard_caps_messages",
        "path_suffixes",
        default=["/v1/messages", "/anthropic/v1/messages"],
    )
)


def _model_matches(model: Optional[str]) -> bool:
    if not _MODEL_PATTERNS:
        return True
    if not model:
        return False
    mn = normalize_model(model)
    for pat in _MODEL_PATTERNS:
        try:
            if fnmatch(mn, pat) or fnmatch(str(model), pat):
                return True
        except Exception:
            continue
    return False


def _host_from_scope(scope: dict[str, Any]) -> str:
    headers = scope.get("headers")
    if not isinstance(headers, (list, tuple)):
        return ""
    for h in headers:
        if not isinstance(h, (tuple, list)) or len(h) != 2:
            continue
        key, val = h
        try:
            if bytes(key).lower() != b"host":
                continue
            host = bytes(val).decode("latin1", errors="ignore").strip()
        except Exception:
            continue
        if not host:
            return ""
        if ":" in host:
            host = host.split(":", 1)[0]
        return host.lower()
    return ""


def _model_log_fields(model: Any, host: str) -> str:
    return format_model_log_fields(model, host=host)


def _patch_proxy_app(module) -> None:
    if getattr(module, _PATCH_FLAG, False):
        return

    app = getattr(module, "app", None)
    if app is None or not hasattr(app, "add_middleware"):
        _LOG.debug(f"proxy_server imported but no app found; skipping module={TARGET_MOD}")
        setattr(module, _PATCH_FLAG, True)
        return

    if getattr(app, "_litellm_ext_hard_caps_messages", False):
        setattr(module, _PATCH_FLAG, True)
        return

    class _HardCapsMessagesMiddleware:
        def __init__(self, app_):
            self.app = app_

        async def __call__(self, scope, receive, send):
            if scope.get("type") != "http":
                return await self.app(scope, receive, send)

            path = (scope.get("path") or "").rstrip("/")
            if not suffix_matches(path, _PATH_SUFFIXES):
                return await self.app(scope, receive, send)

            body_msgs, body, truncated = await read_body_with_limit(receive)
            replay_receive = make_replay_receive(body_msgs, receive)
            if truncated:
                _LOG.debug(f"skipping hard_caps_messages: request body too large path={path!r}")
                return await self.app(scope, replay_receive, send)

            payload = parse_json(body)
            model = payload.get("model") if isinstance(payload, dict) else None
            host = _host_from_scope(scope)
            model_txt = _model_log_fields(model, host)
            _LOG.debug(f"request path={path!r} {model_txt} suffix_match=True")

            if payload is None or not isinstance(payload, dict):
                return await self.app(scope, replay_receive, send)

            if not _model_matches(model):
                _LOG.debug(f"skipping hard_caps_messages: {model_txt} does not match patterns")
                return await self.app(scope, replay_receive, send)

            compact_detected = looks_like_compact(payload)
            before_out = get_requested_output(payload)
            changed = False
            try:
                changed = enforce(payload, allow_raise=False)
            except Exception as e:
                _LOG.debug(f"hard_caps_messages enforce failed: {type(e).__name__}: {e}")
                return await self.app(scope, replay_receive, send)

            if compact_detected:
                routed_model = payload.get("model")
                if normalize_model(model) != normalize_model(routed_model):
                    _LOG.debug(f"route /compact {model!r} -> {routed_model!r}")
                else:
                    _LOG.debug(f"route /compact {model!r} -> {routed_model!r} (already target)")

            if not changed:
                limit_out = None
                try:
                    limit_out = match_limits(normalize_model(model or "")).get("max_output")
                except Exception:
                    limit_out = None
                if before_out is None:
                    _LOG.debug(
                        f"hard_caps_messages skipped {model_txt} max_tokens missing limit={limit_out}"
                    )
                else:
                    _LOG.debug(
                        f"hard_caps_messages unchanged {model_txt} max_tokens={before_out} limit={limit_out}"
                    )
                return await self.app(scope, replay_receive, send)

            after_out = get_requested_output(payload)
            _LOG.debug(
                f"hard_caps_messages enforced {model_txt} max_tokens {before_out}->{after_out}"
            )

            new_body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
            scope2 = update_content_length(scope, len(new_body))
            replay_new_body = make_replay_receive(
                [{"type": "http.request", "body": new_body, "more_body": False}],
                receive,
            )

            return await self.app(
                scope2,
                replay_new_body,
                send,
            )

    app.add_middleware(_HardCapsMessagesMiddleware)  # type: ignore
    setattr(app, "_litellm_ext_hard_caps_messages", True)
    setattr(module, _PATCH_FLAG, True)
    _LOG.debug("installed hard_caps_messages middleware")
    _log_always("installed hard_caps_messages middleware")


def install() -> None:
    if not SETTINGS.is_enabled():
        _LOG.debug("disabled (hard_caps_messages patch not enabled)")
        return

    register_proxy_patch("hard_caps_messages", _patch_proxy_app, order=35)
    install_proxy_registry()
