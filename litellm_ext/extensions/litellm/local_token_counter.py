from __future__ import annotations

import fnmatch
import os
import importlib.abc
import importlib.machinery
import sys
from threading import RLock
from typing import Any, Optional

from ...core.patch import PatchSettings
from ...core.config import get_bool, get_list
from ...core.model_alias import format_model_log_fields
from ...policy import estimate_input_tokens_heuristic

SETTINGS = PatchSettings(
    "local_token_counter",
    enabled_default=True,
    debug_default=False,
    enabled_envs=("LITELLM_EXT_LOCAL_TOKEN_COUNTER",),
    debug_envs=("LITELLM_EXT_LOCAL_TOKEN_COUNTER_DEBUG",),
)
_LOG = SETTINGS.logger("litellm_ext.local_token_counter")

_PATCH_ATTR = "_litellm_ext_local_token_counter_applied"
_ORIG_ATTR = "_litellm_ext_local_token_counter_orig"

_PROVIDERS = tuple((
    os.environ.get("LITELLM_EXT_LOCAL_TOKEN_COUNTER_PROVIDERS")
    or ""
).split(","))

_MODEL_PATTERNS = tuple((
    os.environ.get("LITELLM_EXT_LOCAL_TOKEN_COUNTER_PATTERNS")
    or ""
).split(","))

_HOST_SUBSTRS = tuple((
    os.environ.get("LITELLM_EXT_LOCAL_TOKEN_COUNTER_HOSTS")
    or ""
).split(","))

if not any(v.strip() for v in _PROVIDERS):
    _PROVIDERS = tuple(get_list("extensions", "local_token_counter", "providers", default=["anthropic"]))
if not any(v.strip() for v in _MODEL_PATTERNS):
    _MODEL_PATTERNS = tuple(get_list("extensions", "local_token_counter", "model_patterns", default=["claude-*", "anthropic-*"]))
if not any(v.strip() for v in _HOST_SUBSTRS):
    _HOST_SUBSTRS = tuple(get_list("extensions", "local_token_counter", "host_substrs", default=["api.anthropic.com"]))


def _normalize_list(values: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(v.strip().lower() for v in values if isinstance(v, str) and v.strip())


_PROVIDERS = _normalize_list(_PROVIDERS)
_MODEL_PATTERNS = tuple(v.strip() for v in _MODEL_PATTERNS if isinstance(v, str) and v.strip())
_HOST_SUBSTRS = _normalize_list(_HOST_SUBSTRS)



_FORCE_ANTHROPIC_COUNT_TOKENS = get_bool(
    "extensions", "local_token_counter", "force_anthropic_count_tokens", default=True
)

_COUNT_TOKENS_PATCH_ATTR = "_litellm_ext_local_token_counter_count_tokens"
_COUNT_TOKENS_ORIG_ATTR = "_litellm_ext_local_token_counter_count_tokens_orig"

_TARGET_HANDLER = "litellm.llms.anthropic.count_tokens.handler"
_FINDER_INSTALLED = False

_CALIBRATION_ALPHA_DOWN = 0.35
_CALIBRATION_ALPHA_UP = 0.15
_CALIBRATION_MIN = 0.25
_CALIBRATION_MAX = 4.0
_CALIBRATION_DEFAULT_BY_SCOPE: dict[str, float] = {
    "messages": 0.75,
    "messages+system": 0.72,
    "system": 0.72,
    "text": 0.65,
    "unknown": 0.75,
}
_CALIBRATION_FACTORS: dict[str, float] = {}
_CALIBRATION_LOCK = RLock()


def _matches_model(model: Optional[str]) -> bool:
    if not model:
        return False
    model_norm = str(model).strip().lower()
    for pat in _MODEL_PATTERNS:
        if fnmatch.fnmatch(model_norm, str(pat).strip().lower()):
            return True
    return False


def _matches_provider(kwargs: dict[str, Any]) -> bool:
    provider = kwargs.get("custom_llm_provider") or kwargs.get("provider")
    if isinstance(provider, str) and provider.strip().lower() in _PROVIDERS:
        return True
    return _matches_api_base(kwargs.get("api_base"))


def _matches_api_base(api_base: Any) -> bool:
    if isinstance(api_base, str):
        low = api_base.lower()
        if any(h in low for h in _HOST_SUBSTRS):
            return True
    return False


def _should_use_local_anthropic_count_tokens(model: Any, api_base: Any) -> bool:
    # Respect configured model filters for Anthropic /count_tokens calls.
    if not _matches_model(model if isinstance(model, str) else None):
        return False

    if "anthropic" in _PROVIDERS:
        return True

    return _matches_api_base(api_base)


def _normalize_model(model: Any) -> str:
    return str(model or "").strip().lower()


def _calibration_key(model: Any, scope: str) -> str:
    return f"{_normalize_model(model)}::{scope}"


def _default_calibration_factor(scope: str) -> float:
    s = str(scope or "unknown")
    base = _CALIBRATION_DEFAULT_BY_SCOPE.get(s, _CALIBRATION_DEFAULT_BY_SCOPE["unknown"])
    return float(max(_CALIBRATION_MIN, min(_CALIBRATION_MAX, base)))


def _build_estimation_payload(kwargs: dict[str, Any]) -> tuple[dict[str, Any], str]:
    data: dict[str, Any] = {}
    scope = "unknown"

    if isinstance(kwargs.get("messages"), list):
        data["messages"] = kwargs.get("messages")
        scope = "messages"
    if isinstance(kwargs.get("system"), (str, list, dict)):
        data["system"] = kwargs.get("system")
        scope = "messages+system" if "messages" in data else "system"
    if isinstance(kwargs.get("text"), str) and "messages" not in data and "system" not in data:
        # Treat freeform text as system-like content to avoid message wrapper overhead bias.
        data["system"] = kwargs.get("text")
        scope = "text"

    return data, scope


def _estimate_local_tokens(model: Optional[str], kwargs: dict[str, Any]) -> tuple[int, str]:
    data, scope = _build_estimation_payload(kwargs)
    try:
        est = estimate_input_tokens_heuristic(data)
        return (int(est) if est and est > 0 else 1), scope
    except Exception:
        return 1, scope


def _extract_positive_input_tokens(resp: Any) -> Optional[int]:
    if isinstance(resp, dict):
        val = resp.get("input_tokens")
        try:
            n = int(val) if val is not None else 0
        except Exception:
            return None
        return n if n > 0 else None
    return None


def _estimate_local_tokens_from_messages(messages: Any) -> int:
    data: dict[str, Any] = {}
    if isinstance(messages, list):
        data["messages"] = messages
    est = estimate_input_tokens_heuristic(data)
    if not est or est <= 0:
        est = 1
    return int(est)


def _get_calibration_factor(model: Any, scope: str) -> float:
    key = _calibration_key(model, scope)
    with _CALIBRATION_LOCK:
        return float(_CALIBRATION_FACTORS.get(key, _default_calibration_factor(scope)))


def _update_calibration_factor(model: Any, scope: str, local_raw: int, remote_tokens: int) -> float:
    if local_raw <= 0 or remote_tokens <= 0:
        return _get_calibration_factor(model, scope)

    observed = float(remote_tokens) / float(local_raw)
    observed = max(_CALIBRATION_MIN, min(_CALIBRATION_MAX, observed))
    key = _calibration_key(model, scope)
    with _CALIBRATION_LOCK:
        prev_raw = _CALIBRATION_FACTORS.get(key, None)
        if prev_raw is None:
            updated = observed
        else:
            prev = float(prev_raw)
            alpha = _CALIBRATION_ALPHA_DOWN if observed < prev else _CALIBRATION_ALPHA_UP
            updated = ((1.0 - alpha) * prev) + (alpha * observed)
        updated = max(_CALIBRATION_MIN, min(_CALIBRATION_MAX, updated))
        _CALIBRATION_FACTORS[key] = updated
        return updated


def _calibrate_local_tokens(model: Any, scope: str, local_raw: int) -> tuple[int, float]:
    factor = _get_calibration_factor(model, scope)
    adjusted = int(round(max(1.0, float(local_raw) * factor)))
    if adjusted <= 0:
        adjusted = 1
    return adjusted, factor


def _error_pct(remote_tokens: int, local_tokens: int) -> float:
    return (float(local_tokens - remote_tokens) / float(max(remote_tokens, 1))) * 100.0


def _model_log_fields(model: Any, *, api_base: Any = None) -> str:
    return format_model_log_fields(model, api_base=api_base)


def _log_compare(
    *,
    kind: str,
    model: Any,
    scope: str,
    used: str,
    remote: Any,
    local_raw: int,
    local_adj: int,
    factor: float,
    api_base: Any = None,
    err_pct: Optional[float] = None,
) -> None:
    model_txt = _model_log_fields(model, api_base=api_base)
    msg = (
        f"{kind} compare {model_txt} scope={scope} used={used} "
        f"remote={remote} local_adj={local_adj} local_raw={local_raw} "
        f"factor={factor:.4f}"
    )
    if err_pct is not None:
        msg += f" err_pct={err_pct:+.2f}%"
    _LOG.debug(msg)


def _is_remote_401_error(exc: Exception) -> bool:
    code = getattr(exc, "status_code", None)
    if code == 401:
        return True
    txt = str(exc).lower()
    return ("401" in txt and "unauthorized" in txt)


def _extract_count_tokens_inputs(args: tuple[Any, ...], kwargs: dict[str, Any]) -> tuple[Any, Any, Any]:
    model = kwargs.get("model")
    messages = kwargs.get("messages")
    api_base = kwargs.get("api_base")
    if len(args) >= 1 and model is None:
        model = args[0]
    if len(args) >= 2 and messages is None:
        messages = args[1]
    if len(args) >= 4 and api_base is None:
        api_base = args[3]
    return model, messages, api_base




def _ensure_import_hook() -> None:
    global _FINDER_INSTALLED
    if _FINDER_INSTALLED:
        return
    for f in sys.meta_path:
        if isinstance(f, _PatchFinder):
            _FINDER_INSTALLED = True
            return
    sys.meta_path.insert(0, _PatchFinder())
    _FINDER_INSTALLED = True


class _PatchFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname != _TARGET_HANDLER:
            return None

        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        if not spec or not spec.loader:
            return spec

        original_loader = spec.loader

        class _PatchedLoader(importlib.abc.Loader):
            def create_module(self, spec_):
                create = getattr(original_loader, "create_module", None)
                return create(spec_) if callable(create) else None

            def exec_module(self, module):
                execm = getattr(original_loader, "exec_module", None)
                if not callable(execm):
                    raise ImportError(f"Loader for {fullname} has no exec_module()")
                execm(module)
                _patch_anthropic_count_tokens()

        spec.loader = _PatchedLoader()
        return spec


def _patch_anthropic_count_tokens() -> None:
    if not _FORCE_ANTHROPIC_COUNT_TOKENS:
        return
    try:
        from litellm.llms.anthropic.count_tokens.handler import AnthropicCountTokensHandler  # type: ignore
    except Exception:
        return

    if getattr(AnthropicCountTokensHandler, _COUNT_TOKENS_PATCH_ATTR, False):
        return

    orig = getattr(AnthropicCountTokensHandler, "handle_count_tokens_request", None)
    if not callable(orig):
        return

    async def patched(self, *args: Any, **kwargs: Any):
        orig_fn = getattr(type(self), _COUNT_TOKENS_ORIG_ATTR, None)
        model, messages, api_base = _extract_count_tokens_inputs(args, kwargs)
        scope = "messages"
        local_raw = _estimate_local_tokens_from_messages(messages)
        local_adj, factor = _calibrate_local_tokens(model, scope, local_raw)
        if not callable(orig_fn):
            if _LOG.enabled():
                _log_compare(
                    kind="count_tokens",
                    model=model,
                    scope=scope,
                    used="local",
                    remote="unavailable",
                    local_raw=local_raw,
                    local_adj=local_adj,
                    factor=factor,
                    api_base=api_base,
                )
            return {"input_tokens": local_adj}

        use_local_fallback = _should_use_local_anthropic_count_tokens(model, api_base)

        try:
            remote = await orig_fn(self, *args, **kwargs)
            remote_tokens = _extract_positive_input_tokens(remote)
            if _LOG.enabled():
                if remote_tokens is not None:
                    _update_calibration_factor(model, scope, local_raw, remote_tokens)
                    local_adj, factor = _calibrate_local_tokens(model, scope, local_raw)
                    _log_compare(
                        kind="count_tokens",
                        model=model,
                        scope=scope,
                        used="remote",
                        remote=remote_tokens,
                        local_raw=local_raw,
                        local_adj=local_adj,
                        factor=factor,
                        api_base=api_base,
                        err_pct=_error_pct(remote_tokens, local_adj),
                    )
                else:
                    _log_compare(
                        kind="count_tokens",
                        model=model,
                        scope=scope,
                        used="remote",
                        remote="missing",
                        local_raw=local_raw,
                        local_adj=local_adj,
                        factor=factor,
                        api_base=api_base,
                    )
            return remote
        except Exception as e:
            if use_local_fallback:
                is_401 = _is_remote_401_error(e)
                if _LOG.enabled():
                    _log_compare(
                        kind="count_tokens",
                        model=model,
                        scope=scope,
                        used="local",
                        remote=f"error:{type(e).__name__}",
                        local_raw=local_raw,
                        local_adj=local_adj,
                        factor=factor,
                        api_base=api_base,
                    )
                    if is_401:
                        model_txt = _model_log_fields(model, api_base=api_base)
                        _LOG.debug(
                            f"captured remote 401 in count_tokens {model_txt} "
                            f"api_base={api_base!r} fallback=local"
                        )
                    model_txt = _model_log_fields(model, api_base=api_base)
                    _LOG.debug(
                        f"local count_tokens injected {model_txt} scope={scope} "
                        f"raw={local_raw} adj={local_adj}"
                    )
                return {"input_tokens": local_adj}

            if _LOG.enabled():
                _log_compare(
                    kind="count_tokens",
                    model=model,
                    scope=scope,
                    used="remote-error",
                    remote=f"error:{type(e).__name__}",
                    local_raw=local_raw,
                    local_adj=local_adj,
                    factor=factor,
                    api_base=api_base,
                )
            raise

    try:
        setattr(AnthropicCountTokensHandler, _COUNT_TOKENS_ORIG_ATTR, orig)
        AnthropicCountTokensHandler.handle_count_tokens_request = patched  # type: ignore[assignment]
        setattr(AnthropicCountTokensHandler, _COUNT_TOKENS_PATCH_ATTR, True)
        _LOG.debug("patched AnthropicCountTokensHandler.handle_count_tokens_request (local)")
    except Exception:
        return


def install() -> None:
    if not SETTINGS.is_enabled():
        _LOG.debug("disabled (local_token_counter patch not enabled)")
        return

    _ensure_import_hook()

    try:
        import litellm  # type: ignore
    except Exception:
        # Silent by default; common during multi-process bootstraps.
        return

    _patch_anthropic_count_tokens()

    if getattr(litellm, _PATCH_ATTR, False):
        return

    orig = getattr(litellm, "token_counter", None)
    if not callable(orig):
        _LOG.debug("litellm.token_counter not callable; skipping")
        return

    setattr(litellm, _ORIG_ATTR, orig)

    def patched_token_counter(*args: Any, **kwargs: Any) -> int:
        model = kwargs.get("model") if isinstance(kwargs.get("model"), str) else None
        if _matches_provider(kwargs) or _matches_model(model):
            local_raw, scope = _estimate_local_tokens(model, kwargs)
            local_adj, factor = _calibrate_local_tokens(model, scope, local_raw)
            try:
                remote = int(orig(*args, **kwargs))
                if remote > 0:
                    _update_calibration_factor(model, scope, local_raw, remote)
                    local_adj, factor = _calibrate_local_tokens(model, scope, local_raw)
                    _log_compare(
                        kind="token_counter",
                        model=model,
                        scope=scope,
                        used="remote",
                        remote=remote,
                        local_raw=local_raw,
                        local_adj=local_adj,
                        factor=factor,
                        api_base=kwargs.get("api_base"),
                        err_pct=_error_pct(remote, local_adj),
                    )
                    return remote
                _log_compare(
                    kind="token_counter",
                    model=model,
                    scope=scope,
                    used="local",
                    remote="missing",
                    local_raw=local_raw,
                    local_adj=local_adj,
                    factor=factor,
                    api_base=kwargs.get("api_base"),
                )
            except Exception as e:
                _log_compare(
                    kind="token_counter",
                    model=model,
                    scope=scope,
                    used="local",
                    remote=f"error:{type(e).__name__}",
                    local_raw=local_raw,
                    local_adj=local_adj,
                    factor=factor,
                    api_base=kwargs.get("api_base"),
                )
            model_txt = _model_log_fields(model, api_base=kwargs.get("api_base"))
            _LOG.debug(
                f"local token_counter injected {model_txt} scope={scope} "
                f"raw={local_raw} adj={local_adj}"
            )
            return int(local_adj)
        return int(orig(*args, **kwargs))

    litellm.token_counter = patched_token_counter  # type: ignore[assignment]
    setattr(litellm, _PATCH_ATTR, True)
    _LOG.debug(
        "installed local token_counter patch "
        f"providers={list(_PROVIDERS)} patterns={list(_MODEL_PATTERNS)} hosts={list(_HOST_SUBSTRS)}"
    )
