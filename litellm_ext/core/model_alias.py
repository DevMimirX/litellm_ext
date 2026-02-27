from __future__ import annotations

import os
from typing import Any, Dict, Optional, Tuple
from urllib.parse import urlparse

import yaml

_CACHE_KEY: Optional[tuple[str, Optional[float]]] = None
_ALIAS_TO_PROVIDER: Dict[str, str] = {}
_PROVIDER_HOST_TO_ALIAS: Dict[Tuple[str, str], str] = {}
_PROVIDER_TO_ALIAS_UNIQUE: Dict[str, str] = {}


def canonicalize_model_name(name: Any) -> str:
    if name is None:
        return ""
    s = str(name).strip()
    return s.lower()


def provider_model_name(name: Any) -> str:
    s = str(name or "").strip()
    if "/" in s:
        s = s.split("/")[-1].strip()
    return s


def canonicalize_host(value: Any) -> str:
    if value is None:
        return ""
    s = str(value).strip()
    if not s:
        return ""
    try:
        parsed = urlparse(s if "://" in s else f"https://{s}")
        host = (parsed.hostname or "").strip().lower()
        return host
    except Exception:
        return ""


def _load_model_alias_maps() -> tuple[Dict[str, str], Dict[Tuple[str, str], str], Dict[str, str]]:
    global _CACHE_KEY, _ALIAS_TO_PROVIDER, _PROVIDER_HOST_TO_ALIAS, _PROVIDER_TO_ALIAS_UNIQUE

    cfg_path = (os.environ.get("LITELLM_CONFIG") or "").strip()
    if not cfg_path:
        return {}, {}, {}

    mtime: Optional[float]
    try:
        mtime = os.path.getmtime(cfg_path)
    except Exception:
        mtime = None

    cache_key = (cfg_path, mtime)
    if _CACHE_KEY == cache_key:
        return _ALIAS_TO_PROVIDER, _PROVIDER_HOST_TO_ALIAS, _PROVIDER_TO_ALIAS_UNIQUE

    alias_to_provider: Dict[str, str] = {}
    provider_host_to_alias_tmp: Dict[Tuple[str, str], set[str]] = {}
    provider_to_alias_tmp: Dict[str, set[str]] = {}

    try:
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}

        model_list = cfg.get("model_list")
        if isinstance(model_list, list):
            for item in model_list:
                if not isinstance(item, dict):
                    continue
                alias_raw = item.get("model_name")
                params = item.get("litellm_params")
                if not isinstance(alias_raw, str) or not isinstance(params, dict):
                    continue

                model_raw = params.get("model")
                if not isinstance(model_raw, str):
                    continue

                alias = alias_raw.strip()
                alias_key = canonicalize_model_name(alias)
                provider = provider_model_name(model_raw)
                provider_key = canonicalize_model_name(provider)
                api_base = params.get("api_base")
                host = canonicalize_host(api_base)

                if alias_key and provider:
                    alias_to_provider[alias_key] = provider
                if alias_key and provider_key:
                    provider_to_alias_tmp.setdefault(provider_key, set()).add(alias)
                if alias_key and provider_key and host:
                    key = (provider_key, host)
                    provider_host_to_alias_tmp.setdefault(key, set()).add(alias)
    except Exception:
        alias_to_provider = {}
        provider_host_to_alias_tmp = {}
        provider_to_alias_tmp = {}

    provider_host_to_alias: Dict[Tuple[str, str], str] = {}
    for key, aliases in provider_host_to_alias_tmp.items():
        if len(aliases) == 1:
            provider_host_to_alias[key] = next(iter(aliases))

    provider_to_alias_unique: Dict[str, str] = {}
    for provider_key, aliases in provider_to_alias_tmp.items():
        if len(aliases) == 1:
            provider_to_alias_unique[provider_key] = next(iter(aliases))

    _CACHE_KEY = cache_key
    _ALIAS_TO_PROVIDER = alias_to_provider
    _PROVIDER_HOST_TO_ALIAS = provider_host_to_alias
    _PROVIDER_TO_ALIAS_UNIQUE = provider_to_alias_unique
    return _ALIAS_TO_PROVIDER, _PROVIDER_HOST_TO_ALIAS, _PROVIDER_TO_ALIAS_UNIQUE


def reset_model_alias_cache() -> None:
    global _CACHE_KEY, _ALIAS_TO_PROVIDER, _PROVIDER_HOST_TO_ALIAS, _PROVIDER_TO_ALIAS_UNIQUE
    _CACHE_KEY = None
    _ALIAS_TO_PROVIDER = {}
    _PROVIDER_HOST_TO_ALIAS = {}
    _PROVIDER_TO_ALIAS_UNIQUE = {}


def provider_model_for_alias(model: Any) -> Optional[str]:
    alias_key = canonicalize_model_name(model)
    if not alias_key:
        return None
    alias_to_provider, _, _ = _load_model_alias_maps()
    return alias_to_provider.get(alias_key)


def alias_for_provider_model(model: Any, *, host: Any = None, api_base: Any = None) -> Optional[str]:
    provider = provider_model_name(model)
    provider_key = canonicalize_model_name(provider)
    if not provider_key:
        return None

    _, provider_host_to_alias, provider_to_alias_unique = _load_model_alias_maps()

    host_key = canonicalize_host(host or api_base)
    if host_key:
        alias = provider_host_to_alias.get((provider_key, host_key))
        if alias:
            return alias

    return provider_to_alias_unique.get(provider_key)


def display_model_for_log(model: Any, *, host: Any = None, api_base: Any = None) -> str:
    raw = str(model or "").strip()
    if not raw:
        return ""
    mapped = provider_model_for_alias(raw)
    # Keep explicit non-identity aliases as-is (e.g. "glm-5-ali" -> "glm-5"),
    # but allow host-specific alias selection when raw is provider-like
    # (e.g. "glm-5" on DashScope should display "glm-5-ali").
    if mapped and canonicalize_model_name(provider_model_name(raw)) != canonicalize_model_name(provider_model_name(mapped)):
        return raw
    alias = alias_for_provider_model(raw, host=host, api_base=api_base)
    return alias or raw


def format_model_log_fields(model: Any, *, host: Any = None, api_base: Any = None) -> str:
    display = display_model_for_log(model, host=host, api_base=api_base)
    resolved = str(model or "").strip()
    if display and resolved and display != resolved:
        return f"model={display!r} resolved_model={resolved!r}"
    if display:
        return f"model={display!r}"
    return f"model={model!r}"
