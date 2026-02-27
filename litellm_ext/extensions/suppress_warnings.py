"""Suppress unwanted warnings from LiteLLM."""

from __future__ import annotations

import os
from typing import Any

from ..core.patch import PatchSettings

SETTINGS = PatchSettings(
    "suppress_warnings",
    enabled_default=True,
    debug_default=False,
    enabled_envs=("LITELLM_EXT_SUPPRESS_WARNINGS",),
    debug_envs=("LITELLM_EXT_SUPPRESS_WARNINGS_DEBUG",),
)

_LOG = SETTINGS.logger("litellm_ext.suppress_warnings")

_WARN_PREFIX = "LiteLLM: Failed to fetch remote model cost map"
_WARN_ERROR = "Request URL is missing an 'http://' or 'https://' protocol"


def _ensure_local_model_cost_map_for_blank_url() -> None:
    """Avoid startup warning when model-cost-map URL is explicitly blank."""
    raw_url = os.environ.get("LITELLM_MODEL_COST_MAP_URL")
    if raw_url is None:
        return

    url = raw_url.strip()
    if url:
        return

    previous = os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "true")
    if previous.lower() in {"1", "true", "yes", "y", "on"}:
        _LOG.debug("forced local model cost map for blank LITELLM_MODEL_COST_MAP_URL")


def _render_log_message(msg: Any, args: tuple[Any, ...]) -> str:
    template = str(msg)
    if not args:
        return template
    try:
        return template % args
    except Exception:
        return " ".join([template, *(str(a) for a in args)])


def _is_model_cost_map_url_warning(msg: Any, args: tuple[Any, ...]) -> bool:
    rendered = _render_log_message(msg, args)
    if _WARN_PREFIX not in rendered or _WARN_ERROR not in rendered:
        return False

    # In current LiteLLM the first arg is `url`; only suppress when it is blank.
    if args:
        return str(args[0]).strip() == ""

    # Fallback for pre-formatted logger calls.
    return "from :" in rendered


def _resolve_verbose_logger() -> Any:
    """Import verbose logger across LiteLLM versions."""
    try:
        from litellm import verbose_logger

        return verbose_logger
    except Exception:
        # Older LiteLLM versions expose this logger from litellm_core_utils.
        from litellm.litellm_core_utils.logging import verbose_logger

        return verbose_logger


def _suppress_model_cost_map_warning() -> None:
    """Suppress only the specific invalid model-cost-map-url warning."""
    try:
        from litellm.litellm_core_utils.get_model_cost_map import GetModelCostMap  # noqa: F401

        verbose_logger = _resolve_verbose_logger()
        original_warning = verbose_logger.warning

        def filtered_warning(msg: Any, *args: Any, **kwargs: Any) -> None:
            if _is_model_cost_map_url_warning(msg, args):
                return
            original_warning(msg, *args, **kwargs)

        verbose_logger.warning = filtered_warning
        _LOG.debug("applied model cost map warning suppression")
    except Exception as e:
        _LOG.debug(f"failed to suppress model cost map warning: {e}")


def install() -> None:
    """Install warning suppression patches."""
    _ensure_local_model_cost_map_for_blank_url()
    _suppress_model_cost_map_warning()
