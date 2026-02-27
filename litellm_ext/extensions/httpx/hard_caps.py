from __future__ import annotations

import json
import os
import sys
import tempfile
from functools import wraps
import importlib.abc
import importlib.machinery
from typing import Any, Dict, Optional

import httpx

from ...core.model_alias import display_model_for_log, provider_model_for_alias
from ...core.patch import PatchSettings
from ...core.registry import install_httpx_patch, register_request_mutator
from ...policy import SKIP_PATH_SUFFIXES, enforce, enforce_call, get_requested_output, match_limits, normalize_model

SETTINGS = PatchSettings(
    "hard_caps",
    enabled_default=True,
    debug_default=False,
    enabled_envs=("LITELLM_EXT_HARD_CAPS",),
    debug_envs=("LITELLM_EXT_HARD_CAPS_DEBUG",),
)
_LOG = SETTINGS.logger("litellm_ext.hard_caps")

_PATCH_ATTR_LITELLM = "_litellm_ext_hard_caps_applied"
_PATCH_LOGGED = "_litellm_ext_hard_caps_logged"

_ORIG_LITELLM_SYNC = "_litellm_ext_hard_caps_orig_completion"
_ORIG_LITELLM_ASYNC = "_litellm_ext_hard_caps_orig_acompletion"

_TARGET_LITELLM = "litellm"
_FINDER_INSTALLED = False

# Global flag to track if we've already installed (prevents race conditions in same process)
_INSTALL_CALLED = False

# File-based marker for cross-process idempotency (handles conda run subprocesses)
def _get_install_marker_path() -> str:
    """Get path to a file-based marker for cross-process install tracking."""
    # Use temp dir + PID of the root process for uniqueness
    root_pid = os.environ.get("LITELLM_EXT_INSTALL_PID") or str(os.getpid())
    return os.path.join(tempfile.gettempdir(), f"litellm_ext_hard_caps_installed_{root_pid}")


def _is_installed_marker_present() -> bool:
    """Check if install marker file exists (indicates install already done in this session)."""
    try:
        return os.path.exists(_get_install_marker_path())
    except Exception:
        return False


def _set_installed_marker() -> None:
    """Create install marker file to indicate install is complete."""
    try:
        with open(_get_install_marker_path(), "w") as f:
            f.write("1")
    except Exception:
        pass


JsonDict = Dict[str, Any]


def _req_summary(request: httpx.Request) -> str:
    try:
        url = request.url
        host = getattr(url, "host", "")
        path = getattr(url, "path", "")
        method = request.method.upper() if request.method else ""
        return f"{method} {host}{path}"
    except Exception:
        return "<request>"


def _rewrite_alias_model_for_messages(request: httpx.Request, obj: JsonDict) -> bool:
    if "model" not in obj:
        return False
    try:
        path = (request.url.path or "").lower()
    except Exception:
        path = ""
    if not path.endswith("/v1/messages"):
        return False

    model_raw = obj.get("model")
    mapped = provider_model_for_alias(model_raw)
    if not mapped or str(model_raw).strip() == mapped:
        return False

    obj["model"] = mapped
    _LOG.debug(
        f"(httpx) rewrote outbound model alias {model_raw!r}->{mapped!r} req={_req_summary(request)}"
    )
    return True


def _should_skip_request(request: httpx.Request) -> bool:
    try:
        path = getattr(request.url, "path", "") or ""
        return any(path.endswith(suf) for suf in SKIP_PATH_SUFFIXES)
    except Exception:
        return False


def _get_request_body_bytes(request: httpx.Request) -> bytes:
    content = getattr(request, "content", None)
    if isinstance(content, (bytes, bytearray)):
        return bytes(content)
    if isinstance(content, str):
        return content.encode("utf-8", errors="replace")
    return b""


def _try_parse_json_request(request: httpx.Request) -> Optional[JsonDict]:
    if _should_skip_request(request):
        return None

    body = _get_request_body_bytes(request)
    if not body:
        return None

    b2 = body.lstrip()
    ctype = (request.headers.get("content-type") or "").lower()
    if "application/json" not in ctype and b2[:1] not in (b"{", b"["):
        return None

    try:
        obj = json.loads(b2.decode("utf-8", errors="replace"))
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def _mutate_httpx_request_json_inplace(request: httpx.Request, obj: JsonDict) -> None:
    new_bytes = json.dumps(obj, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    request.headers["content-type"] = "application/json"
    request.headers["content-length"] = str(len(new_bytes))

    if hasattr(request, "_content"):
        try:
            setattr(request, "_content", new_bytes)
        except Exception:
            pass
    if hasattr(request, "_stream"):
        try:
            request._stream = httpx.ByteStream(new_bytes)  # type: ignore[attr-defined]
        except Exception:
            pass


def _hard_caps_mutator(request: httpx.Request) -> Optional[httpx.Response]:
    if _LOG.enabled():
        _LOG.debug(f"(httpx) hard_caps invoked req={_req_summary(request)}")
    if request.method.upper() != "POST":
        return None

    obj = _try_parse_json_request(request)
    if not isinstance(obj, dict) or "model" not in obj:
        return None

    try:
        host = getattr(request.url, "host", "")
        model = obj.get("model")
        model_raw = str(model or "").strip()
        model_log = display_model_for_log(model, host=host)
        # For host-routed aliases (e.g. glm-5 -> glm-5-ali on DashScope), enforce
        # policy against the alias key, then restore provider model before send.
        policy_alias_applied = False
        if isinstance(model_log, str) and model_log and model_log != model_raw:
            obj["model"] = model_log
            policy_alias_applied = True

        before_out = get_requested_output(obj)
        caps_changed = enforce(obj, allow_raise=False)
        if policy_alias_applied and str(obj.get("model") or "").strip() == model_log:
            obj["model"] = model
        alias_rewritten = _rewrite_alias_model_for_messages(request, obj)
        changed = caps_changed or alias_rewritten
        resolved_model = obj.get("model")
        if changed:
            _mutate_httpx_request_json_inplace(request, obj)
            after_out = get_requested_output(obj)
            _LOG.debug(
                f"(httpx) enforced caps model={model_log!r} resolved_model={resolved_model!r} "
                f"max_tokens {before_out}->{after_out} req={_req_summary(request)}"
            )
        elif _LOG.enabled():
            limit_out = None
            try:
                limit_out = match_limits(normalize_model(model_log or model or "")).get("max_output")
            except Exception:
                limit_out = None
            if before_out is None:
                _LOG.debug(
                    f"(httpx) caps skipped model={model_log!r} resolved_model={resolved_model!r} "
                    f"max_tokens missing "
                    f"limit={limit_out} req={_req_summary(request)}"
                )
            else:
                _LOG.debug(
                    f"(httpx) caps unchanged model={model_log!r} resolved_model={resolved_model!r} "
                    f"max_tokens={before_out} "
                    f"limit={limit_out} req={_req_summary(request)}"
                )
    except Exception as e:
        model_err = display_model_for_log(obj.get("model"), host=getattr(request.url, "host", ""))
        _LOG.debug(
            f"(httpx) enforce failed model={model_err!r} "
            f"resolved_model={obj.get('model')!r} "
            f"req={_req_summary(request)}: {type(e).__name__}: {e}"
        )

    return None


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
        if fullname != _TARGET_LITELLM:
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
                # Temporarily restore original loader for resource resolution during exec
                original_module_loader = getattr(module, "__loader__", None)
                original_module_spec_loader = getattr(getattr(module, "__spec__", None), "loader", None)
                try:
                    if hasattr(module, "__loader__"):
                        module.__loader__ = original_loader
                    if hasattr(module, "__spec__") and module.__spec__ is not None:
                        module.__spec__.loader = original_loader
                    execm(module)
                finally:
                    if hasattr(module, "__loader__"):
                        module.__loader__ = original_module_loader
                    if hasattr(module, "__spec__") and module.__spec__ is not None:
                        module.__spec__.loader = original_module_spec_loader
                _patch_litellm()

            def get_resource_reader(self, name):
                # Delegate resource reading to original loader (required for importlib.resources)
                get_reader = getattr(original_loader, "get_resource_reader", None)
                if callable(get_reader):
                    return get_reader(name)
                return None

        spec.loader = _PatchedLoader()
        return spec


def _patch_litellm() -> None:
    try:
        import litellm  # type: ignore
    except Exception:
        # litellm not available yet (common with conda run subprocesses)
        # Silently skip - we'll retry on next install() call
        return

    # Set flag immediately to prevent race conditions with concurrent installs
    if getattr(litellm, _PATCH_ATTR_LITELLM, False):
        return
    setattr(litellm, _PATCH_ATTR_LITELLM, True)

    if not hasattr(litellm, _ORIG_LITELLM_SYNC):
        setattr(litellm, _ORIG_LITELLM_SYNC, getattr(litellm, "completion", None))
    if not hasattr(litellm, _ORIG_LITELLM_ASYNC):
        setattr(litellm, _ORIG_LITELLM_ASYNC, getattr(litellm, "acompletion", None))

    orig_completion = getattr(litellm, _ORIG_LITELLM_SYNC)
    orig_acompletion = getattr(litellm, _ORIG_LITELLM_ASYNC)

    if callable(orig_completion):

        @wraps(orig_completion)
        def patched_completion(*args: Any, **kwargs: Any):
            new_args, new_kwargs = enforce_call(args, kwargs, allow_raise=True)
            return orig_completion(*new_args, **new_kwargs)

        litellm.completion = patched_completion  # type: ignore[assignment]
        _LOG.debug("patched litellm.completion (hard caps wrapper)")

    if callable(orig_acompletion):

        @wraps(orig_acompletion)
        async def patched_acompletion(*args: Any, **kwargs: Any):
            new_args, new_kwargs = enforce_call(args, kwargs, allow_raise=True)
            return await orig_acompletion(*new_args, **new_kwargs)

        litellm.acompletion = patched_acompletion  # type: ignore[assignment]
        _LOG.debug("patched litellm.acompletion (hard caps wrapper)")

    Router = None
    try:
        from litellm.router import Router as _Router  # type: ignore
        Router = _Router
    except Exception:
        Router = getattr(litellm, "Router", None)

    if Router is not None:
        try:
            if hasattr(Router, "completion"):
                orig = Router.completion

                @wraps(orig)
                def patched_router_completion(self: Any, *args: Any, **kwargs: Any):
                    new_args, new_kwargs = enforce_call(args, kwargs, allow_raise=True)
                    return orig(self, *new_args, **new_kwargs)

                Router.completion = patched_router_completion  # type: ignore[assignment]
                _LOG.debug("patched Router.completion (hard caps wrapper)")

            if hasattr(Router, "acompletion"):
                orig = Router.acompletion

                @wraps(orig)
                async def patched_router_acompletion(self: Any, *args: Any, **kwargs: Any):
                    new_args, new_kwargs = enforce_call(args, kwargs, allow_raise=True)
                    return await orig(self, *new_args, **new_kwargs)

                Router.acompletion = patched_router_acompletion  # type: ignore[assignment]
                _LOG.debug("patched Router.acompletion (hard caps wrapper)")
        except Exception as e:
            _LOG.debug(f"Router patch failed: {type(e).__name__}: {e}")

    try:
        if not getattr(litellm, _PATCH_LOGGED, False):
            setattr(litellm, _PATCH_LOGGED, True)
            _LOG.debug(f"litellm patch active pid={os.getpid()} module={litellm.__name__}")
    except Exception:
        pass


def install() -> None:
    global _INSTALL_CALLED
    if not SETTINGS.is_enabled():
        _LOG.debug("disabled (hard_caps patch not enabled)")
        return

    # Check both in-process flag and file-based marker for cross-process idempotency
    if _INSTALL_CALLED or _is_installed_marker_present():
        return

    _ensure_import_hook()

    # If litellm is already imported in this process, patch immediately.
    if _TARGET_LITELLM in sys.modules:
        try:
            _patch_litellm()
        except Exception as e:
            _LOG.debug(f"litellm patch failed: {type(e).__name__}: {e}")

    install_httpx_patch()
    register_request_mutator("hard_caps", _hard_caps_mutator, priority=20)
    _INSTALL_CALLED = True
    _set_installed_marker()
    _LOG.debug("installed hard_caps httpx mutator")
