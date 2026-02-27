from __future__ import annotations

import inspect
import os
import sys
import threading
from typing import Awaitable, Callable, List, Optional, Tuple

import httpx

from .patch import PatchSettings


_SETTINGS = PatchSettings(
    "httpx_registry",
    enabled_default=True,
    debug_default=False,
    enabled_envs=("LITELLM_EXT_HTTPX_REGISTRY",),
    debug_envs=("LITELLM_EXT_HTTPX_REGISTRY_DEBUG",),
)
_LOG = _SETTINGS.logger("litellm_ext.httpx_registry")

_PATCH_ATTR = "_litellm_ext_httpx_registry_patched"
_ORIG_SYNC_ATTR = "_litellm_ext_httpx_registry_orig_send"
_ORIG_ASYNC_ATTR = "_litellm_ext_httpx_registry_orig_async_send"
_MUTATORS_ATTR = "_litellm_ext_httpx_registry_mutators"
_RESP_MUTATORS_ATTR = "_litellm_ext_httpx_registry_response_mutators"
_ASYNC_RESP_MUTATORS_ATTR = "_litellm_ext_httpx_registry_async_response_mutators"

HttpxMutator = Callable[[httpx.Request], Optional[httpx.Response]]
HttpxResponseMutator = Callable[[httpx.Request, httpx.Response], Optional[httpx.Response]]
AsyncHttpxResponseMutator = Callable[[httpx.Request, httpx.Response], Awaitable[Optional[httpx.Response]]]

MutatorEntry = Tuple[int, str, HttpxMutator]
ResponseMutatorEntry = Tuple[int, str, HttpxResponseMutator]
AsyncResponseMutatorEntry = Tuple[int, str, AsyncHttpxResponseMutator]

_LOCK = threading.RLock()


def _req_summary(request: httpx.Request) -> str:
    try:
        url = request.url
        host = getattr(url, "host", "")
        path = getattr(url, "path", "")
        method = request.method.upper() if request.method else ""
        return f"{method} {host}{path}"
    except Exception:
        return "<request>"


def _get_mutator_store() -> List[MutatorEntry]:
    store = getattr(httpx, _MUTATORS_ATTR, None)
    if store is None:
        store = []
        setattr(httpx, _MUTATORS_ATTR, store)
    return store


def _get_response_mutator_store() -> List[ResponseMutatorEntry]:
    store = getattr(httpx, _RESP_MUTATORS_ATTR, None)
    if store is None:
        store = []
        setattr(httpx, _RESP_MUTATORS_ATTR, store)
    return store


def _get_async_response_mutator_store() -> List[AsyncResponseMutatorEntry]:
    store = getattr(httpx, _ASYNC_RESP_MUTATORS_ATTR, None)
    if store is None:
        store = []
        setattr(httpx, _ASYNC_RESP_MUTATORS_ATTR, store)
    return store


def register_request_mutator(name: str, fn: HttpxMutator, *, priority: int = 100) -> None:
    with _LOCK:
        store = _get_mutator_store()
        store[:] = [e for e in store if e[1] != name]
        store.append((int(priority), str(name), fn))
        store.sort(key=lambda e: (e[0], e[1]))
        _LOG.debug(f"registered request mutator name={name} priority={priority} total={len(store)}")


def register_response_mutator(name: str, fn: HttpxResponseMutator, *, priority: int = 100) -> None:
    with _LOCK:
        store = _get_response_mutator_store()
        store[:] = [e for e in store if e[1] != name]
        store.append((int(priority), str(name), fn))
        store.sort(key=lambda e: (e[0], e[1]))
        _LOG.debug(f"registered response mutator name={name} priority={priority} total={len(store)}")


def register_async_response_mutator(name: str, fn: AsyncHttpxResponseMutator, *, priority: int = 100) -> None:
    with _LOCK:
        store = _get_async_response_mutator_store()
        store[:] = [e for e in store if e[1] != name]
        store.append((int(priority), str(name), fn))
        store.sort(key=lambda e: (e[0], e[1]))
        _LOG.debug(f"registered async response mutator name={name} priority={priority} total={len(store)}")


def _snapshot_mutators() -> List[MutatorEntry]:
    with _LOCK:
        return list(_get_mutator_store())


def _snapshot_response_mutators() -> List[ResponseMutatorEntry]:
    with _LOCK:
        return list(_get_response_mutator_store())


def _snapshot_async_response_mutators() -> List[AsyncResponseMutatorEntry]:
    with _LOCK:
        return list(_get_async_response_mutator_store())


def clear_mutators() -> None:
    """Testing utility: remove all registered mutators."""
    with _LOCK:
        _get_mutator_store().clear()
        _get_response_mutator_store().clear()
        _get_async_response_mutator_store().clear()


def install_httpx_patch() -> None:
    if not _SETTINGS.is_enabled():
        _LOG.debug("disabled (httpx registry patch not enabled)")
        return

    if getattr(httpx, _PATCH_ATTR, False):
        return

    if not hasattr(httpx.Client, _ORIG_SYNC_ATTR):
        setattr(httpx.Client, _ORIG_SYNC_ATTR, httpx.Client.send)
    if not hasattr(httpx.AsyncClient, _ORIG_ASYNC_ATTR):
        setattr(httpx.AsyncClient, _ORIG_ASYNC_ATTR, httpx.AsyncClient.send)

    orig_sync_send = getattr(httpx.Client, _ORIG_SYNC_ATTR)
    orig_async_send = getattr(httpx.AsyncClient, _ORIG_ASYNC_ATTR)

    def patched_sync_send(self: httpx.Client, request: httpx.Request, *args, **kwargs) -> httpx.Response:
        for _prio, name, fn in _snapshot_mutators():
            try:
                resp = fn(request)
                if resp is not None:
                    return resp
            except Exception as e:
                _LOG.debug(
                    f"mutator failed name={name} (sync) req={_req_summary(request)}: "
                    f"{type(e).__name__}: {e}"
                )
        resp = orig_sync_send(self, request, *args, **kwargs)
        for _prio, name, fn in _snapshot_response_mutators():
            try:
                new_resp = fn(request, resp)
                if new_resp is not None:
                    resp = new_resp
            except Exception as e:
                _LOG.debug(
                    f"response mutator failed name={name} (sync) req={_req_summary(request)}: "
                    f"{type(e).__name__}: {e}"
                )
        return resp

    async def patched_async_send(self: httpx.AsyncClient, request: httpx.Request, *args, **kwargs) -> httpx.Response:
        for _prio, name, fn in _snapshot_mutators():
            try:
                resp = fn(request)
                if resp is not None:
                    return resp
            except Exception as e:
                _LOG.debug(
                    f"mutator failed name={name} (async) req={_req_summary(request)}: "
                    f"{type(e).__name__}: {e}"
                )
        resp = await orig_async_send(self, request, *args, **kwargs)
        for _prio, name, fn in _snapshot_async_response_mutators():
            try:
                new_resp = fn(request, resp)
                if inspect.isawaitable(new_resp):
                    new_resp = await new_resp
                if new_resp is not None:
                    resp = new_resp
            except Exception as e:
                _LOG.debug(
                    f"response mutator failed name={name} (async) req={_req_summary(request)}: "
                    f"{type(e).__name__}: {e}"
                )
        return resp

    httpx.Client.send = patched_sync_send  # type: ignore[assignment]
    httpx.AsyncClient.send = patched_async_send  # type: ignore[assignment]

    setattr(httpx, _PATCH_ATTR, True)
    pid = os.getpid()
    _LOG.debug(
        "installed httpx send wrappers (sync+async); "
        f"pid={pid} "
        f"request_mutators={len(_snapshot_mutators())} "
        f"response_mutators={len(_snapshot_response_mutators())} "
        f"async_response_mutators={len(_snapshot_async_response_mutators())}"
    )
