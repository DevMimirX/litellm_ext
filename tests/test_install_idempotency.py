import os
import sys
import tempfile
import types
from pathlib import Path



def test_bootstrap_sets_install_pid(monkeypatch):
    import litellm_ext.bootstrap as bootstrap

    monkeypatch.delenv("LITELLM_EXT_INSTALL_PID", raising=False)

    called = []
    monkeypatch.setattr(bootstrap, "install_httpx_patch", lambda: called.append("httpx"))
    monkeypatch.setattr(bootstrap, "install_all", lambda: [])

    bootstrap.install_extensions()

    assert os.environ.get("LITELLM_EXT_INSTALL_PID") == str(os.getpid())
    assert called == ["httpx"]


def test_count_tokens_install_respects_marker(monkeypatch):
    import litellm_ext.extensions.httpx.count_tokens_stub as ct

    monkeypatch.setenv("LITELLM_EXT_INSTALL_PID", "999")
    marker = Path(tempfile.gettempdir()) / "litellm_ext_count_tokens_installed_999"
    marker.write_text("1")

    called = []
    monkeypatch.setattr(ct, "install_httpx_patch", lambda: called.append("httpx"))
    monkeypatch.setattr(ct, "register_request_mutator", lambda *a, **k: called.append("mutator"))

    ct.install()

    assert called == []


def test_count_tokens_install_sets_marker(monkeypatch):
    import litellm_ext.extensions.httpx.count_tokens_stub as ct

    monkeypatch.setenv("LITELLM_EXT_INSTALL_PID", "1001")
    marker = Path(tempfile.gettempdir()) / "litellm_ext_count_tokens_installed_1001"
    if marker.exists():
        marker.unlink()

    called = []
    monkeypatch.setattr(ct, "install_httpx_patch", lambda: called.append("httpx"))
    monkeypatch.setattr(ct, "register_request_mutator", lambda *a, **k: called.append("mutator"))
    # Enable the stub via environment variable
    monkeypatch.setenv("LITELLM_EXT_COUNT_TOKENS_STUB", "1")

    ct.install()

    assert called == ["httpx", "mutator"]
    assert marker.exists()



def test_hard_caps_install_respects_marker(monkeypatch):
    import litellm_ext.extensions.httpx.hard_caps as hc

    monkeypatch.setenv("LITELLM_EXT_INSTALL_PID", "888")
    marker = Path(tempfile.gettempdir()) / "litellm_ext_hard_caps_installed_888"
    marker.write_text("1")

    called = []
    monkeypatch.setattr(hc, "install_httpx_patch", lambda: called.append("httpx"))
    monkeypatch.setattr(hc, "register_request_mutator", lambda *a, **k: called.append("mutator"))

    hc.install()

    assert called == []


def test_hard_caps_install_sets_marker(monkeypatch):
    import litellm_ext.extensions.httpx.hard_caps as hc

    monkeypatch.setenv("LITELLM_EXT_INSTALL_PID", "1002")
    marker = Path(tempfile.gettempdir()) / "litellm_ext_hard_caps_installed_1002"
    if marker.exists():
        marker.unlink()

    called = []
    monkeypatch.setattr(hc, "install_httpx_patch", lambda: called.append("httpx"))
    monkeypatch.setattr(hc, "register_request_mutator", lambda *a, **k: called.append("mutator"))

    hc.install()

    assert called == ["httpx", "mutator"]
    assert marker.exists()


def test_hard_caps_patch_logs_once(monkeypatch):
    import litellm_ext.extensions.httpx.hard_caps as hc

    mod = types.ModuleType("litellm")

    def completion(*args, **kwargs):
        return None

    async def acompletion(*args, **kwargs):
        return None

    class Router:
        def completion(self, *args, **kwargs):
            return None

        async def acompletion(self, *args, **kwargs):
            return None

    mod.completion = completion
    mod.acompletion = acompletion
    mod.Router = Router

    monkeypatch.setitem(sys.modules, "litellm", mod)
    # Ensure patch flags are unset for this synthetic module.
    if hasattr(mod, hc._PATCH_ATTR_LITELLM):
        delattr(mod, hc._PATCH_ATTR_LITELLM)
    if hasattr(mod, hc._PATCH_LOGGED):
        delattr(mod, hc._PATCH_LOGGED)

    hc._patch_litellm()
    assert getattr(mod, hc._PATCH_LOGGED, False) is True
    hc._patch_litellm()
    assert getattr(mod, hc._PATCH_LOGGED, False) is True
