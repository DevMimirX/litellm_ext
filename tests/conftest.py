"""Pytest glue for LiteLLM patch bundle.

The patch bundle is designed to be loaded via PYTHONPATH into a running LiteLLM
proxy process. For unit tests we need to:

1) Make extension modules importable as top-level modules.
2) Provide an isolated, writable config file via LITELLM_EXT_CONFIG_PATH.
3) Keep tests isolated:
   - Reset config to a known baseline between tests.
   - Clear httpx mutators (stored globally on the httpx module).
   - Drop cached patch modules so they re-read config when imported.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

# Ensure extension modules are importable.
PATCH_DIR = Path(__file__).resolve().parents[1]
if str(PATCH_DIR) not in sys.path:
    sys.path.insert(0, str(PATCH_DIR))

_BASELINE_CFG_TEXT: str | None = None
_CFG_PATH: Path | None = None


def _reset_patch_modules() -> None:
    for name in list(sys.modules.keys()):
        if name.startswith("litellm_ext"):
            sys.modules.pop(name, None)


@pytest.fixture(scope="session", autouse=True)
def _session_config(tmp_path_factory: pytest.TempPathFactory):
    """Create a writable config file and point the loader to it."""
    global _BASELINE_CFG_TEXT, _CFG_PATH

    cfg_path = tmp_path_factory.mktemp("litellm_ext") / "extensions.yaml"
    default_cfg = PATCH_DIR / "config" / "extensions.yaml"

    if default_cfg.exists():
        _BASELINE_CFG_TEXT = default_cfg.read_text(encoding="utf-8")
    else:
        _BASELINE_CFG_TEXT = "version: 1\n"

    cfg_path.write_text(_BASELINE_CFG_TEXT, encoding="utf-8")
    _CFG_PATH = cfg_path

    os.environ["LITELLM_EXT_CONFIG_PATH"] = str(cfg_path)

    # Prime cache using the test config path.
    try:
        from litellm_ext.core import config as _cfg

        _cfg.load_config(force=True)
    except Exception:
        pass

    yield


@pytest.fixture(autouse=True)
def _isolation_between_tests():
    """Reset config + module caches and clear httpx mutators between tests."""
    if _CFG_PATH is not None and _BASELINE_CFG_TEXT is not None:
        try:
            _CFG_PATH.write_text(_BASELINE_CFG_TEXT, encoding="utf-8")
        except Exception:
            pass

    # Drop cached patch modules so they re-read config on import.
    _reset_patch_modules()

    # Clear httpx mutators to avoid cross-test interference.
    try:
        from litellm_ext.core import registry as reg

        reg.clear_mutators()
    except Exception:
        pass

    # Clear file-based install markers to allow fresh installs in each test
    try:
        import tempfile
        import os
        for marker in ["litellm_ext_hard_caps_installed_", "litellm_ext_count_tokens_installed_"]:
            for f in Path(tempfile.gettempdir()).glob(f"{marker}*"):
                try:
                    f.unlink()
                except Exception:
                    pass
    except Exception:
        pass

    # Clear model alias cache to avoid cross-test leakage from LITELLM_CONFIG swaps.
    try:
        from litellm_ext.core.model_alias import reset_model_alias_cache

        reset_model_alias_cache()
    except Exception:
        pass

    yield
