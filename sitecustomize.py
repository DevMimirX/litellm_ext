# Auto-loaded by Python on startup if this directory is on PYTHONPATH.

import os
import sys
from pathlib import Path

from litellm_ext.bootstrap import install_extensions


def _env_true(name: str) -> bool:
    value = os.environ.get(name, "")
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _is_agent_config_apply_process() -> bool:
    if not sys.argv:
        return False
    argv0 = sys.argv[0]
    argv0_name = Path(argv0).name
    return (
        argv0_name == "agent-config-apply"
        or argv0.endswith("/litellm_ext/agent_config/cli.py")
        or argv0.endswith("\\litellm_ext\\agent_config\\cli.py")
    )


if not (_env_true("LITELLM_EXT_SKIP_AUTO_PATCH") or _is_agent_config_apply_process()):
    try:
        install_extensions()
    except ModuleNotFoundError as e:
        if getattr(e, "name", None) != "litellm":
            raise
