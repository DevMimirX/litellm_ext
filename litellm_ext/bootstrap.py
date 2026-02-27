from __future__ import annotations

import os

from typing import List

from .core.registry import install_httpx_patch
from .extensions import install_all


def install_extensions() -> List[str]:
    """Install all enabled extensions in the recommended order."""
    os.environ.setdefault("LITELLM_EXT_INSTALL_PID", str(os.getpid()))
    install_httpx_patch()
    return install_all()
