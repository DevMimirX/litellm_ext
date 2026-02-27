# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Environment setup
uv sync                                      # Create .venv and install dependencies
source .venv/bin/activate                   # Activate environment

# Testing
uv run pytest                                # Run all tests
uv run pytest tests/test_file.py              # Run specific test file
uv run pytest -v                              # Verbose output

# Linting and type checking
uv run ruff check .                          # Ruff linting
uv run ruff check . --fix                    # Auto-fix lint issues
uv run mypy .                                # MyPy type checking

# Running the proxy
litellm-up                                   # Start minimal (no DB, no UI) - fastest
litellm-up-db                                # Start with admin UI and database
litellm-down                                 # Stop LiteLLM processes

# Claude Code settings sync (auto-run on startup)
agent-config-apply --tool claude --strategy overwrite --if-changed --optional --quiet --no-backup
agent-config-apply --list-tools             # List supported CLI config adapters
```

## Architecture

This is a Python proxy layer extending LiteLLM using **import hooks and monkey-patching**. Extensions intercept HTTP requests, ASGI middleware, and LiteLLM function calls to add custom behaviors like token enforcement, streaming fixes, and schema conversion.

### Core Pattern: Import Hook Patching

```python
class _PatchFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname != TARGET_MODULE:
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        original_loader = spec.loader
        class _PatchedLoader(importlib.abc.Loader):
            def exec_module(self, module):
                original_loader.exec_module(module)
                _patch_module(module)  # Apply patches after module loads
        spec.loader = _PatchedLoader()
        return spec
```

Key principles:
- `_patched` flags prevent double-patching
- `_orig_*` attributes store original functions for call-through
- Patches apply at module load time via `sitecustomize.py` on PYTHONPATH

### Directory Structure

```
litellm_ext/
├── bootstrap.py           # Entry point: install_extensions()
├── sitecustomize.py       # Auto-loaded on PYTHONPATH
├── core/                  # Patching infrastructure
│   ├── registry.py        # HTTPX mutator registry (priority-ordered)
│   ├── config.py          # YAML config loader with hot-reload
│   └── settings.py        # PatchSettings dataclass
├── extensions/            # Modular extensions (installed by priority)
│   ├── __init__.py        # Extension install ordering
│   ├── httpx/             # Request/response mutators
│   ├── asgi/              # Middleware patches
│   └── litellm/           # Direct LiteLLM patches
├── policy/                # Token estimation and message trimming
└── adapters/              # Schema conversion (Anthropic <-> OpenAI)
```

### Extension Priority Order

Extensions install in this order (lower number = earlier):
1. `suppress_warnings` (5)
2. `count_tokens_stub` (10)
3. `hard_caps` (20)
4. `local_token_counter` (25)
5. `reasoning_replay` (30)
6. `hard_caps_messages` (35)
7. `stream_usage_rewrite` (40)
8. `streaming_sse` (50)
9. `transform` (60)

### HTTPX Mutator Registry

```python
# Register request mutators (modify requests before send)
register_request_mutator(mutator, priority=100)

# Register response mutators (modify responses after receive)
register_response_mutator(mutator, priority=100)      # sync
register_async_response_mutator(mutator, priority=100) # async
```

Mutators run in priority order. Lower priority = runs first.

## Configuration

Two YAML files control behavior:
- `config/litellm.yaml` - Model definitions and provider settings
- `config/extensions.yaml` - Extension enable/disable and settings

Extension settings pattern:
```yaml
extensions:
  hard_caps:
    enabled: true
    debug: false
policy:
  overflow_policy: "reduce_then_trim"
  safety_buffer_tokens: 4096
  model_limits:
    '*':
      max_output: 20000
      max_context: 180000
    deepseek-chat:
      max_output: 8192
      max_context: 131072
```

Environment overrides: `LITELLM_EXT_<EXTENSION_NAME>` enables/disables extensions.

## Key Files

| File | Purpose |
|------|---------|
| `litellm_ext/bootstrap.py` | Entry point for installing all extensions |
| `litellm_ext/core/registry.py` | Central HTTPX mutator registry |
| `litellm_ext/core/config.py` | YAML config loader with fallback paths |
| `litellm_ext/core/model_alias.py` | Model alias resolution and caching |
| `litellm_ext/policy/engine.py` | Token estimation, message trimming logic |
| `litellm_ext/extensions/__init__.py` | Extension installation with priority |
| `litellm_ext/agent_config/cli.py` | Claude Code settings sync CLI |
| `tests/conftest.py` | Test isolation: reset config, clear mutators |

## Test Isolation

Tests must be isolated since patches are global. `tests/conftest.py` handles:
- Resetting config between tests
- Clearing httpx mutators
- Dropping cached patch modules
- Using temp directory for config via `LITELLM_EXT_CONFIG_PATH`

## Environment Variables

**Required:**
- `LITELLM_MASTER_KEY` - Proxy authentication (starts with "sk-")
- Provider keys: `DEEPSEEK_API_KEY`, `ZAI_API_KEY`, `MOONSHOT_API_KEY`, `ARK_API_KEY`, `QWEN_API_KEY`

**Optional:**
- `DATABASE_URL` - For admin UI mode (`litellm-up-db`)
- `LITELLM_CONFIG` - Path to litellm.yaml
- `LITELLM_PATCH_DIR` - Directory containing this repo (for PYTHONPATH)
- `LITELLM_EXT_CONFIG_PATH` - Override extensions.yaml location
- `LITELLM_EXT_CLAUDE_CONFIG` - Path to Claude Code settings (default: config/claude.settings.json)
- `LITELLM_LOCAL_MODEL_COST_MAP` - Set to "true" to use local bundled model cost map only

## Supported Models

DeepSeek, Zhipu/GLM (glm-4.7, glm-5), Moonshot/Kimi (kimi-k2.5), ByteDance/Doubao, Alibaba/Qwen (qwen3-max, qwen3-coder-plus).

## Development Notes

- Python >= 3.12 required
- Uses `from __future__ import annotations` throughout
- Type hints on all public functions
- Silent failures for expected missing modules during bootstrap
- When adding new extensions, update `extensions/__init__.py` with priority
- The `agent-config-apply` CLI syncs Claude Code settings to LiteLLM config on startup
