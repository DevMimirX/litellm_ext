# LiteLLM Extension

A proxy layer for big model access with LiteLLM.

## Setup with uv

This project uses `uv` for dependency management. Follow these steps to set up the environment:

### Prerequisites

- Install `uv`: `curl -LsSf https://astral.sh/uv/install.sh | sh`
- Make sure uv is in your PATH: `export PATH="$HOME/.local/bin:$PATH"`

### Quick Setup

```bash
# Clone the repository
git clone <repository-url>
cd litellm_ext

# Create and activate the uv environment
uv sync

# Activate the environment
source .venv/bin/activate  # On Unix/macOS
.venv\Scripts\activate     # On Windows
```

### Development Setup

```bash
# Install development dependencies
uv add --dev pytest pytest-asyncio fastapi starlette requests

# Run tests
uv run pytest

# Run linting
uv run ruff check .
uv run mypy .
```

### Running the Proxy

#### 1. Set API Keys and Configuration

```bash
# Keys for LiteLLM + providers
export DEEPSEEK_API_KEY=""
export ZAI_API_KEY=""
export MOONSHOT_API_KEY=""
export QWEN_API_KEY=""
export ARK_API_KEY=""
export LITELLM_MASTER_KEY=""

# ---- LiteLLM helpers ----
export LITELLM_CONFIG="$HOME/litellm_ext/config/litellm.yaml"
export LITELLM_PATCH_DIR="$HOME/litellm_ext/"
export LITELLM_DATABASE_URL="postgresql://litellm:litellm@localhost:5432/litellm"
export LITELLM_EXT_CLAUDE_CONFIG="$HOME/litellm_ext/config/claude.settings.json"

# Add to your ~/.zshrc or ~/.bashrc:
unalias litellm-up litellm-up-db litellm-down 2>/dev/null || true

# Stop LiteLLM
litellm-down() {
  command pkill -f "litellm.*--config" >/dev/null 2>&1 || true
  command pkill -f "/bin/litellm" >/dev/null 2>&1 || true
}

# Internal helper: ensure port 4000 is free, otherwise fail fast.
_litellm_assert_port_4000_free() {
  local pids
  pids="$(lsof -tiTCP:4000 -sTCP:LISTEN 2>/dev/null || true)"
  if [[ -n "$pids" ]]; then
    echo "Port 4000 is busy (PID(s): $pids). Stop that process first."
    return 1
  fi
  return 0
}

# Start LiteLLM (NO DB + NO UI) - fastest boot + patch loaded
litellm-up() {
  litellm-down
  _litellm_assert_port_4000_free || return 1
  # Apply Claude Code settings before starting
  uv run --directory "$LITELLM_PATCH_DIR" \
    agent-config-apply --tool claude --strategy overwrite --if-changed --optional --quiet --no-backup
  env -u DATABASE_URL \
    PYTHONPATH="$LITELLM_PATCH_DIR:$PYTHONPATH" \
    DISABLE_ADMIN_UI="True" NO_DOCS="True" NO_REDOC="True" \
    uv run --directory "$LITELLM_PATCH_DIR" \
    litellm --host 0.0.0.0 --port 4000 --config "$LITELLM_CONFIG"
}

# Start LiteLLM with DB/UI (opt-in)
litellm-up-db() {
  litellm-down
  _litellm_assert_port_4000_free || return 1
  # Apply Claude Code settings before starting
  uv run --directory "$LITELLM_PATCH_DIR" \
    agent-config-apply --tool claude --strategy overwrite --if-changed --optional --quiet --no-backup
  DATABASE_URL="$LITELLM_DATABASE_URL" \
    PYTHONPATH="$LITELLM_PATCH_DIR:$PYTHONPATH" \
    DISABLE_ADMIN_UI="False" \
    uv run --directory "$LITELLM_PATCH_DIR" \
    litellm --host 0.0.0.0 --port 4000 --config "$LITELLM_CONFIG"
}

# Usage:
# litellm-up              # Start with minimal features (faster)
# litellm-up-db           # Start with admin UI and database
# uv run --directory "$LITELLM_PATCH_DIR" agent-config-apply --list-tools  # List supported CLI adapters
# uv run --directory "$LITELLM_PATCH_DIR" agent-config-apply --tool claude --dry-run --strategy overwrite  # Generic entrypoint
# uv run --directory "$LITELLM_PATCH_DIR" agent-config-apply --tool claude --strategy overwrite --if-changed --optional --quiet --no-backup  # Safe startup sync
```

#### 2. Start the Proxy

```bash
# Set your API keys first
export DEEPSEEK_API_KEY="your-deepseek-key"
export ZAI_API_KEY="your-zai-key"
export MOONSHOT_API_KEY="your-moonshot-key"
export ARK_API_KEY="your-ark-key"
export QWEN_API_KEY="your-qwen-key"
export LITELLM_MASTER_KEY="your-master-key"
# Optional: only needed for DB/UI mode (`litellm-up-db`)
export DATABASE_URL="sqlite:///litellm.db"

# Add shell functions to your shell profile (zshrc/bashrc)
# (copy the functions block above)

# Start the proxy
litellm-up
```

#### 2a. DB/UI prerequisites (`litellm-up-db`)

If you want to run with DB + Admin UI (`litellm-up-db`), initialize Prisma client binaries and ensure a PostgreSQL image is available locally:

```bash
# Pull PostgreSQL image (official image; optional slimmer tag: postgres:16-alpine)
docker pull postgres:16

# Start local PostgreSQL matching LITELLM_DATABASE_URL
docker run -d --name litellm-postgres \
  -e POSTGRES_USER=litellm \
  -e POSTGRES_PASSWORD=litellm \
  -e POSTGRES_DB=litellm \
  -p 5432:5432 \
  postgres:16

# Generate Prisma client binaries for LiteLLM's bundled schema
LITELLM_REPO_DIR="${LITELLM_PATCH_DIR:-$HOME/litellm_ext}"
LITELLM_PRISMA_SCHEMA="$(uv run --directory "$LITELLM_REPO_DIR" python -c "import pathlib, litellm; print(pathlib.Path(litellm.__file__).resolve().parent / 'proxy' / 'schema.prisma')")"
uv run --directory "$LITELLM_REPO_DIR" prisma generate --schema "$LITELLM_PRISMA_SCHEMA"
```

If the container already exists, use `docker start litellm-postgres` instead of `docker run`.

### Configuration

The proxy is configured via:
- `config/litellm.yaml` - Model definitions
- `config/extensions.yaml` - Extension settings
- `config/claude.settings.json` - Claude Code runtime settings (applied via `agent-config-apply`)

Example `config/claude.settings.json`:

```json
{
  "env": {
    "ANTHROPIC_BASE_URL": "http://127.0.0.1:4000",
    "ANTHROPIC_API_KEY": "sk-litellm",
    "ANTHROPIC_DEFAULT_SONNET_MODEL": "kimi-k2.5-ali",
    "ANTHROPIC_DEFAULT_OPUS_MODEL": "glm-5-ali",
    "ANTHROPIC_DEFAULT_HAIKU_MODEL": "MiniMax-M2.5-ali",
    "CLAUDE_CODE_SUBAGENT_MODEL": "qwen3.5-plus",
    "DISABLE_TELEMETRY": "1",
    "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
    "CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS": "1",
    "CLAUDE_CODE_MAX_OUTPUT_TOKENS": "20000",
    "CLAUDE_AUTOCOMPACT_PCT_OVERRIDE": "95"
  },
  "includeCoAuthoredBy": false,
  "model": "opus"
}
```

### Available Models

The proxy supports multiple providers:
- DeepSeek
- Zhipu/GLM
- Moonshot/Kimi
- ByteDance/Doubao
- Alibaba Cloud/Qwen

### Environment Variables

Required environment variables:
- `LITELLM_MASTER_KEY` - Master key for proxy auth (starts with "sk-")
- Provider-specific API keys (see config files)

Optional environment variables:
- `DATABASE_URL` - Required only for `litellm-up-db` / admin UI mode

### Setup with uv

1. **Install uv package manager** (if not already installed):
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   export PATH="$HOME/.local/bin:$PATH"
   ```

2. **Initialize uv environment and install dependencies**:
   ```bash
   uv sync  # Creates .venv and installs all dependencies
   ```

3. **Activate the uv virtual environment**:
   ```bash
   source .venv/bin/activate  # On Unix/macOS
   .venv\Scripts\activate     # On Windows
   ```

4. **Install LiteLLM with proxy support**:
   ```bash
   uv pip install -U "litellm[proxy]"
   ```

5. **Set your environment variables**:
   ```bash
   export DEEPSEEK_API_KEY="your-deepseek-key"
   export ZAI_API_KEY="your-zai-key"
   export MOONSHOT_API_KEY="your-moonshot-key"
   export ARK_API_KEY="your-ark-key"
   export QWEN_API_KEY="your-qwen-key"
   export LITELLM_MASTER_KEY="your-master-key"
   # Optional: only needed for DB/UI mode (`litellm-up-db`)
   export DATABASE_URL="sqlite:///litellm.db"
   ```

Your uv environment is now ready to run the LiteLLM proxy!

### Manual Setup

Add the following shell functions to your shell profile (`~/.zshrc`, `~/.bashrc`, or `$PROFILE`):

#### Linux/macOS (Zsh/Bash)

```bash
# Add these to ~/.zshrc or ~/.bashrc

# Keys for LiteLLM + providers
export DEEPSEEK_API_KEY=""
export ZAI_API_KEY=""
export MOONSHOT_API_KEY=""
export QWEN_API_KEY=""
export ARK_API_KEY=""
export LITELLM_MASTER_KEY=""

# ---- LiteLLM helpers ----
export LITELLM_CONFIG="$HOME/litellm_ext/config/litellm.yaml"
export LITELLM_PATCH_DIR="$HOME/litellm_ext/"
export LITELLM_DATABASE_URL="postgresql://litellm:litellm@localhost:5432/litellm"
export LITELLM_EXT_CLAUDE_CONFIG="$HOME/litellm_ext/config/claude.settings.json"
export LITELLM_LOCAL_MODEL_COST_MAP="true"  # Use local bundled model cost map only

unalias litellm-up litellm-up-db litellm-down 2>/dev/null || true

# Stop LiteLLM
litellm-down() {
  command pkill -f "litellm.*--config" >/dev/null 2>&1 || true
  command pkill -f "/bin/litellm" >/dev/null 2>&1 || true
}

# Internal helper: ensure port 4000 is free, otherwise fail fast.
_litellm_assert_port_4000_free() {
  local pids
  pids="$(lsof -tiTCP:4000 -sTCP:LISTEN 2>/dev/null || true)"
  if [[ -n "$pids" ]]; then
    echo "Port 4000 is busy (PID(s): $pids). Stop that process first."
    return 1
  fi
  return 0
}

# Start LiteLLM (NO DB + NO UI) - fastest boot + patch loaded
litellm-up() {
  litellm-down
  _litellm_assert_port_4000_free || return 1
  # Apply Claude Code settings before starting
  uv run --directory "$LITELLM_PATCH_DIR" \
    agent-config-apply --tool claude --strategy overwrite --if-changed --optional --quiet --no-backup
  env -u DATABASE_URL \
    PYTHONPATH="$LITELLM_PATCH_DIR:$PYTHONPATH" \
    DISABLE_ADMIN_UI="True" NO_DOCS="True" NO_REDOC="True" \
    uv run --directory "$LITELLM_PATCH_DIR" \
    litellm --host 0.0.0.0 --port 4000 --config "$LITELLM_CONFIG"
}

# Start LiteLLM with DB/UI (opt-in)
litellm-up-db() {
  litellm-down
  _litellm_assert_port_4000_free || return 1
  # Apply Claude Code settings before starting
  uv run --directory "$LITELLM_PATCH_DIR" \
    agent-config-apply --tool claude --strategy overwrite --if-changed --optional --quiet --no-backup
  DATABASE_URL="$LITELLM_DATABASE_URL" \
    PYTHONPATH="$LITELLM_PATCH_DIR:$PYTHONPATH" \
    DISABLE_ADMIN_UI="False" \
    uv run --directory "$LITELLM_PATCH_DIR" \
    litellm --host 0.0.0.0 --port 4000 --config "$LITELLM_CONFIG"
}
```

#### Windows (PowerShell)

```powershell
# Add these to your PowerShell profile ($PROFILE)

# ----------------------------
# LiteLLM + Provider Keys
# ----------------------------
$env:DEEPSEEK_API_KEY = ""
$env:ZAI_API_KEY      = ""
$env:MOONSHOT_API_KEY = ""
$env:QWEN_API_KEY      = ""
$env:ARK_API_KEY       = ""
$env:LITELLM_MASTER_KEY = "sk-litellm"

# ----------------------------
# LiteLLM helpers
# ----------------------------
$env:LITELLM_CONFIG     = "$env:USERPROFILE\litellm_ext\config\litellm.yaml"
$env:LITELLM_PATCH_DIR  = "$env:USERPROFILE\litellm_ext"
$env:LITELLM_DATABASE_URL = "postgresql://litellm:litellm@localhost:5432/litellm"
$env:LITELLM_EXT_CLAUDE_CONFIG = "$env:USERPROFILE\litellm_ext\config\claude.settings.json"
$env:LITELLM_LOCAL_MODEL_COST_MAP = "true"  # Use local bundled model cost map only

function litellm-up {
    # Stop any existing LiteLLM processes
    litellm-down

    # Check if port 4000 is available
    $port4000 = netstat -ano | findstr ":4000 "
    if ($port4000) {
        Write-Host "Port 4000 is busy. Stop that process first." -ForegroundColor Red
        return 1
    }

    # Apply Claude Code settings before starting
    uv run --directory "$env:LITELLM_PATCH_DIR" `
        agent-config-apply --tool claude --strategy overwrite --if-changed --optional --quiet --no-backup | Out-Null

    # NO DB + NO UI = fastest boot
    $env:DISABLE_ADMIN_UI = "True"
    $env:NO_DOCS  = "True"
    $env:NO_REDOC = "True"

    # Don't set DATABASE_URL to force "no DB"
    Remove-Item Env:\DATABASE_URL -ErrorAction SilentlyContinue

    # Patch dir prepended to PYTHONPATH
    $env:PYTHONPATH = "$env:LITELLM_PATCH_DIR;$env:PYTHONPATH"

    uv run --directory "$env:LITELLM_PATCH_DIR" litellm --config "$env:LITELLM_CONFIG" --host 0.0.0.0 --port 4000
}

function litellm-up-db {
    # Stop any existing LiteLLM processes
    litellm-down

    # Check if port 4000 is available
    $port4000 = netstat -ano | findstr ":4000 "
    if ($port4000) {
        Write-Host "Port 4000 is busy. Stop that process first." -ForegroundColor Red
        return 1
    }

    # Apply Claude Code settings before starting
    uv run --directory "$env:LITELLM_PATCH_DIR" `
        agent-config-apply --tool claude --strategy overwrite --if-changed --optional --quiet --no-backup | Out-Null

    # DB + UI enabled
    $env:DISABLE_ADMIN_UI = "False"
    $env:NO_DOCS  = "False"
    $env:NO_REDOC = "False"

    $env:DATABASE_URL = $env:LITELLM_DATABASE_URL
    $env:PYTHONPATH = "$env:LITELLM_PATCH_DIR;$env:PYTHONPATH"

    uv run --directory "$env:LITELLM_PATCH_DIR" litellm --config "$env:LITELLM_CONFIG" --host 0.0.0.0 --port 4000
}

function litellm-down {
    # Kill any LiteLLM processes
    Get-Process -ErrorAction SilentlyContinue |
        Where-Object {
            ($_.ProcessName -eq "litellm") -or
            ($_.Path -match "litellm") -or
            ($_.CommandLine -match "litellm.*--config")
        } |
        ForEach-Object {
            try {
                Stop-Process -Id $_.Id -Force -ErrorAction SilentlyContinue
            } catch {}
        }
}
```

After adding to your profile, reload it and then use:
- `litellm-up` - Start with minimal features (faster)
- `litellm-up-db` - Start with admin UI and database
- `litellm-down` - Stop LiteLLM
- `uv run --directory "$LITELLM_PATCH_DIR" agent-config-apply --list-tools` - List supported CLI config adapters
- `uv run --directory "$LITELLM_PATCH_DIR" agent-config-apply --tool claude --dry-run --strategy overwrite` - Generic adapter invocation
- `uv run --directory "$LITELLM_PATCH_DIR" agent-config-apply --tool claude --strategy overwrite --if-changed --optional --quiet --no-backup` - Safe startup sync

### Troubleshooting

- If you see "Failed to fetch remote model cost map" warnings, ensure this patch dir is on `PYTHONPATH` and set `LITELLM_LOCAL_MODEL_COST_MAP=true`
- Make sure all required API keys are set in your environment
- Check the configuration files for any syntax errors
- If you encounter `importlib.resources` errors, verify you're on current code/dependencies (`uv sync`) and that startup uses this repo via `PYTHONPATH`
- If startup fails with `Unable to find Prisma binaries. Please run 'prisma generate' first.`, run:
  `LITELLM_REPO_DIR="${LITELLM_PATCH_DIR:-$HOME/litellm_ext}" && LITELLM_PRISMA_SCHEMA="$(uv run --directory "$LITELLM_REPO_DIR" python -c "import pathlib, litellm; print(pathlib.Path(litellm.__file__).resolve().parent / 'proxy' / 'schema.prisma')")" && uv run --directory "$LITELLM_REPO_DIR" prisma generate --schema "$LITELLM_PRISMA_SCHEMA"`
