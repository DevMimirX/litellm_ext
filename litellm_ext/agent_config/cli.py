from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Sequence

from ..core.logging import format_log_line
from .adapters import ADAPTERS, AgentConfigAdapter
from .engine import ApplyOptions, apply_json_config


_LOGGER_NAME = "litellm_ext.agent_config_apply"


def log(message: str, *, quiet: bool = False) -> None:
    if quiet:
        return
    print(format_log_line(_LOGGER_NAME, message), file=sys.stderr, flush=True)


def _env_flag(name: str) -> bool:
    value = os.environ.get(name, "")
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _resolve_option_flag(args_value: bool, adapter: AgentConfigAdapter, name: str) -> bool:
    global_env = f"LITELLM_EXT_AGENT_CONFIG_{name}"
    tool_env = f"{adapter.option_env_prefix}_{name}"
    return args_value or _env_flag(global_env) or _env_flag(tool_env)


def _resolve_strategy(args_strategy: str, adapter: AgentConfigAdapter) -> str:
    global_env = os.environ.get("LITELLM_EXT_AGENT_CONFIG_STRATEGY", "").strip().lower()
    tool_env = os.environ.get(f"{adapter.option_env_prefix}_STRATEGY", "").strip().lower()
    strategy = tool_env or global_env or args_strategy
    if strategy not in {"overwrite", "merge"}:
        raise ValueError(
            f"Invalid strategy '{strategy}'. Use 'overwrite' or 'merge'."
        )
    return strategy


def _build_parser(*, prog: str | None, description: str | None) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=prog,
        description=description or "Apply agent CLI settings from repo-managed config files.",
    )
    parser.add_argument(
        "--tool",
        choices=sorted(ADAPTERS.keys()),
        help="Agent tool adapter to apply (e.g. claude)",
    )
    parser.add_argument(
        "--list-tools",
        action="store_true",
        help="List available agent config adapters",
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="Path to source config file (overrides adapter defaults)",
    )
    parser.add_argument(
        "--target",
        type=Path,
        help="Path to target settings file (overrides adapter defaults)",
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Skip creating backup of existing target settings",
    )
    parser.add_argument(
        "--strategy",
        choices=("overwrite", "merge"),
        default="overwrite",
        help="How to apply source config into target settings",
    )
    parser.add_argument(
        "--if-changed",
        action="store_true",
        help="Write only when resulting target differs from current target",
    )
    parser.add_argument(
        "--optional",
        action="store_true",
        help="Exit successfully when source config is missing",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress non-error logs",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without making changes",
    )
    return parser


def _print_tools() -> None:
    for adapter in sorted(ADAPTERS.values(), key=lambda a: a.name):
        print(f"{adapter.name}\t{adapter.description}")


def _resolve_adapter(args: argparse.Namespace) -> AgentConfigAdapter:
    tool = args.tool
    if tool is None:
        raise ValueError("Missing required --tool. Use --list-tools to inspect available adapters.")
    adapter = ADAPTERS.get(tool)
    if adapter is None:
        raise ValueError(f"Unsupported tool adapter: {tool}")
    return adapter


def _run_for_adapter(
    adapter: AgentConfigAdapter,
    args: argparse.Namespace,
) -> int:
    quiet = _resolve_option_flag(args.quiet, adapter, "QUIET")
    optional = _resolve_option_flag(args.optional, adapter, "OPTIONAL")
    if_changed = _resolve_option_flag(args.if_changed, adapter, "IF_CHANGED")
    strategy = _resolve_strategy(args.strategy, adapter)

    source_path = adapter.resolve_source(args.config, quiet=quiet)
    if source_path is None:
        if optional:
            log(f"{adapter.name}: source config missing, skipping.", quiet=quiet)
            return 0
        print(
            f"Error: Could not find source config for tool '{adapter.name}'. "
            "Provide --config or set adapter env var.",
            file=sys.stderr,
        )
        return 1

    target_path = adapter.resolve_target(args.target)
    options = ApplyOptions(
        backup=not args.no_backup,
        strategy=strategy,
        if_changed=if_changed,
        quiet=quiet,
    )
    log(
        f"start tool={adapter.name} source={source_path} target={target_path} "
        f"strategy={strategy} if_changed={if_changed} backup={options.backup}",
        quiet=quiet,
    )

    if args.dry_run:
        print(f"Tool: {adapter.name}")
        print(f"Would apply: {source_path}")
        print(f"Target: {target_path}")
        print(f"Strategy: {strategy}")
        print(f"Write only if changed: {if_changed}")
        if target_path.exists() and not args.no_backup:
            print(f"Would backup to: {target_path.with_suffix('.json.bak')}")
        return 0

    try:
        result = apply_json_config(source_path, target_path, options=options)
    except (ValueError, FileNotFoundError, OSError) as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    log(
        f"complete tool={adapter.name} changed={result.changed} target={result.target_path}",
        quiet=quiet,
    )

    return 0


def main(
    argv: Sequence[str] | None = None,
    prog: str | None = None,
    description: str | None = None,
) -> int:
    parser = _build_parser(prog=prog, description=description)
    args = parser.parse_args(argv)

    if args.list_tools:
        _print_tools()
        return 0

    try:
        adapter = _resolve_adapter(args)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    return _run_for_adapter(adapter, args)


if __name__ == "__main__":
    sys.exit(main())
