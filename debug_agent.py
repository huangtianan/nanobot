#!/usr/bin/env python3
"""
本地调试入口：在 Cursor / VS Code 里打开本文件，对 nanobot 源码下断点后「运行 / 调试当前文件」即可。

前置：在仓库根目录已执行 `pip install -e .`，并激活同一 venv。

示例（终端）：
  python debug_agent.py
  python debug_agent.py -m "你好" -c C:\\Users\\你\\.nanobot\\config.json
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="nanobot one-shot agent run for IDE debugging")
    p.add_argument("-m", "--message", default="帮我查下广州今天的天气", help="发给 agent 的内容")
    p.add_argument(
        "-c",
        "--config",
        default=None,
        help="config.json 路径；省略则使用 nanobot 默认配置路径",
    )
    p.add_argument("-w", "--workspace", default=None, help="覆盖配置里的 workspace 目录")
    p.add_argument(
        "-s",
        "--session",
        default="cli:direct",
        help="会话 key，默认 cli:direct",
    )
    p.add_argument(
        "--quiet-logs",
        action="store_true",
        help="关闭 nanobot 包内 loguru 日志（默认开启，便于看 Tool call 等）",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    # 从仓库根运行脚本时，确保能 import 到已安装的 nanobot（推荐 pip install -e .）
    repo_root = Path(__file__).resolve().parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    from loguru import logger

    from nanobot.agent.loop import AgentLoop
    from nanobot.bus.queue import MessageBus
    from nanobot.cli.commands import _load_runtime_config, _make_provider
    from nanobot.config.paths import get_cron_dir
    from nanobot.cron.service import CronService
    from nanobot.utils.helpers import sync_workspace_templates

    if args.quiet_logs:
        logger.disable("nanobot")
    else:
        logger.enable("nanobot")

    cfg = _load_runtime_config(args.config, workspace=args.workspace)
    sync_workspace_templates(cfg.workspace_path)

    bus = MessageBus()
    provider = _make_provider(cfg)
    cron = CronService(get_cron_dir() / "jobs.json")

    agent_loop = AgentLoop(
        bus=bus,
        provider=provider,
        workspace=cfg.workspace_path,
        model=cfg.agents.defaults.model,
        max_iterations=cfg.agents.defaults.max_tool_iterations,
        context_window_tokens=cfg.agents.defaults.context_window_tokens,
        web_search_config=cfg.tools.web.search,
        web_proxy=cfg.tools.web.proxy or None,
        exec_config=cfg.tools.exec,
        cron_service=cron,
        restrict_to_workspace=cfg.tools.restrict_to_workspace,
        mcp_servers=cfg.tools.mcp_servers,
        channels_config=cfg.channels,
        data_agent_config=cfg.tools.data_agent,
    )

    async def run() -> str:
        try:
            # 在此处或 AgentLoop / provider 内下断点即可单步
            return await agent_loop.process_direct(
                args.message,
                session_key=args.session,
                on_progress=None,
            )
        finally:
            await agent_loop.close_mcp()

    text = asyncio.run(run())
    print(text or "(empty reply)")


if __name__ == "__main__":
    main()
