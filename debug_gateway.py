#!/usr/bin/env python3
"""
本地调试入口：在 Cursor / VS Code 里运行本文件，等价于 `nanobot gateway`。

示例（终端）：
  python debug_gateway.py
  python debug_gateway.py -p 8765 -c C:\\Users\\你\\.nanobot\\config.json
  python debug_gateway.py -w D:\\code\\my-workspace -v
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="nanobot gateway run for IDE debugging")
    p.add_argument("-p", "--port", type=int, default=None, help="Gateway port")
    p.add_argument("-w", "--workspace", default=None, help="Workspace directory")
    p.add_argument("-c", "--config", default=None, help="Path to config file")
    p.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    # 从仓库根运行脚本时，确保可导入本地 nanobot 源码
    repo_root = Path(__file__).resolve().parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    from nanobot.cli.commands import gateway

    gateway(
        port=args.port,
        workspace=args.workspace,
        verbose=args.verbose,
        config=args.config,
    )


if __name__ == "__main__":
    main()
