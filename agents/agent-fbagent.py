#!/usr/bin/env python3
"""agent_benchmark entry point for fbagent (fixed token budget per step).

    agent-fbagent.py [--model M --endpoint E --budget B] --non-interactive "<task>"

Defaults to Claude Haiku through the local run:// shim.
"""

from __future__ import annotations

import os
import sys

from fbagent.__main__ import main

if __name__ == "__main__":
    argv = sys.argv[1:]
    if "--endpoint" not in argv:
        argv = ["--model", "claude-haiku",
                "--endpoint", f"run://{os.path.expanduser('~')}/bin/claude-haiku-completions.py",
                *argv]
    main(argv)
