#!/usr/bin/env python3
"""Benchmark-harness entry point for fbagent (fixed token budget per step).

    agent-fbagent.py [--model M --endpoint E --budget B] --non-interactive "<task>"

Without ``--endpoint``, the model and endpoint come from ``$FBAGENT_MODEL`` and
``$FBAGENT_ENDPOINT`` (an OpenAI-compatible base URL or a ``run://`` shim).
"""

from __future__ import annotations

import os
import sys

from fbagent.__main__ import main

if __name__ == "__main__":
    argv = sys.argv[1:]
    if "--endpoint" not in argv:
        endpoint = os.environ.get("FBAGENT_ENDPOINT")
        if not endpoint:
            sys.exit("agent-fbagent: pass --endpoint or set $FBAGENT_ENDPOINT")
        argv = ["--model", os.environ.get("FBAGENT_MODEL", "default"),
                "--endpoint", endpoint, *argv]
    main(argv)
