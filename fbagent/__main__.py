"""Command-line entry point: ``fbagent TASK``."""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

import agentknit

from ._loop import run


def _print_event(kind: str, data: dict[str, Any]) -> None:
    if kind == "fb_step":
        print(f"🧠 [{data['step']}] act prompt={data['act_prompt_tokens']} "
              f"cached={data['act_cached_tokens']} | summary prompt={data['summary_prompt_tokens']} "
              f"cached={data['summary_cached_tokens']} | raw={data['raw_step_chars']}ch "
              f"→ digest={data['digest_tokens']}tok", file=sys.stderr)
        print(f"   {data['digest']}", file=sys.stderr)
    elif kind not in ("fb_final", "final_answer"):
        agentknit._default_event_handler(kind, data)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="fbagent", description=__doc__)
    p.add_argument("task")
    p.add_argument("--model", default="claude-haiku")
    p.add_argument("--endpoint", required=True,
                   help="OpenAI-compatible endpoint or run:///path/to/shim")
    p.add_argument("--budget", type=int, default=200, help="max digest tokens per step")
    p.add_argument("--max-steps", type=int, default=20)
    p.add_argument("--json", action="store_true", help="print the result as JSON on stdout")
    args = p.parse_args(argv)

    if args.json:  # exec_shell echoes live to stdout by default
        agentknit.set_tool_output_stream(sys.stderr)
    schema = agentknit.load_specification(args.model, args.endpoint)
    client = agentknit.create_client(schema)
    res = run(client, schema.get("model", args.model), args.task, budget=args.budget,
              max_steps=args.max_steps, schema=schema,
              on_event=None if args.json else _print_event)
    if args.json:
        json.dump({"final_reply": res.final_reply,
                   "final_prompt_tokens": res.final_prompt_tokens,
                   "steps": [s.__dict__ for s in res.steps]}, sys.stdout, indent=2)
        print()
    else:
        print(f"\n» {res.final_reply}" if res.final_reply is not None
              else f"⚠️ no final answer after {args.max_steps} steps")
    sys.exit(0 if res.final_reply is not None else 1)


if __name__ == "__main__":
    main()
