"""Fixed-budget-per-step agent, as an agentknit ``step_reducer``.

agentknit runs the loop (tools, hooks, journal, events, sandbox).  After each
tool step, the reducer asks the model — via ``side_query``, which reuses the
cached prefix of the step's own request — to summarize the step in at most
``budget`` tokens, and replaces the raw step by that digest.  After k steps
the context is ``prefix + k * (budget + c)`` tokens at most.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import agentknit
from agentknit import StepReduction

SYSTEM_PROMPT = """\
Your memory works in steps: after every step, the raw tool calls and tool
outputs are discarded and replaced by a digest of at most {budget} tokens that
you write yourself. Past steps appear as "[memory of your step N]" messages:
those actions really happened, their effects are on disk, never redo them.
File contents you read do not survive the step, so prefer write_file over
str_replace for edits. Batch independent tool calls into one step.
When the task is complete, reply with the final answer and no tool call."""

SUMMARIZE_PROMPT = """\
[memory] Step {step} is over. Its tool calls and outputs above will be
DISCARDED and replaced by your digest. Write the digest now: HARD LIMIT
{budget} tokens, aim for {words} words, anything longer is cut off. Keep only
what future steps need: what you did, key facts learned (exact values, errors,
command outputs that matter; relative paths), and what remains. If the task is
finished, end with "DONE: <final answer>". No preamble, do not call tools."""

MEMORY_TEMPLATE = """\
[memory of your step {step}, already executed — its effects are on disk]
{digest}

Do the next remaining action with a tool call, or give the final answer."""

DONE_MARKER = "DONE:"

EventHandler = Callable[[str, dict[str, Any]], None]


@dataclass
class StepRecord:
    step: int
    tool_calls: list[str]
    act_prompt_tokens: int
    act_cached_tokens: int
    summary_prompt_tokens: int
    summary_cached_tokens: int
    digest_tokens: int
    raw_step_chars: int
    digest: str


@dataclass
class Result:
    final_reply: str | None
    steps: list[StepRecord] = field(default_factory=list)
    messages: list[dict[str, Any]] = field(default_factory=list)
    final_prompt_tokens: int = 0
    usage_totals: dict[str, int] = field(default_factory=dict)  # act + digest calls


def _default_schema(model: str) -> dict[str, Any]:
    specs, dispatch = agentknit.default_tool_spec()
    return {"model": model, "endpoint": "", "tool_specs": specs, "tool_dispatch": dispatch}


def run(client: Any, model: str, task: str, *, budget: int = 200,
        max_steps: int = 20, schema: dict[str, Any] | None = None,
        cwd: str | None = None, on_event: EventHandler | None = None) -> Result:
    """Run *task* to completion with a context growing by ≤ *budget* tokens per step.

    *schema* is an agentknit spec (from ``load_specification``); by default a
    minimal one with agentknit's default tools.  *on_event* receives every
    agentknit event plus ``fb_step`` / ``fb_final``.
    """
    emit = on_event or (lambda _t, _d: None)
    result = Result(final_reply=None)
    act_usage: list[dict[str, Any]] = []
    side_usage: list[dict[str, Any]] = []
    exhausted = False

    def on_usage(_t: str, data: dict[str, Any]) -> None:
        (side_usage if data.get("purpose") == "side_query" else act_usage).append(data)

    def reducer(session: Any, step: list[dict[str, Any]], *, client: Any,
                model: str) -> StepReduction:
        nonlocal exhausted
        n = len(result.steps) + 1
        side_usage.clear()
        digest = agentknit.side_query(
            client, model, session,
            SUMMARIZE_PROMPT.format(step=n, budget=budget, words=int(budget * 0.4)),
            max_tokens=budget, preamble=None, count_usage=True)
        calls = [tc["function"]["name"] for tc in step[0].get("tool_calls") or []]
        if not digest:  # the endpoint ignored tool_choice=none
            digest = "called " + ", ".join(calls)
        act = act_usage[-1] if act_usage else {}
        summ = side_usage[-1] if side_usage else {}
        rec = StepRecord(
            step=n, tool_calls=calls,
            act_prompt_tokens=act.get("prompt", 0), act_cached_tokens=act.get("cached", 0),
            summary_prompt_tokens=summ.get("prompt", 0),
            summary_cached_tokens=summ.get("cached", 0),
            digest_tokens=summ.get("completion", 0),
            raw_step_chars=sum(len(str(m.get("content") or "")) + len(str(m.get("tool_calls") or ""))
                               for m in step),
            digest=digest)
        result.steps.append(rec)
        emit("fb_step", dict(rec.__dict__))
        # User role, not assistant: models imitate their own past messages and
        # would reply with a digest instead of acting.
        kept = [{"role": "user", "content": MEMORY_TEMPLATE.format(step=n, digest=digest)}]
        # The digest is the model's own judgement of the step: trust its DONE,
        # weaker models otherwise keep re-verifying forever.
        if DONE_MARKER in digest:
            return StepReduction(kept, final_reply=digest.split(DONE_MARKER, 1)[1].strip())
        if n >= max_steps:
            exhausted = True
            return StepReduction(kept, final_reply=f"[stopped after {n} steps] {digest}")
        return StepReduction(kept)

    session = agentknit.init_session(
        schema or _default_schema(model), non_interactive=True,
        system_prompt_supplement=f"Working directory: {cwd or os.getcwd()}\n\n"
                                 + SYSTEM_PROMPT.format(budget=budget),
        # Providers with a minimum cacheable size (Haiku: 4096) prove no cache
        # on short prompts; the per-step records report caching instead.
        strict_cache_proof=False,
        # Weak models echo the <session_time> preamble into their answers.
        time_awareness_enabled=False,
        on_event=on_event or (lambda _t, _d: None),
        step_reducer=reducer)
    agentknit.subscribe(session, "usage", on_usage)
    res = agentknit.run_turn(client, model, session, task)

    result.messages = session["messages"]
    result.usage_totals = dict(session["usage_totals"])
    result.final_prompt_tokens = act_usage[-1].get("prompt", 0) if act_usage else 0
    result.final_reply = None if exhausted else res.final_reply
    emit("fb_final", {"text": result.final_reply, "prompt_tokens": result.final_prompt_tokens})
    return result
