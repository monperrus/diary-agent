"""Fixed-budget-per-step agent loop.

Each step the model acts (text + tool calls), the tools run, and then the
whole raw step is summarized by the model itself into at most ``budget``
tokens.  Only that digest is appended to the conversation, so after k steps
the context is ``prefix + k * (budget + c)`` tokens at most.

The summarize request is ``messages + raw_step + [instruction]`` where
``messages`` is byte-identical to the prefix of the act request that
preceded it: the provider's prompt cache serves that prefix.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import agentknit

SYSTEM_PROMPT = """\
You are a coding agent working in {cwd}. Use the tools to complete the task.
Your memory works in steps: after every step, the raw tool calls and tool
outputs are discarded and replaced by a digest of at most {budget} tokens that
you write yourself. Past steps appear as "[step N] ..." assistant messages:
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

CONTINUE_PROMPT = "Step executed and recorded. Do the next remaining action."

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


def _cached(usage: Any) -> int:
    """Cached prompt tokens across agentknit's and openai's usage shapes."""
    if usage is None:
        return 0
    val = getattr(usage, "cached_tokens", None)
    if val is None:
        details = getattr(usage, "prompt_tokens_details", None)
        val = getattr(details, "cached_tokens", None) if details else None
    return int(val or 0)


def _tool_call_item(tc: Any) -> dict[str, Any]:
    return {"id": tc.id, "type": "function",
            "function": {"name": tc.function.name, "arguments": tc.function.arguments}}


def _execute(tc: Any, tool_dispatch: dict[str, Any], max_chars: int) -> str:
    try:
        args = json.loads(tc.function.arguments or "{}")
        if not isinstance(args, dict):
            raise TypeError("arguments must be a JSON object")
    except (json.JSONDecodeError, TypeError) as exc:
        return f"ERROR: malformed arguments for {tc.function.name}: {exc}"
    text, _meta = agentknit.dispatch(tc.function.name, args, tool_dispatch)
    if len(text) > max_chars:
        text = text[:max_chars] + f"\n[... truncated {len(text) - max_chars} chars]"
    return text


def run(client: Any, model: str, task: str, *, budget: int = 200,
        max_steps: int = 20, max_tool_chars: int = 20_000,
        cwd: str | None = None, on_event: EventHandler | None = None) -> Result:
    """Run *task* to completion with a context growing by ≤ *budget* tokens per step."""
    emit = on_event or (lambda _t, _d: None)
    tools, tool_dispatch = agentknit.default_tool_spec()
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT.format(cwd=cwd or os.getcwd(), budget=budget)},
        {"role": "user", "content": task},
    ]
    result = Result(final_reply=None, messages=messages)

    for step in range(1, max_steps + 1):
        # 1. act: the prefix is messages, as-is.
        resp = client.chat.completions.create(model=model, messages=list(messages),
                                              tools=tools, temperature=0)
        msg = resp.choices[0].message
        result.final_prompt_tokens = getattr(resp.usage, "prompt_tokens", 0) or 0
        if not msg.tool_calls:
            result.final_reply = msg.content or ""
            emit("final", {"step": step, "text": result.final_reply,
                           "prompt_tokens": result.final_prompt_tokens})
            messages.append({"role": "assistant", "content": result.final_reply})
            return result

        # 2. execute tools; the raw step lives only in this local list.
        assistant: dict[str, Any] = {"role": "assistant", "content": msg.content or None,
                                     "tool_calls": [_tool_call_item(tc) for tc in msg.tool_calls]}
        raw_step: list[dict[str, Any]] = [assistant]
        for tc in msg.tool_calls:
            emit("tool_call", {"step": step, "name": tc.function.name,
                               "args": tc.function.arguments})
            out = _execute(tc, tool_dispatch, max_tool_chars)
            raw_step.append({"role": "tool", "tool_call_id": tc.id, "content": out})

        # 3. summarize, reusing the same prefix (and the same tools, which
        #    providers such as Anthropic put before the messages in the cache).
        instruction = {"role": "user", "content": SUMMARIZE_PROMPT.format(
            step=step, budget=budget, words=int(budget * 0.4))}
        sresp = client.chat.completions.create(
            model=model, messages=list(messages) + raw_step + [instruction],
            tools=tools, tool_choice="none", max_tokens=budget, temperature=0)
        digest = (sresp.choices[0].message.content or "").strip()
        if not digest:  # model ignored tool_choice=none: fall back to a mechanical digest
            digest = "called " + ", ".join(tc.function.name for tc in msg.tool_calls)

        # 4. append only the digest.
        messages.append({"role": "assistant", "content": f"[step {step}] {digest}"})
        messages.append({"role": "user", "content": CONTINUE_PROMPT})

        rec = StepRecord(
            step=step,
            tool_calls=[tc.function.name for tc in msg.tool_calls],
            act_prompt_tokens=result.final_prompt_tokens,
            act_cached_tokens=_cached(resp.usage),
            summary_prompt_tokens=getattr(sresp.usage, "prompt_tokens", 0) or 0,
            summary_cached_tokens=_cached(sresp.usage),
            digest_tokens=getattr(sresp.usage, "completion_tokens", 0) or 0,
            raw_step_chars=sum(len(json.dumps(m)) for m in raw_step),
            digest=digest,
        )
        result.steps.append(rec)
        emit("step", rec.__dict__)

        # The digest is the model's own judgement of the step: trust its DONE,
        # weaker models otherwise keep re-verifying forever.
        if DONE_MARKER in digest:
            result.final_reply = digest.split(DONE_MARKER, 1)[1].strip()
            emit("final", {"step": step, "text": result.final_reply,
                           "prompt_tokens": result.final_prompt_tokens})
            return result

    return result
