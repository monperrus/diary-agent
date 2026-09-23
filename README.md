# fbagent — fixed token budget per step

A coding agent built on [agentknit](../agentknit) in which the context grows
**linearly with the number of steps, with coefficient B** (the per-step budget).

## Principle

At each step the model returns text + tool calls, and the tools run. Instead of
appending that raw material (arguments, file contents, command outputs) to the
conversation, the model is asked to summarize the step in ≤ B tokens, and **only
the digest** is appended:

```
messages = [system, task, d1, go, d2, go, …]          # di ≤ B tokens, go = constant
act:        messages                         → text + tool_calls  → run tools
summarize:  messages + raw_step + "digest ≤B" → di   (max_tokens=B, tool_choice=none)
append:     assistant "[step i] di", user "Step executed …"
```

Context after k steps ≤ prefix + k·(B + c), whatever the size of tool outputs.

**Prefix reuse**: the summarize request starts with the exact `messages` (same tools too)
as the act request just before it, and the next act request starts with that same
prefix again. The provider's prompt cache serves it, so each summary costs only the
raw step plus B output tokens.

Stopping: the model answers without a tool call, or its digest contains `DONE: <answer>`.

## Usage

```
pip install -e .
fbagent "fix the failing test in tests/" --endpoint https://openrouter.ai/api/v1 --model qwen/qwen3-coder --budget 200
fbagent TASK --endpoint run://$HOME/bin/claude-sonnet-5-completions.py --model claude-sonnet-5 --json
```

```python
import agentknit, fbagent
schema = agentknit.load_specification(model, endpoint)
res = fbagent.run(agentknit.create_client(schema), model, task, budget=200)
res.final_reply, res.steps   # per step: prompt/cached tokens of both calls, digest
```

## Implementation

The whole agent is one agentknit `step_reducer` (`fbagent/_loop.py`): agentknit's
`run_turn` drives the loop (tools, hooks, journal, events, sandbox executor), and after
every tool step the reducer calls `side_query(..., max_tokens=B, preamble=None,
count_usage=True)` for the digest and returns a `StepReduction` holding the digest
(with `final_reply=` set when the digest says `DONE:`). Tools are agentknit's defaults
(`read_file`, `write_file`, `str_replace`, `exec_shell`) or those of the loaded spec.

## End-to-end results (B = 120)

Task: create fib.py printing 15 Fibonacci numbers, run it, write test_fib.py, run pytest, report.
The prefix is larger than a bare loop's because it includes agentknit's system prompt.

Claude Sonnet 5, 5 steps, ✅ correct (377, 3 passed):
- step 1 write×2: act prompt 3528 (cached 0), summary prompt 4354 (cached 3526)
- step 2 shell: act prompt 3649 (cached 3526), summary prompt 4430 (cached 3647)
- step 3 shell: act prompt 3768 (cached 3647), summary prompt 4549 (cached 3766)
- step 4 shell: act prompt 3858 (cached 3766), summary prompt 4639 (cached 3856)
- step 5 shell: act prompt 3954 (cached 3856), summary prompt 5363 (cached 3952) → DONE

Every summary call hits the cache for its act call's whole prefix (minus 2 tokens), and
every act call hits the cache written by the previous summary call.

Claude Haiku 4.5, 4 steps, ✅ correct (377, 4 passed). Act prompts 2713 → 2774 → 2882 → 2972.
It shows no cached tokens because Haiku caches nothing below 4096 prompt tokens.

Context grows by 60–140 tokens per step (≤ B + ~20 overhead), while raw steps were
465–2425 chars and are never seen again.

## Lessons from the runs

- 🔁 Amnesia loop: `str_replace` needs the file's exact content, which does not survive a
  120-token digest, so Haiku re-read the same file 16 times. Fix: the system prompt says
  read content vanishes and prefers `write_file`.
- ♻️ Redo loop: Haiku didn't trust `[step N]` assistant messages as proof the step
  had run, so it rewrote the same file again and again. Fix: the system prompt says past
  steps really happened, and the user message says "Step executed and recorded".
- 🛑 Weak models re-verify forever after finishing, so the loop trusts `DONE:` in the digest.
- 📤 agentknit's `exec_shell` echoes to stdout, so `--json` moves it to stderr.
- ⏱️ Haiku copied agentknit's `<session_time>` preamble into its answer, so fbagent
  turns time awareness off.

## Development

```
pip install -e '.[dev]' && pytest && ruff check . && mypy fbagent
```
