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

Tools are agentknit's defaults (`read_file`, `write_file`, `str_replace`, `exec_shell`),
dispatched with `agentknit.dispatch`.

## End-to-end results (B = 120)

Task: create fib.py printing 15 Fibonacci numbers, run it, write test_fib.py, run pytest, report.

Claude Sonnet 5, 3 steps, ✅ correct (377, 2 passed):
- step 1 write×2: act prompt 1252 (cached 0), summary prompt 1885 (cached 1250)
- step 2 read×2: act prompt 1389 (cached 1250), summary prompt 1972 (cached 1387)
- step 3 shell: act prompt 1508 (cached 1387), summary prompt 2422 (cached 1506) → DONE

Claude Haiku 4.5, 4 steps, ✅ correct (377, 7 passed). Act prompts 1183 → 1325 → 1452 → 1594.
It shows no cached tokens because Haiku caches nothing below 4096 prompt tokens.

Context grows by 119–142 tokens per step (= B + ~15 overhead), while raw steps were
550–1950 chars and are never seen again.

## Lessons from the runs

- 🔁 Amnesia loop: `str_replace` needs the file's exact content, which does not survive a
  120-token digest, so Haiku re-read the same file 16 times. Fix: the system prompt says
  read content vanishes and prefers `write_file`.
- ♻️ Redo loop: Haiku didn't trust `[step N]` assistant messages as proof the step
  had run, so it rewrote the same file again and again. Fix: the system prompt says past
  steps really happened, and the user message says "Step executed and recorded".
- 🛑 Weak models re-verify forever after finishing, so the loop trusts `DONE:` in the digest.
- 📤 agentknit's `exec_shell` echoes to stdout, so `--json` moves it to stderr.

## Development

```
pip install -e '.[dev]' && pytest && ruff check . && mypy fbagent
```
