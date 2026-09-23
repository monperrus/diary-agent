# The Fixed-Budget Agent

A coding-agent concept car: **the context grows by at most B tokens per step**,
whatever the tools return.

In a normal agent loop, every tool call and every tool output (a whole file, a
test log, a directory listing) is appended to the conversation forever. Here,
at the end of each step, the model itself summarizes what just happened into a
digest of at most B tokens, and **only the digest** is appended. The raw step
is thrown away. After k steps the context is `prefix + k·(B + c)` tokens: linear
in the number of steps, with coefficient B.

The summarization reuses the prompt cache. The summarize request starts with
exactly the same prefix (system prompt, tools, previous digests) as the request
that produced the step, so the provider serves that prefix from cache. The next
step's request again starts with the same prefix.

```
messages = [system, task, m1, m2, …]            # mi = digest of step i, ≤ B tokens
act:        messages                              → text + tool calls → run tools
summarize:  messages + raw step + "digest ≤ B"    → di   (max_tokens = B, no tools)
append:     user "[memory of your step i, already executed] di …"
```

## Implementation

Built on [agentknit](https://github.com/monperrus/agentknit), for any
OpenAI-compatible `/chat/completions` endpoint. The whole agent is one
agentknit `step_reducer` (see `fbagent/_loop.py`):

- agentknit's `run_turn` drives the loop: tools, hooks, journal, events, sandbox.
- After each tool step, the reducer asks for the digest with
  `side_query(..., max_tokens=B, preamble=None, count_usage=True)`. That call
  sends the session's exact prefix plus the raw step, so the cache hits.
- It returns a `StepReduction` holding one memory message, which replaces the
  raw step in the history. When the digest says `DONE: <answer>`, it sets
  `final_reply`, which ends the run.

`step_reducer` and the `side_query` options were added to agentknit for this
concept car.

## Usage

```
pip install git+https://github.com/monperrus/fixed-budget-agent
fbagent "fix the failing test in tests/" --endpoint https://openrouter.ai/api/v1 \
    --model qwen/qwen3-coder --budget 200
fbagent TASK --endpoint run:///path/to/completions-shim --model M --json
```

```python
import agentknit, fbagent

schema = agentknit.load_specification(model, endpoint)
res = fbagent.run(agentknit.create_client(schema), model, task, budget=200, schema=schema)
res.final_reply, res.steps, res.usage_totals
```

Each entry in `res.steps` records the step's tool calls, the prompt and cached
tokens of both calls (act and summarize), the raw step size and the digest.

`agents/agent-fbagent.py` is an entry point for benchmark harnesses that call
`agent --non-interactive "<task>"`. It reads `$FBAGENT_ENDPOINT` and
`$FBAGENT_MODEL`.

## Results

### Worked example (B = 120)

Task: create fib.py printing 15 Fibonacci numbers, run it, write test_fib.py,
run pytest, report.

Claude Sonnet 5, 5 steps, ✅ correct:
- step 1 (write×2): act prompt 3528, cached 0; summary prompt 4354, cached 3526
- step 2 (shell): act prompt 3649, cached 3526; summary prompt 4430, cached 3647
- step 3 (shell): act prompt 3768, cached 3647; summary prompt 4549, cached 3766
- step 4 (shell): act prompt 3858, cached 3766; summary prompt 4639, cached 3856
- step 5 (shell): act prompt 3954, cached 3856; summary prompt 5363, cached 3952 → DONE

Every summary call hits the cache for its act call's whole prefix (minus 2
tokens), and every act call hits the cache written by the previous summary
call. The context grows by 90–121 tokens per step, while raw steps were
1.2–2.4k characters.

### 15-task coding benchmark (Claude Haiku 4.5, B = 200)

The benchmark has 15 small agentic coding tasks, each checked by an oracle.
The baseline is a plain agentknit loop on the same model, which keeps every
tool call and output in context. It ran a month earlier (2026-08-16) with a
shorter system prompt (about 1.3k tokens, against 2.6k for fbagent).

Pass rate:
- baseline, full context: 15/15
- fbagent v1, digests stored as assistant messages: 8/15
- fbagent v2, digests stored as user-role memory messages: 11/15

The 4 v2 failures:
- buggy-script-fix: hit the 180 s task timeout.
- data-pipeline-recovery and dead-code-removal: hit the 40-step cap, going
  round in circles. Both need exact file content across steps, which a
  200-token digest cannot carry.
- git-log-analysis: finished, but the report was wrong.

Context size, on the longest tasks:
- binary-format-re: the baseline's last prompt was 31.8k tokens after 28 calls,
  about +1.1k per call. fbagent's was 8.7k after 25 steps, about +245 per step.
- test-authoring: 22.4k after 24 calls for the baseline, 4.5k after 8 steps
  for fbagent.

Cost over the 15 tasks:
- baseline: 1.36M prompt tokens (82% cached), 1000 s wall time.
- fbagent v2: 1.88M prompt tokens (71% cached), 1182 s wall time.

fbagent makes two calls per step, and on hard tasks it takes more steps
because details are lost between steps.

Takeaway: the context stays bounded as designed, at 3–5× less than the
baseline. That does not make it cheaper overall: the digest calls and the
extra steps cost more than the smaller context saves. The price is also
accuracy, on tasks that need verbatim state across steps.

## Lessons

- 🪞 **Models imitate their own past messages.** With digests stored as
  assistant messages, Haiku often "acted" by writing another digest and no
  tool call, which ends the run. Storing digests as user-role memory messages
  fixed it.
- 🔁 **Amnesia loops.** `str_replace` needs a file's exact text, which does not
  survive a 120-token digest, so Haiku re-read the same file 16 times. The
  system prompt now says that read content vanishes and to prefer `write_file`.
- ♻️ **Redo loops.** Haiku did not trust summaries of its past steps and redid
  them. The memory message now says the step was already executed and its
  effects are on disk.
- 🛑 **Endless re-verification.** After finishing, weak models keep checking.
  The loop trusts `DONE:` in the digest.
- 💸 **The cache has a floor.** Haiku caches nothing below 4096 prompt tokens, so
  short runs pay the summary prefix in full; Sonnet 5 hits the cache on every call.

## Development

```
pip install -e '.[dev]' && pytest && ruff check . && mypy fbagent
```

MIT license.
