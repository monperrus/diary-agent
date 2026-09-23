"""Tests for the fixed-budget-per-step loop, against a scripted fake client."""

from __future__ import annotations

import copy
import itertools
import json
from types import SimpleNamespace as NS
from typing import Any

import fbagent


def _usage(prompt: int, completion: int, cached: int) -> Any:
    return NS(prompt_tokens=prompt, completion_tokens=completion,
              total_tokens=prompt + completion, cached_tokens=cached,
              cache_creation_tokens=0)


def _tool_resp(i: int, cmd: str) -> Any:
    tc = NS(id=f"c{i}", type="function", custom_input=None,
            function=NS(name="exec_shell", arguments=json.dumps({"command": cmd})))
    return NS(choices=[NS(message=NS(content="working", tool_calls=[tc]))],
              usage=_usage(100, 10, 0))


def _text_resp(text: str) -> Any:
    return NS(choices=[NS(message=NS(content=text, tool_calls=None))],
              usage=_usage(100, 5, 90))


class FakeClient:
    """Acts with n shell calls, then answers; summarizes with *digest*."""

    def __init__(self, n_steps: int, digest: str = "digest") -> None:
        self.n_steps = n_steps
        self.digest = digest
        self.calls: list[dict[str, Any]] = []
        self.base_url = NS(host="api.example.test")
        self.chat = NS(completions=NS(create=self.create))

    def create(self, **kw: Any) -> Any:
        self.calls.append(copy.deepcopy({k: v for k, v in kw.items()
                                         if not k.startswith("on_")}))
        if kw.get("tool_choice") == "none":
            return _text_resp(self.digest)
        acts = sum(1 for c in self.calls if c.get("tool_choice") != "none")
        if acts <= self.n_steps:
            return _tool_resp(acts, f"echo {'x' * 5000}")
        return _text_resp("done")


def _acts(client: FakeClient) -> list[dict[str, Any]]:
    return [c for c in client.calls if c.get("tool_choice") != "none"]


def test_only_digests_enter_context() -> None:
    client = FakeClient(3)
    res = fbagent.run(client, "m", "task", budget=50)
    assert res.final_reply == "done"
    assert len(res.steps) == 3
    for m in res.messages:
        assert m["role"] in ("system", "user", "assistant")
        assert "tool_calls" not in m
        assert "xxxxx" not in json.dumps(m)
    digests = [m["content"] for m in res.messages
               if str(m.get("content")).startswith("[step")]
    assert digests == ["[step 1] digest", "[step 2] digest", "[step 3] digest"]


def test_context_grows_linearly() -> None:
    client = FakeClient(4)
    fbagent.run(client, "m", "task", budget=50)
    sizes = [len(json.dumps(c["messages"])) for c in _acts(client)]
    deltas = {b - a for a, b in itertools.pairwise(sizes)}
    assert len(deltas) == 1  # constant growth per step, regardless of 5k-char tool outputs


def test_summary_reuses_prefix_and_caps_tokens() -> None:
    client = FakeClient(2)
    fbagent.run(client, "m", "task", budget=77)
    act, summ, next_act = client.calls[0], client.calls[1], client.calls[2]
    n = len(act["messages"])
    assert summ["messages"][:n] == act["messages"]
    assert next_act["messages"][:n] == act["messages"]
    assert summ["tools"] == act["tools"]
    assert summ["max_tokens"] == 77
    assert summ["messages"][n]["tool_calls"][0]["function"]["name"] == "exec_shell"
    assert summ["messages"][n + 1]["role"] == "tool"


def test_step_records_usage() -> None:
    res = fbagent.run(FakeClient(1), "m", "task", budget=50)
    (rec,) = res.steps
    assert rec.act_prompt_tokens == 100 and rec.act_cached_tokens == 0
    assert rec.summary_cached_tokens == 90 and rec.digest_tokens == 5
    assert rec.raw_step_chars > 5000


def test_done_in_digest_ends_run() -> None:
    client = FakeClient(100, digest="ran it. DONE: 42")
    res = fbagent.run(client, "m", "task", budget=10)
    assert res.final_reply == "42"
    assert len(res.steps) == 1


def test_max_steps_without_answer() -> None:
    res = fbagent.run(FakeClient(100), "m", "task", budget=10, max_steps=2)
    assert res.final_reply is None
    assert len(res.steps) == 2
