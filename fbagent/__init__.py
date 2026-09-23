"""fbagent — a coding agent whose context grows by a fixed token budget per step."""

from __future__ import annotations

from ._loop import Result, StepRecord, run

__all__ = ["Result", "StepRecord", "run"]
