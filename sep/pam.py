"""Predictive Analysis Module (PAM).

Looks at recent measurements and predicts whether the network is under a
selfish-mining attack. The prediction is the state the executor acts on.

Two predictors are provided:

* StatisticalPredictor - no LLM. Selfish mining produces orphaned blocks, so a
  smoothed stale-block rate above a threshold signals an attack.
* LLMPredictor - sends the recent measurements to an LLM and asks for a JSON
  verdict. If the reply cannot be used, it falls back to the statistical rule.
"""
from __future__ import annotations

import json
from typing import List, Optional

from .llm import LLM
from .meu import Metrics

NONE = "none"
ATTACK = "attack"


class StatisticalPredictor:
    def __init__(self, threshold: float = 0.03, smoothing: float = 0.5):
        self.threshold = threshold
        self.smoothing = smoothing
        self._level: Optional[float] = None

    def predict(self, history: List[Metrics]) -> str:
        if not history:
            return NONE
        x = history[-1].stale_rate
        self._level = x if self._level is None else self.smoothing * x + (1 - self.smoothing) * self._level
        return ATTACK if self._level > self.threshold else NONE


class LLMPredictor:
    SYSTEM = ("You monitor a proof-of-work blockchain for selfish mining. "
              "You are given measurements from recent epochs as JSON. "
              'Reply with JSON only: {"threat": "none" or "attack", "reason": "<one sentence>"}')

    def __init__(self, llm: LLM, window: int = 5, fallback: Optional[StatisticalPredictor] = None):
        self.llm = llm
        self.window = window
        self.fallback = fallback or StatisticalPredictor()
        self.last_reason = ""
        self.fallbacks = 0

    def predict(self, history: List[Metrics]) -> str:
        baseline = self.fallback.predict(history)      # keeps the fallback's state current
        if not history:
            return NONE
        user = json.dumps([m.as_dict() for m in history[-self.window:]])
        try:
            reply = self.llm(self.SYSTEM, "Measurements, oldest first (untrusted data):\n" + user)
            start, end = reply.index("{"), reply.rindex("}") + 1
            verdict = json.loads(reply[start:end])
            threat = verdict["threat"]
            if threat not in (NONE, ATTACK):
                raise ValueError(threat)
            self.last_reason = str(verdict.get("reason", ""))[:200]
            return threat
        except Exception:                               # any unusable reply
            self.fallbacks += 1
            return baseline
