"""Adaptive Strategy Executor (ASE).

Decides which fork-choice policy the honest nodes should run, using tabular
Q-learning. The state is the threat level predicted by PAM. The actions are
policy settings: the probability of switching to the later-seen branch in a tie.
The reward comes from the MEU.
"""
from __future__ import annotations

import random
from typing import Dict, List, Sequence, Tuple


class AdaptiveStrategyExecutor:
    def __init__(self, actions: Sequence[float] = (0.0, 0.5, 1.0), learning_rate: float = 0.3,
                 discount: float = 0.5, epsilon: float = 0.1, seed: int = 0):
        self.actions: List[float] = list(actions)
        self.lr = learning_rate
        self.discount = discount
        self.epsilon = epsilon
        self.rng = random.Random(seed)
        self.q: Dict[Tuple[str, float], float] = {}

    def _q(self, state: str, action: float) -> float:
        return self.q.get((state, action), 0.0)

    def choose(self, state: str, current: float) -> float:
        """Epsilon-greedy choice. When several actions look equally good, keep
        the current one, so that code is not regenerated without a reason."""
        if self.rng.random() < self.epsilon:
            return self.rng.choice(self.actions)
        best = max(self._q(state, a) for a in self.actions)
        candidates = [a for a in self.actions if self._q(state, a) >= best - 1e-12]
        return current if current in candidates else candidates[0]

    def learn(self, state: str, action: float, reward: float, next_state: str) -> None:
        future = max(self._q(next_state, a) for a in self.actions)
        old = self._q(state, action)
        self.q[(state, action)] = old + self.lr * (reward + self.discount * future - old)
