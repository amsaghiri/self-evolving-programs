"""The self-evolving loop: MEU -> PAM -> ASE -> SRCG -> deploy -> run -> learn."""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import List, Optional

from .ase import AdaptiveStrategyExecutor
from .meu import Metrics, MonitoringEvaluationUnit
from .pam import NONE, StatisticalPredictor
from .simulator import MiningNetwork
from .srcg import Generation, SelfReplicatingCodeGenerator


@dataclass
class EpochRecord:
    epoch: int
    threat: str
    generation: int
    p_switch: float
    pool_revenue: float      # ground truth, for reporting
    stale_rate: float
    reward: float
    p_pool_first: float
    attack: bool


class AdaptiveAttacker:
    """A selfish pool that also learns. It can time the release of its block so
    that honest miners tend to see it first, or tend to see it last, and it keeps
    whichever has earned it more. This is what makes a fixed, predictable
    fork-choice rule exploitable."""

    def __init__(self, strengths=(0.9, 0.1), learning_rate: float = 0.4, epsilon: float = 0.15, seed: int = 1):
        self.arms = list(strengths)
        self.value = {a: 0.0 for a in self.arms}
        self.tried = {a: 0 for a in self.arms}
        self.lr, self.epsilon = learning_rate, epsilon
        self.rng = random.Random(seed)
        self.current = self.arms[0]

    def choose(self) -> float:
        untried = [a for a in self.arms if self.tried[a] == 0]
        if untried:
            self.current = untried[0]
        elif self.rng.random() < self.epsilon:
            self.current = self.rng.choice(self.arms)
        else:
            self.current = max(self.arms, key=lambda a: self.value[a])
        return self.current

    def learn(self, revenue: float) -> None:
        a = self.current
        self.tried[a] += 1
        self.value[a] += self.lr * (revenue - self.value[a]) if self.tried[a] > 1 else revenue


class SelfEvolvingDefense:
    def __init__(self, network: MiningNetwork, srcg: SelfReplicatingCodeGenerator,
                 pam=None, ase: Optional[AdaptiveStrategyExecutor] = None,
                 meu: Optional[MonitoringEvaluationUnit] = None, enabled: bool = True):
        self.network = network
        self.srcg = srcg
        self.pam = pam or StatisticalPredictor()
        self.ase = ase or AdaptiveStrategyExecutor()
        self.meu = meu or MonitoringEvaluationUnit()
        self.enabled = enabled                    # False = fixed policy, for baselines
        self.generation: Generation = srcg.genesis()
        self.p_switch = 0.0
        self.history: List[Metrics] = []
        self.records: List[EpochRecord] = []
        self.network.policy = self.generation.policy

    def deploy(self, p_switch: float, context: str = "") -> None:
        """Generate, validate, and install a new generation with the given policy."""
        code = self.srcg.write_policy(p_switch, context)
        self.generation = self.srcg.reproduce(self.generation, code)
        self.p_switch = p_switch
        self.network.policy = self.generation.policy

    def run(self, epochs: int, blocks_per_epoch: int = 3000, attack_from: int = 0,
            attacker: Optional[AdaptiveAttacker] = None) -> List[EpochRecord]:
        state = self.pam.predict(self.history)
        for epoch in range(epochs):
            self.network.attack = epoch >= attack_from
            if attacker is not None and self.network.attack:
                self.network.p_pool_first = attacker.choose()

            # ASE picks a policy for this epoch; SRCG produces the code if it changed.
            action = self.ase.choose(state, self.p_switch) if self.enabled else self.p_switch
            if action != self.p_switch:
                self.deploy(action, context=f"Predicted threat: {state}.")

            result = self.network.run_epoch(blocks_per_epoch)

            # MEU measures, PAM predicts the next state, ASE learns.
            metrics = self.meu.measure(result)
            self.history.append(metrics)
            reward = self.meu.reward(metrics)
            next_state = self.pam.predict(self.history)
            if self.enabled:
                self.ase.learn(state, action, reward, next_state)
            if attacker is not None and self.network.attack:
                attacker.learn(result.pool_revenue)

            self.records.append(EpochRecord(
                epoch=epoch, threat=state, generation=self.generation.number, p_switch=self.p_switch,
                pool_revenue=result.pool_revenue, stale_rate=metrics.stale_rate, reward=reward,
                p_pool_first=self.network.p_pool_first, attack=self.network.attack))
            state = next_state
        return self.records
