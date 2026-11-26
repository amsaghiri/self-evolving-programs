"""Monitoring and Evaluation Unit (MEU).

Turns the raw outcome of an epoch into the measurements the other components
use. Everything here can be computed by an honest node from public data: which
blocks were found, by whom, and which of them ended up on the main chain.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict

from .simulator import EpochResult


@dataclass
class Metrics:
    stale_rate: float            # orphaned blocks / all blocks found
    tie_rate: float              # races / all blocks found
    largest_producer: str        # miner (or pool) that found the most blocks
    largest_hash_share: float    # its share of all blocks found: an estimate of its hash power
    largest_main_share: float    # its share of main-chain blocks: its revenue
    largest_excess: float        # revenue minus hash share; positive means it earns more than is fair

    def as_dict(self) -> Dict[str, float]:
        return asdict(self)


class MonitoringEvaluationUnit:
    def measure(self, r: EpochResult) -> Metrics:
        largest = max(r.mined_by, key=lambda m: (r.mined_by[m], m))
        hash_share = r.mined_by[largest] / r.blocks_mined
        main_share = r.main_by.get(largest, 0) / r.main_chain
        return Metrics(
            stale_rate=r.stale / r.blocks_mined,
            tie_rate=r.ties / r.blocks_mined,
            largest_producer=largest,
            largest_hash_share=hash_share,
            largest_main_share=main_share,
            largest_excess=main_share - hash_share,
        )

    @staticmethod
    def reward(m: Metrics) -> float:
        """Learning signal for the executor: the less the largest producer earns
        above its fair share, the better."""
        return -m.largest_excess
