"""Mining-race simulator for selfish mining (the SM1 strategy of Eyal and Sirer).

The network has `n_nodes` miners with equal hash power. A fraction of them
collude as one selfish pool; the rest are honest. Blocks are found one at a
time. The pool follows SM1: it withholds blocks and releases them to win races.

The only place honest behaviour matters is a *tie*: two competing branches of
the same length, one honest and one from the pool. Each honest miner then has
to decide which branch to extend. That decision is the `policy`, a function

    policy(first_seen, later_seen, rng) -> first_seen or later_seen

and it is the piece of code the framework evolves.

Which branch an honest miner sees first depends on how well the pool is
connected. `p_pool_first` is the probability that an honest miner hears the
pool's block before the honest one.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Callable, Dict

POOL = "pool"
HONEST_BRANCH = "honest"
POOL_BRANCH = "pool"

Policy = Callable[[str, str, random.Random], str]


def first_seen_policy(first_seen: str, later_seen: str, rng: random.Random) -> str:
    """Bitcoin's default rule: keep mining on the block you heard about first."""
    return first_seen


@dataclass
class EpochResult:
    """What happened in one epoch. All fields are observable by an honest node."""

    blocks_mined: int = 0                       # every block found, main chain or not
    main_chain: int = 0                         # blocks that ended up on the main chain
    stale: int = 0                              # blocks that were orphaned
    ties: int = 0                               # races between two equal branches
    mined_by: Dict[str, int] = field(default_factory=dict)   # miner -> blocks found
    main_by: Dict[str, int] = field(default_factory=dict)    # miner -> blocks on main chain

    def _credit(self, miner: str, on_main: bool) -> None:
        self.mined_by[miner] = self.mined_by.get(miner, 0) + 1
        self.blocks_mined += 1
        if on_main:
            self.main_by[miner] = self.main_by.get(miner, 0) + 1
            self.main_chain += 1
        else:
            self.stale += 1

    @property
    def pool_revenue(self) -> float:
        """Pool's share of main-chain blocks. Ground truth, used for reporting only."""
        return self.main_by.get(POOL, 0) / self.main_chain if self.main_chain else 0.0


class MiningNetwork:
    def __init__(self, n_nodes: int = 100, selfish_fraction: float = 0.2,
                 p_pool_first: float = 0.9, attack: bool = True, seed: int = 0):
        self.n_nodes = n_nodes
        self.n_selfish = round(n_nodes * selfish_fraction)
        self.n_honest = n_nodes - self.n_selfish
        self.alpha = self.n_selfish / n_nodes     # pool's share of hash power
        self.p_pool_first = p_pool_first
        self.attack = attack
        self.rng = random.Random(seed)
        self.policy: Policy = first_seen_policy
        # SM1 state carried across epochs
        self._lead = 0            # private blocks the pool is ahead by
        self._tie = False         # a published race is in progress
        self._tie_honest_miner = ""

    # -- helpers ---------------------------------------------------------------
    def _honest_miner(self) -> str:
        return f"h{self.rng.randrange(self.n_honest):02d}"

    def _honest_choice(self) -> str:
        """Branch that one honest miner extends during a tie, according to the policy."""
        if self.rng.random() < self.p_pool_first:
            first, later = POOL_BRANCH, HONEST_BRANCH
        else:
            first, later = HONEST_BRANCH, POOL_BRANCH
        choice = self.policy(first, later, self.rng)
        if choice not in (first, later):
            raise ValueError("policy must return one of the two branches")
        return choice

    # -- one epoch -------------------------------------------------------------
    def run_epoch(self, n_blocks: int = 3000) -> EpochResult:
        r = EpochResult()
        while r.blocks_mined < n_blocks:
            pool_finds = self.rng.random() < self.alpha

            if not self.attack:                      # the pool mines honestly
                r._credit(POOL if pool_finds else self._honest_miner(), on_main=True)
                continue

            if self._tie:
                r.ties += 1
                if pool_finds:                       # pool extends its branch and wins
                    r._credit(POOL, True); r._credit(POOL, True)
                    r._credit(self._tie_honest_miner, False)
                else:
                    miner = self._honest_miner()
                    if self._honest_choice() == POOL_BRANCH:
                        r._credit(POOL, True); r._credit(miner, True)
                        r._credit(self._tie_honest_miner, False)
                    else:
                        r._credit(self._tie_honest_miner, True); r._credit(miner, True)
                        r._credit(POOL, False)
                self._tie = False
                self._lead = 0
            elif pool_finds:
                self._lead += 1                      # withhold the new block
            else:
                miner = self._honest_miner()
                if self._lead == 0:
                    r._credit(miner, True)
                elif self._lead == 1:                # pool publishes: a tie begins
                    self._tie = True
                    self._tie_honest_miner = miner
                    self._lead = 0
                elif self._lead == 2:                # pool publishes both and wins
                    r._credit(POOL, True); r._credit(POOL, True)
                    r._credit(miner, False)
                    self._lead = 0
                else:                                # pool releases one block, stays ahead
                    r._credit(POOL, True)
                    r._credit(miner, False)
                    self._lead -= 1
        return r


def sm1_revenue(alpha: float, gamma: float) -> float:
    """Closed-form relative revenue of an SM1 pool (Eyal and Sirer, 2014).

    alpha is the pool's hash-power share; gamma is the fraction of honest hash
    power that mines on the pool's block during a tie.
    """
    num = alpha * (1 - alpha) ** 2 * (4 * alpha + gamma * (1 - 2 * alpha)) - alpha ** 3
    den = 1 - alpha * (1 + (2 - alpha) * alpha)
    return num / den


def effective_gamma(p_switch: float, p_pool_first: float) -> float:
    """Gamma produced by a policy that switches to the later-seen branch with
    probability p_switch, when the pool's block is seen first with probability
    p_pool_first."""
    return (1 - p_switch) * p_pool_first + p_switch * (1 - p_pool_first)
