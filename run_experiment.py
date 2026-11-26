#!/usr/bin/env python3
"""Run the selfish-mining scenarios and print a summary.

    python run_experiment.py                 # all scenarios, default settings
    python run_experiment.py --epochs 300 --seed 7
    python run_experiment.py --plot results.png      # needs matplotlib
"""
from __future__ import annotations

import argparse
import statistics
import tempfile
from collections import Counter

from sep.ase import AdaptiveStrategyExecutor
from sep.framework import AdaptiveAttacker, SelfEvolvingDefense
from sep.simulator import MiningNetwork, effective_gamma, sm1_revenue
from sep.srcg import SelfReplicatingCodeGenerator


def run(name, *, evolving, fixed_p=0.0, adaptive_attacker, args):
    slug = "".join(c if c.isalnum() else "-" for c in name).strip("-")
    while "--" in slug:
        slug = slug.replace("--", "-")
    workdir = tempfile.mkdtemp(prefix="sep-generations-") if args.keep is None else f"{args.keep}/{slug}"
    net = MiningNetwork(args.nodes, args.selfish, p_pool_first=args.p_pool_first, seed=args.seed)
    defense = SelfEvolvingDefense(
        net, SelfReplicatingCodeGenerator(workdir),
        ase=AdaptiveStrategyExecutor(seed=args.seed), enabled=evolving)
    if not evolving and fixed_p != 0.0:
        defense.deploy(fixed_p)
    attacker = AdaptiveAttacker(seed=args.seed + 1) if adaptive_attacker else None
    records = defense.run(args.epochs, args.blocks, attack_from=args.attack_from, attacker=attacker)
    return name, records, defense


def summarize(name, records, defense, args):
    attacked = [r for r in records if r.attack]
    late = attacked[len(attacked) // 2:]                 # second half: after learning has settled
    revenue = statistics.mean(r.pool_revenue for r in late)
    policies = Counter(r.p_switch for r in late)
    share = ", ".join(f"p={p:g}: {100 * n / len(late):.0f}%" for p, n in sorted(policies.items()))
    return (name, revenue, revenue - defense.network.alpha, defense.generation.number, share)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--nodes", type=int, default=100)
    ap.add_argument("--selfish", type=float, default=0.2, help="fraction of nodes in the selfish pool")
    ap.add_argument("--p-pool-first", type=float, default=0.9,
                    help="chance an honest miner hears the pool's block first (static attacker)")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--blocks", type=int, default=3000, help="blocks per epoch")
    ap.add_argument("--attack-from", type=int, default=20, help="epoch at which the attack starts")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--keep", default=None, help="directory in which to keep the generation files")
    ap.add_argument("--plot", default=None, help="write a figure to this file (needs matplotlib)")
    args = ap.parse_args()

    alpha = round(args.nodes * args.selfish) / args.nodes
    print(f"Network: {args.nodes} nodes, {round(args.nodes * args.selfish)} in the selfish pool "
          f"(fair share {alpha:.3f}). Attack starts at epoch {args.attack_from}.")
    print("Theory (Eyal-Sirer) for this pool, by fraction of honest power that sides with it in a tie:")
    for g in (0.9, 0.5, 0.1):
        print(f"  gamma={g}: revenue {sm1_revenue(alpha, g):.4f}")
    print()

    scenarios = [
        run("static attacker, no defense (first-seen rule)", evolving=False, adaptive_attacker=False, args=args),
        run("static attacker, self-evolving defense", evolving=True, adaptive_attacker=False, args=args),
        run("adaptive attacker, fixed 'always switch' rule", evolving=False, fixed_p=1.0, adaptive_attacker=True, args=args),
        run("adaptive attacker, fixed uniform tie-breaking", evolving=False, fixed_p=0.5, adaptive_attacker=True, args=args),
        run("adaptive attacker, self-evolving defense", evolving=True, adaptive_attacker=True, args=args),
    ]

    print(f"{'scenario':<48}{'pool revenue':>13}{'vs fair':>9}{'generations':>13}  policy in use (second half)")
    for s in scenarios:
        name, rev, excess, gens, share = summarize(*s, args)
        print(f"{name:<48}{rev:>13.4f}{excess:>+9.4f}{gens:>13}  {share}")

    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        def smooth(values, window=10):
            out = []
            for i in range(len(values)):
                chunk = values[max(0, i - window + 1):i + 1]
                out.append(sum(chunk) / len(chunk))
            return out

        fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True, gridspec_kw={"height_ratios": [3, 2]})
        for name, records, _ in scenarios:
            axes[0].plot([r.epoch for r in records], smooth([r.pool_revenue for r in records]),
                         label=name, linewidth=1.6)
        axes[0].axhline(alpha, color="black", linestyle="--", linewidth=1, label="fair share")
        axes[0].axvline(args.attack_from, color="gray", linestyle=":", linewidth=1)
        axes[0].set_ylabel("pool revenue share (10-epoch mean)")
        axes[0].legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, 1.32), ncol=2, frameon=False)
        for name, records, _ in scenarios:
            if "self-evolving" in name:
                axes[1].step([r.epoch for r in records], [r.p_switch for r in records], label=name, where="post")
        axes[1].axvline(args.attack_from, color="gray", linestyle=":", linewidth=1)
        axes[1].set_yticks([0, 0.5, 1])
        axes[1].set_ylabel("deployed policy (p_switch)")
        axes[1].set_xlabel("epoch (dotted line: attack starts)")
        axes[1].legend(fontsize=8, loc="lower right")
        fig.tight_layout()
        fig.savefig(args.plot, dpi=150)
        print(f"\nfigure written to {args.plot}")


if __name__ == "__main__":
    main()
