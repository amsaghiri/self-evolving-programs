"""Tests. Standard library only:  python -m unittest discover -s tests -v"""
import os
import subprocess
import sys
import tempfile
import unittest
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from sep.ase import AdaptiveStrategyExecutor
from sep.framework import AdaptiveAttacker, SelfEvolvingDefense
from sep.meu import MonitoringEvaluationUnit
from sep.pam import ATTACK, NONE, LLMPredictor, StatisticalPredictor
from sep.simulator import MiningNetwork, effective_gamma, sm1_revenue
from sep.srcg import (GENESIS_SOURCE, PolicyRejected, SelfReplicatingCodeGenerator,
                      load_generation, policy_template, validate_policy)

GOOD = "def choose_branch(first_seen, later_seen, rng):\n    return later_seen if rng.random() < 0.5 else first_seen"


class Simulator(unittest.TestCase):
    def test_matches_eyal_sirer_closed_form(self):
        for alpha in (0.2, 0.3, 0.4):
            for gamma in (0.1, 0.5, 0.9):
                net = MiningNetwork(100, alpha, p_pool_first=gamma, seed=1)
                sim = net.run_epoch(300_000).pool_revenue
                self.assertAlmostEqual(sim, sm1_revenue(alpha, gamma), delta=0.004,
                                       msg=f"alpha={alpha} gamma={gamma}")

    def test_policy_changes_gamma_as_predicted(self):
        for p in (0.0, 0.5, 1.0):
            ns = {"__builtins__": {}}
            exec(policy_template(p), ns)
            net = MiningNetwork(100, 0.2, p_pool_first=0.9, seed=2)
            net.policy = ns["choose_branch"]
            expected = sm1_revenue(0.2, effective_gamma(p, 0.9))
            self.assertAlmostEqual(net.run_epoch(300_000).pool_revenue, expected, delta=0.004)

    def test_without_attack_nothing_is_orphaned(self):
        r = MiningNetwork(100, 0.2, attack=False, seed=3).run_epoch(50_000)
        self.assertEqual(r.stale, 0)
        self.assertAlmostEqual(r.pool_revenue, 0.2, delta=0.01)

    def test_block_accounting_is_consistent(self):
        r = MiningNetwork(100, 0.3, seed=4).run_epoch(20_000)
        self.assertEqual(r.blocks_mined, r.main_chain + r.stale)
        self.assertEqual(sum(r.mined_by.values()), r.blocks_mined)
        self.assertEqual(sum(r.main_by.values()), r.main_chain)


class Quine(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.srcg = SelfReplicatingCodeGenerator(self.dir)

    def emit(self, path):
        return subprocess.run([sys.executable, path, "--emit"], capture_output=True, text=True, check=True).stdout

    def test_genesis_prints_its_own_source(self):
        g0 = self.srcg.genesis()
        with open(g0.path, encoding="utf-8") as f:
            on_disk = f.read()
        self.assertEqual(self.emit(g0.path), on_disk)
        self.assertEqual(on_disk, GENESIS_SOURCE)

    def test_generation_does_not_read_its_own_file(self):
        self.assertNotIn("open(", GENESIS_SOURCE)
        self.assertNotIn("__file__", GENESIS_SOURCE)

    def test_offspring_is_a_quine_with_the_new_policy(self):
        g0 = self.srcg.genesis()
        g1 = self.srcg.reproduce(g0, policy_template(0.5))
        g2 = self.srcg.reproduce(g1, policy_template(1.0))
        for g in (g1, g2):
            with open(g.path, encoding="utf-8") as f:
                self.assertEqual(self.emit(g.path), f.read())
        self.assertEqual((g1.number, g2.number), (1, 2))
        self.assertAlmostEqual(g1.measured_p_switch, 0.5, delta=0.03)
        self.assertEqual(g2.measured_p_switch, 1.0)
        self.assertIn(g0.sha256, g1.source)            # each generation records its parent
        self.assertIn(g1.sha256, g2.source)

    def test_only_the_policy_differs_between_generations(self):
        g0 = self.srcg.genesis()
        g1 = self.srcg.reproduce(g0, policy_template(0.5))
        def core(src):
            ns = {"__name__": "generation"}
            exec(src, ns)
            t = ns["_T"]
            t = t.replace(ns["policy_source"](), "")
            t = t.replace("GENERATION = %d\n" % ns["GENERATION"], "")
            return t.replace('PARENT_SHA256 = "%s"\n' % ns["PARENT_SHA256"], "")
        self.assertEqual(core(g0.source), core(g1.source))

    def test_tampered_generation_is_refused(self):
        with self.assertRaises(ValueError):
            load_generation(GENESIS_SOURCE.replace("P_SWITCH = 0.0", "P_SWITCH = 0.9", 1))


class PolicyValidation(unittest.TestCase):
    def reject(self, code):
        with self.assertRaises(PolicyRejected):
            validate_policy(code)

    def test_accepts_a_valid_policy(self):
        self.assertAlmostEqual(validate_policy(GOOD), 0.5, delta=0.03)

    def test_rejects_unsafe_or_wrong_code(self):
        head = "def choose_branch(first_seen, later_seen, rng):\n"
        self.reject("import os\n" + head + "    return first_seen")
        self.reject(head + "    while True:\n        pass")
        self.reject(head + "    for i in (1, 2):\n        pass\n    return first_seen")
        self.reject(head + "    return open('/etc/hostname')")
        self.reject(head + "    return rng.__class__")
        self.reject(head + "    return rng.seed(1)")
        self.reject(head + "    return 'neither'")
        self.reject(head + "    return first_seen +")
        self.reject("def other(a, b, c):\n    return a")
        self.reject("def choose_branch(a, b, c):\n    return a")
        self.reject(head + "    return first_seen\n# --- POLICY END ---")


class Predictors(unittest.TestCase):
    def metrics(self, attack):
        r = MiningNetwork(100, 0.2, attack=attack, seed=5).run_epoch(3000)
        return MonitoringEvaluationUnit().measure(r)

    def test_statistical_predictor(self):
        self.assertEqual(StatisticalPredictor().predict([]), NONE)
        self.assertEqual(StatisticalPredictor().predict([self.metrics(False)]), NONE)
        self.assertEqual(StatisticalPredictor().predict([self.metrics(True)]), ATTACK)

    def test_llm_predictor_uses_a_valid_reply(self):
        p = LLMPredictor(lambda system, user: 'Sure. {"threat": "attack", "reason": "many orphans"}')
        self.assertEqual(p.predict([self.metrics(False)]), ATTACK)
        self.assertEqual(p.fallbacks, 0)

    def test_llm_predictor_falls_back_on_a_bad_reply(self):
        for reply in ("I cannot say.", '{"threat": "maybe"}', "{not json}"):
            p = LLMPredictor(lambda system, user, reply=reply: reply)
            self.assertEqual(p.predict([self.metrics(True)]), ATTACK)      # the statistical answer
            self.assertEqual(p.fallbacks, 1)

    def test_meu_sees_the_pool_earning_more_than_its_share(self):
        m = self.metrics(True)
        self.assertEqual(m.largest_producer, "pool")
        self.assertGreater(m.largest_excess, 0)
        self.assertGreater(m.stale_rate, 0.1)


class CodeGeneratorWithLLM(unittest.TestCase):
    def srcg(self, reply):
        return SelfReplicatingCodeGenerator(tempfile.mkdtemp(), llm=lambda system, user: reply)

    def test_valid_llm_code_is_used(self):
        s = self.srcg("```python\n" + GOOD + "\n```")
        self.assertEqual(s.write_policy(0.5), GOOD)
        self.assertEqual(s.rejected, 0)

    def test_unsafe_llm_code_is_replaced_by_the_template(self):
        s = self.srcg("import os\ndef choose_branch(first_seen, later_seen, rng):\n    return first_seen")
        self.assertEqual(s.write_policy(0.5), policy_template(0.5))
        self.assertEqual(s.rejected, 1)

    def test_llm_code_with_the_wrong_behaviour_is_replaced(self):
        s = self.srcg(GOOD)                       # switches half the time, but 1.0 was requested
        self.assertEqual(s.write_policy(1.0), policy_template(1.0))
        self.assertEqual(s.rejected, 1)


class Executor(unittest.TestCase):
    def test_keeps_the_current_policy_when_nothing_is_better(self):
        ase = AdaptiveStrategyExecutor(epsilon=0.0)
        self.assertEqual(ase.choose(NONE, 0.5), 0.5)

    def test_learns_the_better_action(self):
        ase = AdaptiveStrategyExecutor(epsilon=0.0)
        for _ in range(20):
            ase.learn(ATTACK, 0.0, -0.02, ATTACK)
            ase.learn(ATTACK, 1.0, +0.05, ATTACK)
        self.assertEqual(ase.choose(ATTACK, 0.0), 1.0)


class EndToEnd(unittest.TestCase):
    def run_defense(self, adaptive, seed=42, epochs=160):
        net = MiningNetwork(100, 0.2, p_pool_first=0.9, seed=seed)
        d = SelfEvolvingDefense(net, SelfReplicatingCodeGenerator(tempfile.mkdtemp()),
                                ase=AdaptiveStrategyExecutor(seed=seed))
        attacker = AdaptiveAttacker(seed=seed + 1) if adaptive else None
        records = d.run(epochs, 3000, attack_from=20, attacker=attacker)
        late = records[(20 + epochs) // 2:]
        revenue = sum(r.pool_revenue for r in late) / len(late)
        return d, records, Counter(r.p_switch for r in late).most_common(1)[0][0], revenue

    def test_no_code_is_generated_before_the_attack(self):
        d, records, _, _ = self.run_defense(adaptive=False)
        self.assertTrue(all(r.threat == NONE for r in records[:20]))
        self.assertLessEqual(records[19].generation, 3)     # only occasional exploration

    def test_static_attacker_is_pushed_below_its_fair_share(self):
        d, _, policy, revenue = self.run_defense(adaptive=False)
        self.assertEqual(policy, 1.0)
        self.assertLess(revenue, 0.17)
        self.assertGreater(d.generation.number, 0)

    def test_adaptive_attacker_leads_to_uniform_tie_breaking(self):
        _, _, policy, revenue = self.run_defense(adaptive=True)
        self.assertEqual(policy, 0.5)
        self.assertLess(revenue, 0.2)

    def test_every_saved_generation_is_a_valid_quine(self):
        d, _, _, _ = self.run_defense(adaptive=True, epochs=60)
        files = sorted(os.listdir(d.srcg.workdir))
        self.assertEqual(len(files), d.generation.number + 1)
        for name in files:
            path = os.path.join(d.srcg.workdir, name)
            with open(path, encoding="utf-8") as f:
                load_generation(f.read(), path)             # raises if not a quine


if __name__ == "__main__":
    unittest.main()
