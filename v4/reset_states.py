"""Reset states of our method, for the G1 water-glass task (2026-11-20).

User: "the idea we want to try our method on other tasks, so the used method should be the same, so you have to use reset
state it can be success-guided or uniform". The excavator paper's method (agxcave ``Excavator-MultiTask-3D-Sparse-v1``, the
reset curriculum of ``multi_task_3d_cfg.MultiTask3DExcavationV1EnvCfg``) keeps a fixed bank of start states and draws every
reset from it, after AlphaNIST (https://alpha-nist.github.io/). Here the bank holds

    standard starts   the glass and the target at seeded random spots (the env's layout rule), the robot at READY; as many as
                      the demo starts times ``standard_fraction / (1 - standard_fraction)`` (0.5: half the bank, as the
                      excavator's ``standard_fraction``)
    demo starts       the start-bank file (make_start_bank.py): pregrasp, grasped, carry, lower and, with --recovery, slipped,
                      near_miss, far_pregrasp, each a state of a successful demonstration

and every episode's binary success is recorded against the entry it started from. Draw modes:

    adaptive   ``type_pooled`` (the paper's): every start type pools its latest 64 outcomes with 2 pseudo-outcomes at 0.5,
               p_g = (successes + 2 * 0.5) / (n + 2), w_g = eps_q + sqrt(p_g (1 - p_g)) with eps_q = 0.05, and each entry of
               type g is drawn with probability proportional to w_g (a type's share = its bank share times w_g)
    uniform    the same bank drawn uniformly (the paper's baseline)

``sgs_sampler.py`` is agxcave's ``agxcave/agxpolicy/sgs/sampler.py`` byte for byte (feat/mt3d-v1 eebcde5, sha256
79f0a244...0b9e9), so both tasks run the same sampler code.
"""

from __future__ import annotations

import numpy as np

from sgs_sampler import SuccessGuidedSampler, make_profile

PROFILE = "type_pooled"
PAPER_PARAMS = {"eps_q": 0.05, "history": 64, "prior": 0.5, "prior_strength": 2.0}  # agxcave multi_task_3d_cfg.V1_SGS
STANDARD = "standard"


class ResetStates:
    """The bank's entries: ``[0, n_standard)`` standard layouts, then the demo starts in file order."""

    def __init__(self, types: list[str], type_index: np.ndarray, standard_start: np.ndarray, standard_target: np.ndarray,
                 mode: str = "adaptive", seed: int = 0):
        self.n_standard = len(standard_start)
        self.n_demo = len(type_index)
        self.standard_start, self.standard_target = np.asarray(standard_start), np.asarray(standard_target)
        self.groups = [STANDARD] * self.n_standard + [str(types[int(t)]) for t in type_index]
        self.sampler = SuccessGuidedSampler(len(self.groups), make_profile(PROFILE, PAPER_PARAMS), mode=mode, seed=seed,
                                            groups=self.groups)
        self.episodes = 0
        self._group_of = np.array(self.groups)

    @staticmethod
    def standard_count(n_demo: int, standard_fraction: float) -> int:
        return int(round(n_demo * standard_fraction / (1.0 - standard_fraction)))

    def draw(self, n: int) -> np.ndarray:
        return np.array([self.sampler.sample() for _ in range(n)], dtype=np.int64)

    def record(self, entry: int, success: bool) -> None:
        self.sampler.record(int(entry), bool(success))
        self.episodes += 1

    def summary(self) -> dict:
        """Per start type: its pooled success estimate and its current draw share."""
        probs = self.sampler.probabilities()
        share = {g: float(probs[self._group_of == g].sum()) for g in sorted(set(self.groups))}
        rate = self.sampler.group_success_rate()
        return {"mode": self.sampler.mode, "episodes": self.episodes,
                "types": {g: {"p": round(rate.get(g, float("nan")), 3), "share": round(share[g], 3)} for g in share}}
