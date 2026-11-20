"""Success-guided selection of reset configurations from a fixed bank, AGX-free.

Every bank entry keeps its latest binary mission outcomes. A *sampling profile* turns them into the
adaptive distribution; the *draw mode* (``adaptive``, ``uniform``, ``sequential``) is separate.

``sgs_tempered``: the tempered kernel of Zhang, Guaman Castro, Yin et al., *A Balanced Data Diet*
(arXiv:2610.12465, Eqs. 1-2),

    w_i = (p_i + eps)^(kappa t) * (1 - p_i + eps)^(kappa (1 - t)),    P(i) = softmax(log(w_i + eps) / T),

with this repository's two smoothing additions: an unseen entry starts at ``prior`` and every
estimate gets ``prior_strength`` pseudo-outcomes at ``prior`` (the paper keeps H = 100 outcomes and
does not say how unseen entries start).

``alphanist_eq4``: AlphaNIST (arXiv:2610.13354, Eq. 4). ``p_i`` is the plain mean of the latest
``history`` outcomes, 0 for an unseen entry, and

    q_i  proportional to  eps_q + sqrt(p_i (1 - p_i)),

normalized over the valid entries, with no temperature and no smoothing. The paper gives no values
for ``eps_q`` or the history length; configurations choose them.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import ClassVar

import numpy as np

SAMPLING_MODES = ("adaptive", "uniform", "sequential")
"""``adaptive``: the sampling profile's distribution; ``uniform``: the same bank drawn uniformly (the
paper's baseline); ``sequential``: entry ``k mod N`` for episode ``k`` (fixed evaluation sets)."""
_MODE_ALIASES = {"sgs": "adaptive"}


def canonical_mode(mode: str) -> str:
    mode = _MODE_ALIASES.get(str(mode), str(mode))
    if mode not in SAMPLING_MODES:
        raise ValueError(f"mode must be one of {SAMPLING_MODES}, got {mode!r}")
    return mode


@dataclass(frozen=True)
class SgsParams:
    """The ``sgs_tempered`` profile."""

    name: ClassVar[str] = "sgs_tempered"

    target: float = 0.5
    """``t``: the success rate the kernel favours."""
    concentration: float = 1.0
    """``kappa``: how sharply the kernel peaks at ``t``."""
    temperature: float = 2.0
    """``T``: softmax temperature over the log-weights."""
    eps: float = 1e-4
    """Floor inside the kernel; keeps every probability non-zero."""
    history: int = 32
    """Outcomes kept per entry (circular buffer)."""
    prior: float = 0.5
    """Success estimate of an entry that has no outcome yet."""
    prior_strength: float = 2.0
    """Pseudo-outcomes at ``prior`` added to every estimate (0 = the paper's plain mean)."""

    def __post_init__(self):
        if not 0.0 <= self.target <= 1.0:
            raise ValueError(f"target must be in [0, 1], got {self.target}")
        if self.concentration < 0.0 or self.temperature <= 0.0 or self.eps <= 0.0:
            raise ValueError("concentration must be >= 0, temperature and eps > 0")
        if self.history < 1:
            raise ValueError(f"history must be >= 1, got {self.history}")
        if not 0.0 <= self.prior <= 1.0:
            raise ValueError(f"prior must be in [0, 1], got {self.prior}")
        if self.prior_strength < 0.0:
            raise ValueError(f"prior_strength must be >= 0, got {self.prior_strength}")

    def to_dict(self) -> dict:
        return asdict(self)

    def success_rate(self, history: SuccessHistory) -> np.ndarray:
        return history.success_rate(self.prior, self.prior_strength)

    def probabilities(self, success: np.ndarray, enabled: np.ndarray | None = None) -> np.ndarray:
        return sgs_probabilities(success, self, enabled)


@dataclass(frozen=True)
class AlphaNistEq4Params:
    """The ``alphanist_eq4`` profile."""

    name: ClassVar[str] = "alphanist_eq4"

    eps_q: float
    """Floor added to every weight; keeps every valid entry sampleable."""
    history: int
    """``L``: the latest outcomes averaged per entry."""

    def __post_init__(self):
        if not np.isfinite(self.eps_q) or self.eps_q <= 0.0:
            raise ValueError(f"eps_q must be finite and > 0, got {self.eps_q}")
        if isinstance(self.history, bool) or int(self.history) != self.history or self.history < 1:
            raise ValueError(f"history must be an integer >= 1, got {self.history}")

    def to_dict(self) -> dict:
        return asdict(self)

    def success_rate(self, history: SuccessHistory) -> np.ndarray:
        return history.success_rate(prior=0.0, strength=0.0)

    def probabilities(self, success: np.ndarray, enabled: np.ndarray | None = None) -> np.ndarray:
        p = np.clip(np.asarray(success, dtype=np.float64), 0.0, 1.0)
        return _normalize(self.eps_q + np.sqrt(p * (1.0 - p)), enabled)


@dataclass(frozen=True)
class TypePooledParams:
    """The ``type_pooled`` profile: AlphaNIST's kernel on success pooled per start type.

    A bank of ~1,300 entries meets a few thousand training episodes, so each entry is tried only a few
    times: per-entry rates are mostly noise, and Eq. 4 gives an untried or never-solved entry only its
    floor, which starves exactly the hard starts. Here all entries of one start type (``groups`` passed
    to the sampler, e.g. ``task/strategy``) share one window of their latest ``history`` outcomes,
    smoothed with ``prior_strength`` pseudo-outcomes at ``prior``:

        p_g = (successes_g + s prior) / (n_g + s),    w_g = eps_q + sqrt(p_g (1 - p_g)),

    and every entry of type g gets w_g, so a type is drawn with probability proportional to its share
    of the bank times w_g. An untried type starts at ``prior`` (0.5: full weight).

    With ``base="type"`` each entry's weight is divided by the number of enabled entries of its type, so a
    type is drawn with probability proportional to w_g alone (its bank share no longer counts) and an
    entry uniformly within its type.
    """

    name: ClassVar[str] = "type_pooled"
    grouped: ClassVar[bool] = True

    eps_q: float = 0.05
    """Floor added to every weight (AlphaNIST's eps_q)."""
    history: int = 64
    """Latest outcomes kept per start type."""
    prior: float = 0.5
    """Success rate of an untried type and centre of the smoothing."""
    prior_strength: float = 2.0
    """Pseudo-outcomes at ``prior`` added to every type's window."""
    base: str = "bank"
    """``"bank"``: a type's draw share is its bank share times w_g. ``"type"``: proportional to w_g alone."""

    def __post_init__(self):
        if not np.isfinite(self.eps_q) or self.eps_q <= 0.0:
            raise ValueError(f"eps_q must be finite and > 0, got {self.eps_q}")
        if isinstance(self.history, bool) or int(self.history) != self.history or self.history < 1:
            raise ValueError(f"history must be an integer >= 1, got {self.history}")
        if not 0.0 <= self.prior <= 1.0:
            raise ValueError(f"prior must be in [0, 1], got {self.prior}")
        if self.prior_strength < 0.0:
            raise ValueError(f"prior_strength must be >= 0, got {self.prior_strength}")
        if self.base not in ("bank", "type"):
            raise ValueError(f"base must be 'bank' or 'type', got {self.base!r}")

    def to_dict(self) -> dict:
        return asdict(self)

    def success_rate(self, history: SuccessHistory) -> np.ndarray:
        """Per-group rates when ``history`` holds one row per group (the sampler maps them to entries)."""
        return history.success_rate(self.prior, self.prior_strength)

    def probabilities(self, success: np.ndarray, enabled: np.ndarray | None = None) -> np.ndarray:
        """``success`` already holds each entry's type rate."""
        p = np.clip(np.asarray(success, dtype=np.float64), 0.0, 1.0)
        return _normalize(self.eps_q + np.sqrt(p * (1.0 - p)), enabled)


PROFILES = {
    SgsParams.name: SgsParams,
    AlphaNistEq4Params.name: AlphaNistEq4Params,
    TypePooledParams.name: TypePooledParams,
}


def make_profile(name: str = SgsParams.name, params: dict | None = None):
    if name not in PROFILES:
        raise ValueError(f"profile must be one of {tuple(PROFILES)}, got {name!r}")
    return PROFILES[name](**(params or {}))


def _normalize(weights: np.ndarray, enabled: np.ndarray | None) -> np.ndarray:
    w = np.asarray(weights, dtype=np.float64)
    if enabled is not None:
        enabled = np.asarray(enabled, dtype=bool)
        if not enabled.any():
            raise ValueError("every configuration is disabled")
        w = np.where(enabled, w, 0.0)
    return w / w.sum()


def sgs_logits(success: np.ndarray, params: SgsParams) -> np.ndarray:
    """``l_i = log(w_i + eps)`` for success estimates ``success`` (SGS Eq. 1)."""
    p = np.clip(np.asarray(success, dtype=np.float64), 0.0, 1.0)
    k, t, eps = params.concentration, params.target, params.eps
    w = (p + eps) ** (k * t) * (1.0 - p + eps) ** (k * (1.0 - t))
    return np.log(w + eps)


def sgs_probabilities(success: np.ndarray, params: SgsParams, enabled: np.ndarray | None = None) -> np.ndarray:
    """Sampling distribution ``softmax(l / T)`` (SGS Eq. 2); disabled entries get 0."""
    z = sgs_logits(success, params) / params.temperature
    if enabled is not None:
        enabled = np.asarray(enabled, dtype=bool)
        if not enabled.any():
            raise ValueError("every configuration is disabled")
        z = np.where(enabled, z, -np.inf)
    z = z - np.max(z)
    e = np.exp(z)
    return e / e.sum()


def validate_distribution(probabilities, num_configs: int) -> np.ndarray:
    """A pushed distribution: shape ``(num_configs,)``, finite, non-negative, with positive mass."""
    probs = np.asarray(probabilities, dtype=np.float64)
    if probs.shape != (num_configs,):
        raise ValueError(f"expected {num_configs} probabilities, got shape {probs.shape}")
    if not np.isfinite(probs).all() or (probs < 0.0).any():
        raise ValueError("probabilities must be finite and non-negative")
    if probs.sum() <= 0.0:
        raise ValueError("probabilities have no mass")
    return probs


class SuccessHistory:
    """The last ``history`` binary outcomes of each of ``num_configs`` configurations."""

    def __init__(self, num_configs: int, history: int):
        if num_configs < 1:
            raise ValueError("num_configs must be >= 1")
        self.num_configs = int(num_configs)
        self.history = int(history)
        self._outcomes = np.zeros((self.num_configs, self.history), dtype=np.int8)
        self._filled = np.zeros(self.num_configs, dtype=np.int64)
        self._next = np.zeros(self.num_configs, dtype=np.int64)
        self.visits = np.zeros(self.num_configs, dtype=np.int64)
        """Total outcomes ever recorded per configuration (not capped at ``history``)."""

    def record(self, index: int, success: bool) -> None:
        i = int(index)
        if not 0 <= i < self.num_configs:
            raise IndexError(f"configuration {i} outside [0, {self.num_configs})")
        self._outcomes[i, self._next[i]] = 1 if success else 0
        self._next[i] = (self._next[i] + 1) % self.history
        self._filled[i] = min(self._filled[i] + 1, self.history)
        self.visits[i] += 1

    def success_rate(self, prior: float, strength: float = 0.0) -> np.ndarray:
        """Mean of the kept outcomes with ``strength`` pseudo-outcomes at ``prior``;
        ``prior`` where nothing was recorded yet."""
        totals = self._outcomes.sum(axis=1, dtype=np.float64)
        filled = self._filled.astype(np.float64)
        smoothed = (totals + strength * prior) / np.maximum(filled + strength, 1e-12)
        return np.where(filled > 0, smoothed, float(prior))

    def state_dict(self) -> dict:
        return {
            "outcomes": self._outcomes.copy(),
            "filled": self._filled.copy(),
            "next": self._next.copy(),
            "visits": self.visits.copy(),
        }

    def load_state_dict(self, state: dict) -> None:
        outcomes = np.asarray(state["outcomes"], dtype=np.int8)
        filled = np.asarray(state["filled"], dtype=np.int64)
        nxt = np.asarray(state["next"], dtype=np.int64)
        visits = np.asarray(state["visits"], dtype=np.int64)
        if outcomes.shape != self._outcomes.shape:
            raise ValueError(f"history shape {outcomes.shape} does not match {self._outcomes.shape}")
        for name, a in (("filled", filled), ("next", nxt), ("visits", visits)):
            if a.shape != (self.num_configs,):
                raise ValueError(f"history {name} shape {a.shape} does not match ({self.num_configs},)")
        if (filled < 0).any() or (filled > self.history).any() or (nxt < 0).any() or (nxt >= self.history).any():
            raise ValueError("history counters out of range")
        self._outcomes, self._filled, self._next = outcomes.copy(), filled.copy(), nxt.copy()
        self.visits = visits.copy()


class SuccessGuidedSampler:
    """Draws configuration indices from a fixed buffer by a sampling profile, uniformly, or in sequence.

    ``disable(i)`` removes a configuration for good (a start state that failed its validity check at
    reset), in every mode. ``groups`` labels every configuration with its start type; a grouped profile
    (``type_pooled``) keeps one outcome window per group, every other profile ignores the labels.
    """

    def __init__(
        self,
        num_configs: int,
        params: SgsParams | AlphaNistEq4Params | TypePooledParams | None = None,
        mode: str = "adaptive",
        seed: int | None = None,
        groups=None,
    ):
        self.mode = canonical_mode(mode)
        self.params = params or SgsParams()
        self.num_configs = int(num_configs)
        self.history = SuccessHistory(self.num_configs, self.params.history)
        self.enabled = np.ones(self.num_configs, dtype=bool)
        self._rng = np.random.default_rng(seed)
        self._cursor = 0
        self.grouped = bool(getattr(self.params, "grouped", False))
        self.group_names: list[str] = []
        self.group_index = np.zeros(self.num_configs, dtype=np.int64)
        self.group_history: SuccessHistory | None = None
        if self.grouped:
            if groups is None or len(groups) != self.num_configs:
                raise ValueError(f"profile {self.params.name!r} needs one group label per configuration")
            labels = [str(g) for g in groups]
            self.group_names = sorted(set(labels))
            lookup = {g: i for i, g in enumerate(self.group_names)}
            self.group_index = np.array([lookup[g] for g in labels], dtype=np.int64)
            self.group_history = SuccessHistory(len(self.group_names), self.params.history)

    @property
    def profile(self) -> str:
        return self.params.name

    def record(self, index: int, success: bool) -> None:
        self.history.record(index, success)
        if self.group_history is not None:
            self.group_history.record(int(self.group_index[int(index)]), success)

    def disable(self, index: int) -> None:
        self.enabled[int(index)] = False
        if not self.enabled.any():
            raise RuntimeError("every configuration has been disabled")

    def success_rate(self) -> np.ndarray:
        """Per-configuration estimate; a grouped profile gives every configuration its group's rate."""
        if self.group_history is not None:
            return self.params.success_rate(self.group_history)[self.group_index]
        return self.params.success_rate(self.history)

    def group_success_rate(self) -> dict[str, float]:
        """Each group's rate (empty for a profile without groups)."""
        if self.group_history is None:
            return {}
        rates = self.params.success_rate(self.group_history)
        return {g: float(rates[i]) for i, g in enumerate(self.group_names)}

    def probabilities(self) -> np.ndarray:
        if self.mode == "adaptive":
            probs = self.params.probabilities(self.success_rate(), self.enabled)
            if self.grouped and getattr(self.params, "base", "bank") == "type":
                size = np.bincount(
                    self.group_index, weights=self.enabled.astype(np.float64), minlength=len(self.group_names)
                )
                probs = _normalize(probs / np.maximum(size[self.group_index], 1.0), self.enabled)
            return probs
        return _normalize(np.ones(self.num_configs), self.enabled)

    def sample(self, probabilities: np.ndarray | None = None) -> int:
        """Next configuration. ``probabilities`` (a distribution pushed from the process that sees every
        worker's outcomes) replaces the local estimate; entries this sampler disabled keep probability 0,
        and a pushed distribution with no mass on the remaining entries is an error, not a fallback."""
        if self.mode == "sequential":
            for _ in range(self.num_configs):
                index = self._cursor % self.num_configs
                self._cursor += 1
                if self.enabled[index]:
                    return index
            raise RuntimeError("every configuration has been disabled")
        if probabilities is None:
            probs = self.probabilities()
        else:
            probs = np.where(self.enabled, validate_distribution(probabilities, self.num_configs), 0.0)
            total = probs.sum()
            if total <= 0.0:
                raise RuntimeError(
                    "the pushed distribution has no mass on this worker's valid configurations: "
                    "every configuration is invalid either globally or locally"
                )
            probs = probs / total
        return int(self._rng.choice(self.num_configs, p=probs))

    def state_dict(self) -> dict:
        return {
            "mode": self.mode,
            "profile": self.profile,
            "params": self.params.to_dict(),
            "history": self.history.state_dict(),
            "enabled": self.enabled.copy(),
            "cursor": self._cursor,
            **(
                {"group_history": self.group_history.state_dict(), "group_names": list(self.group_names)}
                if self.group_history is not None
                else {}
            ),
        }

    def load_state_dict(self, state: dict) -> None:
        """All-or-nothing: a state that does not fit leaves this sampler unchanged."""
        enabled = np.asarray(state["enabled"], dtype=bool)
        if enabled.shape != (self.num_configs,):
            raise ValueError(f"enabled mask shape {enabled.shape} does not match ({self.num_configs},)")
        if not enabled.any():
            raise ValueError("every configuration is disabled in the saved state")
        history = SuccessHistory(self.num_configs, self.params.history)
        history.load_state_dict(state["history"])
        group_history = None
        if self.group_history is not None and "group_history" in state:
            if list(state.get("group_names", [])) != self.group_names:
                raise ValueError("saved group labels do not match this sampler's groups")
            group_history = SuccessHistory(len(self.group_names), self.params.history)
            group_history.load_state_dict(state["group_history"])
        self.history, self.enabled = history, enabled.copy()
        if group_history is not None:
            self.group_history = group_history
        self._cursor = int(state.get("cursor", 0))
