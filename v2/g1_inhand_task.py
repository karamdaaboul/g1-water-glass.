"""G1 water-glass task v2 (2026-11-18): the glass is held IN the right Dex3 hand (g1_inhand_env.py); carry it from the home pose
to a target without spilling. Copied from the v1 task (g1_water_task.py) with all its fixes; only the glass and the bank differ.

Episode: start at HOME (glass upright, water at rest); target = a random entry of target_bank.pt (glass base position reachable with
the glass upright). Policy at 15 Hz, actions = right-arm joint velocity commands in [-1, 1] (x 1 rad/s), sparse reward:
  success (reward 1, terminate): glass base within SUCCESS_DIST of the target, glass speed < SUCCESS_SPEED, tilt < SUCCESS_TILT, at most
           SUCCESS_SPILL of the water outside the glass
  failure (terminate): more than FAIL_SPILL spilled, or the glass base leaves the safe box (into the body or below the tray rim)
  timeout: EPISODE_S
Per-step cost (info, for a cost critic): increase of the spilled fraction this step (>= 0). The pre-reset observation of finished
envs is kept as the final observation (the off-policy learner needs the true successor state).
Observation (37): arm q rel (7), arm qd (7), glass base pos rel home (3), glass up (3), glass lin vel (3), target - glass (3),
water in glass (1), water centre of mass in the glass frame rel the cavity centre / cavity half size (3), last action (7, last).
Velocities (arm qd, glass velocity, the success speed test) are step means, (x_t - x_{t-1}) / step_dt: the joint and body velocities
read back from the coupled Newton solver disagree with the motion (2026-11-18: 0.19 rad/s rms off on an overdamped arm that tracks
its command to 0.001 rad/s; glass read 0.06 m/s while it moved 0.23 m/s). The water uses pose differences, so the physics is unaffected.
"""
from __future__ import annotations

import os

import isaaclab.envs.mdp as mdp
import torch
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from isaaclab.utils import math as math_utils

from g1_glass_env import RIGHT_ARM
from g1_inhand_env import BOTTOM, CAVITY, DECIMATION, R_IN, SIM_DT  # noqa: F401
from g1_inhand_env import G1InHandEnv as G1WaterEnv
from g1_inhand_env import G1InHandEnvCfg as G1WaterEnvCfg
from g1_inhand_env import glass_home_base as _cup_home_base_pos

SUCCESS_DIST, SUCCESS_SPEED, SUCCESS_TILT, SUCCESS_SPILL = 0.04, 0.10, 15.0 / 57.2958, 0.10
FAIL_SPILL = 0.30
SAFE_LO, SAFE_HI = (0.12, -0.55, 0.50), (0.70, 0.35, 1.15)  # right-hand workspace
EPISODE_S = 8.0
MIN_TARGET_DIST = 0.12  # targets at least this far from the home glass position
BANK = os.path.join(os.path.dirname(os.path.abspath(__file__)), "target_bank.pt")


# --- observation terms (env = G1WaterTaskEnv) ---
def cup_pos_rel_home(env):
    return env.cup_pos_e() - env.home_pos


def cup_up(env):
    _, q = env.cup_pose_w()
    return math_utils.quat_apply(q, env.z_axis)


def cup_lin_vel(env):
    return env.cup_vel_fd()


def arm_qd(env):
    return env.arm_qd_fd()


def target_rel(env):
    return env.target_pos - env.cup_pos_e()


def water_state(env):
    inside = env.water_in_cup().float()
    frac = inside.mean(-1, keepdim=True)
    pos, q = env.cup_pose_w()
    pts = env.water.data.particle_pos_w.torch
    local = math_utils.quat_apply_inverse(q[:, None].expand(-1, pts.shape[1], -1), pts - pos[:, None])
    w = inside / inside.sum(-1, keepdim=True).clamp_min(1.0)
    com = (local * w[..., None]).sum(1) - env.cavity_center
    return torch.cat([frac, com / env.cavity_half], -1)


# --- reward / terminations ---
def success(env):
    return env.success_now()


def failed_spill(env):
    return env.spilled_fraction() > FAIL_SPILL


def left_safe_box(env):
    p = env.cup_pos_e()
    return ((p < env.safe_lo) | (p > env.safe_hi)).any(-1)


@configclass
class TaskObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        arm_q = ObsTerm(func=mdp.joint_pos_rel, params={"asset_cfg": SceneEntityCfg("robot", joint_names=RIGHT_ARM)})
        arm_qd = ObsTerm(func=arm_qd)
        cup_pos = ObsTerm(func=cup_pos_rel_home)
        cup_up = ObsTerm(func=cup_up)
        cup_vel = ObsTerm(func=cup_lin_vel)
        target = ObsTerm(func=target_rel)
        water = ObsTerm(func=water_state)
        last_action = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class TaskRewardsCfg:
    success = RewTerm(func=success, weight=1.0 / (SIM_DT * DECIMATION))  # the reward manager multiplies by the step dt: 1 per success


@configclass
class TaskTerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    success = DoneTerm(func=success)
    spilled = DoneTerm(func=failed_spill)
    unsafe = DoneTerm(func=left_safe_box)


OBS_TERMS = ["arm_q", "arm_qd", "cup_pos", "cup_up", "cup_vel", "target", "water", "last_action"]
OBS_TERM_DIMS = [7, 7, 3, 3, 3, 3, 4, 7]


@configclass
class G1WaterTaskEnvCfg(G1WaterEnvCfg):
    observations: TaskObservationsCfg = TaskObservationsCfg()
    rewards: TaskRewardsCfg = TaskRewardsCfg()
    terminations: TaskTerminationsCfg = TaskTerminationsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.episode_length_s = EPISODE_S


class G1WaterTaskEnv(G1WaterEnv):
    cfg: G1WaterTaskEnvCfg

    def __init__(self, cfg, **kwargs):
        # everything the observation terms read exists before the base constructor computes them once
        d, n = cfg.sim.device, cfg.scene.num_envs
        self._ready = False
        bank = torch.load(BANK)
        self.bank_q, self.bank_pos = bank["q"].to(d), bank["cup_pos"].to(d)
        self.home_pos = torch.tensor(_cup_home_base_pos(), device=d)
        self.z_axis = torch.tensor([0.0, 0.0, 1.0], device=d).expand(n, -1)
        self.cavity_center = torch.tensor([0.0, 0.0, BOTTOM + 0.5 * CAVITY], device=d)
        self.cavity_half = torch.tensor([R_IN, R_IN, 0.5 * CAVITY], device=d)
        self.safe_lo, self.safe_hi = torch.tensor(SAFE_LO, device=d), torch.tensor(SAFE_HI, device=d)
        self.eligible = torch.nonzero((self.bank_pos - self.home_pos).norm(dim=-1) >= MIN_TARGET_DIST).flatten()
        self.target_idx = self.eligible[torch.randint(len(self.eligible), (n,), device=d)]
        # Evaluation: G1W_EVAL_TARGETS=<seed> gives env i its k-th episode's target from one seeded permutation of the eligible
        # targets, entry k * num_envs + i, so every policy sees the same targets whatever its episode lengths.
        ev = os.environ.get("G1W_EVAL_TARGETS")
        self.eval_perm = None if ev is None else torch.randperm(len(self.eligible), generator=torch.Generator().manual_seed(int(ev))).to(d)
        self.episode_count = torch.zeros(n, dtype=torch.long, device=d)
        self.target_pos = self.bank_pos[self.target_idx].clone()
        self.target_q = self.bank_q[self.target_idx].clone()
        self.prev_spill = torch.zeros(n, device=d)
        self.spill_cost = torch.zeros(n, device=d)
        self.final_obs = None
        self._final_spill = torch.zeros(n, device=d)
        self._q_prev = self._p_prev = None  # step-start arm q and glass position (step-mean velocities)
        super().__init__(cfg, **kwargs)
        self._arm_ids = self.action_manager.get_term("right_arm")._joint_ids.long()
        self._ready = True

    def cup_pos_e(self):
        pos, _ = self.cup_pose_w()
        return pos - self.scene.env_origins

    def success_now(self):
        if not self._ready:
            return torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        near = (self.cup_pos_e() - self.target_pos).norm(dim=-1) < SUCCESS_DIST
        slow = self.cup_vel_fd().norm(dim=-1) < SUCCESS_SPEED
        return near & slow & (self.cup_tilt() < SUCCESS_TILT) & (self.spilled_fraction() <= SUCCESS_SPILL)

    def _arm_q(self):
        return self.robot.data.joint_pos.torch[:, self._arm_ids]

    def arm_qd_fd(self):
        if self._q_prev is None:
            return torch.zeros(self.num_envs, 7, device=self.device)
        return (self._arm_q() - self._q_prev) / self.step_dt

    def cup_vel_fd(self):
        if self._p_prev is None:
            return torch.zeros(self.num_envs, 3, device=self.device)
        return (self.cup_pos_e() - self._p_prev) / self.step_dt

    def step(self, action):
        self._q_prev, self._p_prev = self._arm_q().clone(), self.cup_pos_e().clone()
        obs, rew, term, trunc, extras = super().step(action)
        spill = self.spilled_fraction()
        done = term | trunc
        # cost = increase of the spill this step; finished envs: their last increase (prev_spill holds the pre-reset value)
        self.spill_cost = torch.where(done, self._final_spill - self.prev_spill, spill - self.prev_spill).clamp_min(0.0)
        self.prev_spill = torch.where(done, torch.zeros_like(spill), spill)
        extras["final_observation"] = self.final_obs if self.final_obs is not None else obs["policy"]
        extras["costs"] = self.spill_cost
        extras["success"] = rew > 0.5
        return obs, rew, term, trunc, extras

    def _reset_idx(self, env_ids):
        ids = torch.arange(self.num_envs, device=self.device)[env_ids] if isinstance(env_ids, slice) else torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        if self._ready:
            # the true successor observation and spill of the finished envs, before the reset overwrites them
            fo = self.observation_manager.compute_group("policy")
            self.final_obs = fo.clone()
            self._final_spill = self.spilled_fraction().clone()
            if self.eval_perm is None:
                self.target_idx[ids] = self.eligible[torch.randint(len(self.eligible), (len(ids),), device=self.device)]
            else:
                k = (self.episode_count[ids] * self.num_envs + ids) % len(self.eval_perm)
                self.target_idx[ids] = self.eligible[self.eval_perm[k]]
                self.episode_count[ids] += 1
            self.target_pos[ids] = self.bank_pos[self.target_idx[ids]]
            self.target_q[ids] = self.bank_q[self.target_idx[ids]]
        super()._reset_idx(env_ids)
        if self._q_prev is not None:  # a new episode starts at rest
            self._q_prev[ids] = self._arm_q()[ids]
            self._p_prev[ids] = self.cup_pos_e()[ids]
