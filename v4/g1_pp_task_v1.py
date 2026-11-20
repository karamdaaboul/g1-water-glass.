"""G1 pick-and-place of a full water glass as an RL task (v4, 2026-11-19), on g1_pp_env.G1PPEnv.

User decisions (2026-11-19): our method with hard limits on the command change (and the spline chunks); soft constraints on
spill and glass speed; "glass down" not allowed: a failure, and a soft constraint whose cost comes from the learned tipping
predictor (tip_train.py, option B; no execution filter); the reward and cost critics see privileged simulator state, the
actor only what the real robot can measure.

Actions (8): right-arm joint velocity commands (7, x 1 rad/s) and a gripper command g (> 0 close, <= 0 open).
  close: the hand closes; the grasp completes CLOSE_STEPS after the close command (the fingers' closing time, 0.4 s): the weld
         engages then if the glass is in the hand's ring (glass axis within RING_R of the ring centre at grasp height +- RING_DZ,
         glass upright within RING_TILT). Welding at once let the closing fingers push the held glass to a ~6.5 deg tilt.
  open:  release gate (hard, in the env): a held glass is let go only when it stands on the table (base within PLACE_DZ of the
         table top, tilt < PLACE_TILT); otherwise the open command is ignored and the hand stays closed. The glass can never be
         dropped from the air.
Episode: glass at a random start spot, random target spot >= 12 cm away (g1_pp_env), robot at READY. Sparse reward 1 on success
(terminal): glass base within SUCCESS_DIST of the target, tilt < SUCCESS_TILT, released, ring centre > AWAY from the grasp point,
spilled <= SUCCESS_SPILL. Failures (terminal): glass down (tilt > 45 deg or off the table), spill > FAIL_SPILL. Timeout EPISODE_S.

Observation = [privileged (N_PRIV) | actor (N_ACTOR, ends with the last action)]; the actor network reads only the actor part.
  privileged: glass lin vel (3), glass ang vel (3), spilled fraction (1), water centre of mass in the glass frame rel. the cavity
              centre / cavity half size (3), tipping risk of the last action (1)
  actor:      arm q rel default (7), arm qd (7), hand closed (1), glass held (1), ring centre (3), glass base (3), glass up (3),
              target - glass base (3), last action (8)
  All velocities are finite differences over the policy step (Newton's velocity readback is ~3x below the actual motion).
Costs (3 channels, extras["costs"] (N, 3)):
  0 spill: increase of the spilled fraction this step
  1 down:  tipping risk p(glass down within 1 s | state, command) of the free glass (tip_model_action.pt), + 1 when it goes down
  2 speed: |glass velocity|^2 (m^2/s^2); the trainer scales it by (1 - gamma) (a discounted mean of the squared speed)
"""
from __future__ import annotations

import math
import os

import isaaclab.envs.mdp as mdp
import torch
import torch.nn as nn
from isaaclab.managers import ActionTerm, ActionTermCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from isaaclab.utils import math as mu

from g1_pp_env import (BOTTOM, CAVITY, DECIMATION, GRASP_H, R_IN, SIM_DT, TABLE_TOP, G1PPEnv, G1PPEnvCfg)  # first: puts ../ and ../v3 on sys.path
from g1_glass_env import RIGHT_ARM, JointVelocityServoActionCfg  # noqa: E402
from tip_features import HAND_BODIES, tip_features

HERE = os.path.dirname(os.path.abspath(__file__))
EPISODE_S = float(os.environ.get("G1P_EPISODE_S", "20.0"))
SUCCESS_DIST, SUCCESS_TILT, AWAY = 0.03, math.radians(5.0), 0.04
SUCCESS_SPILL = float(os.environ.get("G1P_SUCCESS_SPILL", "0.02"))
FAIL_SPILL, DOWN_TILT = 0.30, math.radians(45.0)
RING_R, RING_DZ, RING_TILT = 0.02, 0.03, math.radians(10.0)  # the scripted demonstrator's grip error is up to ~16 mm / 25 mm
PLACE_DZ, PLACE_TILT = 0.015, math.radians(8.0)  # release gate: base within 1.5 cm of the table, tilt < 8 deg
CLOSE_STEPS = 6
TIP_MODEL = os.environ.get("G1P_TIP_MODEL", os.path.join(HERE, "tipdata", "tip_model_action.pt"))
N_PRIV, N_ACTOR = 11, 36


class GripperCommandAction(ActionTerm):
    """One gripper command per env; the task env reads it (close / open with the ring test and the release gate)."""

    cfg: "GripperCommandActionCfg"

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._raw = torch.zeros(self.num_envs, 1, device=self.device)

    @property
    def action_dim(self) -> int:
        return 1

    @property
    def raw_actions(self):
        return self._raw

    @property
    def processed_actions(self):
        return self._raw

    def process_actions(self, actions):
        self._raw[:] = actions

    def apply_actions(self):
        pass

    def reset(self, env_ids=None) -> None:
        if env_ids is None:
            self._raw[:] = 0.0
        else:
            self._raw[env_ids] = 0.0


@configclass
class GripperCommandActionCfg(ActionTermCfg):
    class_type: type = GripperCommandAction
    asset_name: str = "robot"


# --- observation terms (env = G1PPTaskEnv) ---
def priv_obs(env):
    return env.privileged_obs()


def actor_obs(env):
    return env.actor_obs()


def success(env):
    return env.success_now()


def glass_down(env):
    return env.glass_down_now()


def spilled(env):
    return env.spilled_fraction() > FAIL_SPILL


@configclass
class TaskActionsCfg:
    right_arm = JointVelocityServoActionCfg(asset_name="robot", joint_names=RIGHT_ARM, max_vel=1.0, hold_first_step=True)
    gripper = GripperCommandActionCfg()


@configclass
class TaskObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        priv = ObsTerm(func=priv_obs)
        actor = ObsTerm(func=actor_obs)

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
    down = DoneTerm(func=glass_down)
    spilled = DoneTerm(func=spilled)


@configclass
class G1PPTaskEnvCfg(G1PPEnvCfg):
    actions: TaskActionsCfg = TaskActionsCfg()
    observations: TaskObservationsCfg = TaskObservationsCfg()
    rewards: TaskRewardsCfg = TaskRewardsCfg()
    terminations: TaskTerminationsCfg = TaskTerminationsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.episode_length_s = EPISODE_S


class TipModel(nn.Module):
    def __init__(self, path, device):
        super().__init__()
        ck = torch.load(path, map_location=device, weights_only=False)
        self.net = nn.Sequential(nn.Linear(ck["in_dim"], 256), nn.ELU(), nn.Linear(256, 256), nn.ELU(), nn.Linear(256, 1)).to(device)
        self.net.load_state_dict(ck["state_dict"])
        self.mean = torch.as_tensor(ck["mean"], device=device)
        self.std = torch.as_tensor(ck["std"], device=device)
        assert ck["variant"] == "action", "the env computes the risk from the state and the current command"

    @torch.no_grad()
    def forward(self, feats, action):
        x = (torch.cat([feats, action], -1) - self.mean) / self.std
        return torch.sigmoid(self.net(x)[:, 0])


class G1PPTaskEnv(G1PPEnv):
    cfg: G1PPTaskEnvCfg

    def __init__(self, cfg, **kwargs):
        d, n = cfg.sim.device, cfg.scene.num_envs
        self._ready = False
        self.tip_model = TipModel(TIP_MODEL, d)
        self.hand_closed = torch.zeros(n, dtype=torch.bool, device=d)
        self.close_steps = torch.zeros(n, dtype=torch.long, device=d)
        self.cavity_center = torch.tensor([0.0, 0.0, BOTTOM + 0.5 * CAVITY], device=d)
        self.cavity_half = torch.tensor([R_IN, R_IN, 0.5 * CAVITY], device=d)
        self.prev_spill = torch.zeros(n, device=d)
        self.costs = torch.zeros(n, 3, device=d)
        self.risk = torch.zeros(n, device=d)
        self.final_obs = None
        self._final_spill = torch.zeros(n, device=d)
        self._prev = None  # step-start positions (finite-difference velocities)
        self._final_speed2 = torch.zeros(n, device=d)
        super().__init__(cfg, **kwargs)
        self._arm_ids = self.action_manager.get_term("right_arm")._joint_ids
        self._hand_bodies = [self.robot.body_names.index(b) for b in HAND_BODIES]
        self._prev = self._pos_state()
        self._ready = True

    # --- state for features ---
    def _pos_state(self):
        gp, gq = self.glass_pose_w()
        o = self.scene.env_origins
        return {"gpos": gp - o, "gquat": gq.clone(), "bpos": self.robot.data.body_pos_w.torch[:, self._hand_bodies] - o[:, None],
                "q": self.robot.data.joint_pos.torch[:, self._arm_ids].clone(), "closed": self.hand_closed.float()}

    def _feats(self):
        return tip_features(self._pos_state(), self._prev, self.step_dt, TABLE_TOP)

    def glass_base_e(self):
        return self.glass_pose_w()[0] - self.scene.env_origins

    def glass_vel(self):
        if self._prev is None:
            return torch.zeros(self.num_envs, 3, device=self.device)
        return (self.glass_base_e() - self._prev["gpos"]) / self.step_dt

    def in_ring(self):
        g = self.grip_point_w()
        s = self.stem_mid_w()
        horiz = (g - s)[:, :2].norm(dim=-1)
        return (horiz < RING_R) & ((g - s)[:, 2].abs() < RING_DZ) & (self.glass_tilt() < RING_TILT)

    def on_table(self):
        return (self.glass_base_e()[:, 2] < TABLE_TOP + PLACE_DZ) & (self.glass_tilt() < PLACE_TILT)

    def glass_down_now(self):
        return (self.glass_tilt() > DOWN_TILT) | (self.glass_base_e()[:, 2] < TABLE_TOP - 0.03)

    def success_now(self):
        if not self._ready:
            return torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        place = (self.glass_base_e()[:, :2] - self.target_xy).norm(dim=-1) < SUCCESS_DIST
        away = (self.grip_point_w() - self.stem_mid_w()).norm(dim=-1) > AWAY
        return place & (self.glass_tilt() < SUCCESS_TILT) & ~self.grasped & away & (self.spilled_fraction() <= SUCCESS_SPILL)

    # --- observations ---
    def privileged_obs(self):
        n = self.num_envs
        if not getattr(self, "_ready", False):
            return torch.zeros(n, N_PRIV, device=self.device)
        gv = self.glass_vel()
        gw = tip_features(self._pos_state(), self._prev, self.step_dt, TABLE_TOP)[:, 6:9]
        spill = self.spilled_fraction()[:, None]
        if self.water_sim is not None:
            import warp as wp  # noqa: PLC0415
            inside = wp.to_torch(self.water_sim.inside).view(n, -1).float()
            pos, q = self.glass_pose_w()
            pts = self.water_sim.positions()
            local = mu.quat_apply_inverse(q[:, None].expand(-1, pts.shape[1], -1).reshape(-1, 4), (pts - pos[:, None]).reshape(-1, 3)).view(n, -1, 3)
            w = inside / inside.sum(-1, keepdim=True).clamp_min(1.0)
            com = ((local * w[..., None]).sum(1) - self.cavity_center) / self.cavity_half
        else:
            com = torch.zeros(n, 3, device=self.device)
        return torch.cat([gv, gw, spill, com, self.risk[:, None]], -1)

    def actor_obs(self):
        n = self.num_envs
        if not getattr(self, "_ready", False):
            return torch.zeros(n, N_ACTOR, device=self.device)
        q = self.robot.data.joint_pos.torch[:, self._arm_ids]
        q0 = self.robot.data.default_joint_pos.torch[:, self._arm_ids]
        qd = (q - self._prev["q"]) / self.step_dt
        gb = self.glass_base_e()
        up = mu.quat_apply(self.glass_pose_w()[1], torch.tensor([0.0, 0.0, 1.0], device=self.device).expand(n, -1))
        tgt = torch.cat([self.target_xy, torch.full((n, 1), TABLE_TOP, device=self.device)], -1) - gb
        ring = self.grip_point_w() - self.scene.env_origins
        return torch.cat([q - q0, qd, self.hand_closed.float()[:, None], self.grasped.float()[:, None], ring, gb, up, tgt,
                          self.action_manager.action], -1)

    # --- step ---
    def _gripper(self, g):
        ids = torch.arange(self.num_envs, device=self.device)
        close = g > 0.0
        # close: the hand closes; the weld engages if the glass is in the ring
        new_close = close & ~self.hand_closed
        if new_close.any():
            self.set_hand(ids[new_close], True)
        self.close_steps = torch.where(close, self.close_steps + 1, torch.zeros_like(self.close_steps))
        grab = close & ~self.grasped & (self.close_steps >= CLOSE_STEPS) & self.in_ring()
        if grab.any():
            self.set_weld(ids[grab], True)
        # open: a held glass only on the table (release gate); a free hand opens
        want_open = ~close & self.hand_closed
        gate = ~self.grasped | self.on_table()
        do_open = want_open & gate
        if (do_open & self.grasped).any():
            self.set_weld(ids[do_open & self.grasped], False)
        if do_open.any():
            self.set_hand(ids[do_open], False)
        self.hand_closed = (self.hand_closed | close) & ~do_open

    def step(self, action):
        action = torch.as_tensor(action, device=self.device, dtype=torch.float32)
        self._gripper(action[:, 7])
        feats = self._feats()
        free = ~self.grasped
        risk = torch.where(free, self.tip_model(feats, action[:, :7].clamp(-1, 1)), torch.zeros(self.num_envs, device=self.device))
        self._prev = self._pos_state()
        obs, rew, term, trunc, extras = super().step(action)
        done = term | trunc
        spill = self.spilled_fraction()
        d_spill = torch.where(done, self._final_spill - self.prev_spill, spill - self.prev_spill).clamp_min(0.0)
        self.prev_spill = torch.where(done, torch.zeros_like(spill), spill)
        went_down = done & self.termination_manager.get_term("down")
        v2 = torch.where(done, self._final_speed2, self.glass_vel().square().sum(-1))
        self.costs = torch.stack([d_spill, risk + went_down.float(), v2], -1)
        self.risk = torch.where(done, torch.zeros_like(risk), risk)
        extras["final_observation"] = self.final_obs if self.final_obs is not None else obs["policy"]
        extras["costs"] = self.costs
        extras["success"] = rew > 0.5
        return obs, rew, term, trunc, extras

    def _reset_idx(self, env_ids):
        ids = torch.arange(self.num_envs, device=self.device)[env_ids] if isinstance(env_ids, slice) else torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        if self._ready:
            self.final_obs = self.observation_manager.compute_group("policy").clone()
            self._final_spill = self.spilled_fraction().clone()
            self._final_speed2 = self.glass_vel().square().sum(-1)
        else:
            self._final_speed2 = torch.zeros(self.num_envs, device=self.device)
        super()._reset_idx(env_ids)
        self.hand_closed[ids] = False
        self.close_steps[ids] = 0
        if self._ready:
            cur = self._pos_state()
            for k in self._prev:
                self._prev[k][ids] = cur[k][ids]
            self.risk[ids] = 0.0
