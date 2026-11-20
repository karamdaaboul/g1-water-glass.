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

Observation = [privileged (N_PRIV) | constraint block (N_CONS) | actor (N_ACTOR, ends with the last action)]. The actor network reads
only the actor part; the hard limits read the constraint block and the actor part (all measurable on the robot); the critics all.
  privileged: glass lin vel (3), glass ang vel (3), spilled fraction (1), water centre of mass in the glass frame rel. the cavity
              centre / cavity half size (3), hand-glass distance (1, nearest hand point to the glass surface, m)
  constraint: ring-centre linear Jacobian (3 x 7, row-major, m per rad) and palm angular Jacobian (3 x 7), world frame, arm joints
  actor:      arm q rel default (7), arm qd (7), hand closed (1), glass held (1), ring centre (3), glass base (3), glass up (3),
              target - glass base (3), hand open direction (3, palm frame (1, 1, 0) / sqrt 2 in the world, horizontal), last action (8)
  obs_layout() gives the indices the trainer and the agent's hard limits read.
  All velocities are finite differences over the policy step (Newton's velocity readback is ~3x below the actual motion).
Costs (4 channels, extras["costs"] (N, 4)):
  0 spill:   increase of the spilled fraction this step
  1 disturb: 1 when the glass goes down (user 2026-11-19: events only; the tipping model is not used in training) + PUSH_K x the
             horizontal distance (m) the FREE glass moved this step (free before and after the step; user 2026-11-19, option 5: the
             policy had learned to push the glass to the target instead of carrying it). PUSH_K = 10: a 1 cm push costs 0.1.
  2 speed:   |glass velocity|^2 (m^2/s^2); the trainer scales it by (1 - gamma) (a discounted mean of the squared speed)
  3 contact: label, not a cost: a hand point within CONTACT_D of the free glass's surface after the step (the hard approach limit's
             predictor learns it)
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

from g1_pp_env import (BOTTOM, CAVITY, DECIMATION, GRASP_H, HAND_CLOSED, HAND_OPEN, R_IN, SIM_DT, TABLE_TOP, G1PPEnv, G1PPEnvCfg)  # first: puts ../ and ../v3 on sys.path
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
# 2026-11-19 (user items 2 + 4): the waist (yaw, pitch) joins the arm for reach; a 'released' flag (the glass was let go this
# episode) for the lift-off limit; actor observation noise and a per-episode action delay of 0 or 1 policy step.
WAIST = ["waist_yaw_joint", "waist_pitch_joint"]
ARM_J = RIGHT_ARM + WAIST
N_J = len(ARM_J)  # 9 joint velocity commands, then the gripper
GRIP = N_J
N_PRIV, N_CONS, N_ACTOR = 11, 6 * N_J, 2 * N_J + 3 + 15 + N_J + 1
OBS_NOISE = os.environ.get("G1P_OBS_NOISE", "1") == "1"
# Grasp (2026-11-20, user: real grasp): "friction" = the Dex3 holds the glass by contact alone; held = hand closed (>= CLOSE_STEPS)
# with the glass inside the ring, refreshed after every physics step. "weld" = the old palm-glass weld (v5 and earlier).
WELD = os.environ.get("G1P_GRASP", "friction") == "weld"
ACT_DELAY_P = float(os.environ.get("G1P_ACT_DELAY_P", "0.5"))  # share of episodes whose commands arrive one policy step late
PUSH_K = float(os.environ.get("G1P_PUSH_K", "10.0"))
# Partial-progress starts (option 3, user 2026-11-19 'if it didnt work use 3'): G1P_START_BANK=<make_start_bank.py npz> sends a share
# G1P_START_FRACTION (default 0.5) of the resets to a demo state (pregrasp / grasped / carry / lower), as the excavator's reset bank.
START_BANK = os.environ.get("G1P_START_BANK")
START_FRACTION = float(os.environ.get("G1P_START_FRACTION", "0.5"))
CONTACT_D = 0.02


def obs_layout():
    """Absolute indices into the task observation (before the trainer appends anything; it inserts only before the last action)."""
    a, j = N_PRIV + N_CONS, N_J
    return {"jg": N_PRIV, "jw": N_PRIV + 3 * j, "q": a, "qd": a + j, "closed": a + 2 * j, "held": a + 2 * j + 1,
            "released": a + 2 * j + 2, "ring": a + 2 * j + 3, "glass": a + 2 * j + 6, "up": a + 2 * j + 9, "target": a + 2 * j + 12,
            "open_dir": a + 2 * j + 15, "last_action": a + 2 * j + 18}


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


def cons_obs(env):
    return env.constraint_obs()


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
    right_arm = JointVelocityServoActionCfg(asset_name="robot", joint_names=ARM_J, max_vel=(1.0,) * 7 + (0.5, 0.5), hold_first_step=True,
                                            preserve_order=True)  # arm then waist (yaw, pitch), x 1 / 0.5 rad/s
    gripper = GripperCommandActionCfg()


@configclass
class TaskObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        priv = ObsTerm(func=priv_obs)
        cons = ObsTerm(func=cons_obs)
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
        self.hand_closed = torch.zeros(n, dtype=torch.bool, device=d)
        self.released = torch.zeros(n, dtype=torch.bool, device=d)
        self.delayed = torch.zeros(n, dtype=torch.bool, device=d)
        self.policy_action = torch.zeros(n, N_J + 1, device=d)
        self.close_steps = torch.zeros(n, dtype=torch.long, device=d)
        self.cavity_center = torch.tensor([0.0, 0.0, BOTTOM + 0.5 * CAVITY], device=d)
        self.cavity_half = torch.tensor([R_IN, R_IN, 0.5 * CAVITY], device=d)
        self.prev_spill = torch.zeros(n, device=d)
        self.costs = torch.zeros(n, 4, device=d)
        self.hand_dist = torch.full((n,), 1.0, device=d)
        self.final_obs = None
        self._final_spill = torch.zeros(n, device=d)
        self._prev = None  # step-start positions (finite-difference velocities)
        self._final_speed2 = torch.zeros(n, device=d)
        self._final_dist = torch.ones(n, device=d)
        self._final_free = torch.ones(n, dtype=torch.bool, device=d)
        self._final_gpos = torch.zeros(n, 3, device=d)
        self._final_clean = torch.zeros(n, 9, device=d)
        self.start_bank = None
        if START_BANK:
            import numpy as _np  # noqa: PLC0415
            b = _np.load(START_BANK)
            self.start_bank = {k: torch.as_tensor(b[k], device=d) for k in ("q", "closed", "held", "glass", "up", "target", "type")
                               + (("hand_q",) if "hand_q" in b.files else ())}
            if not WELD and "hand_q" not in self.start_bank:
                raise ValueError(f"{START_BANK}: the friction grasp needs a start bank with finger angles (demos v5+)")
            print(f"[task] start bank {START_BANK}: {len(b['type'])} starts, share {START_FRACTION}", flush=True)
        self.push_m = torch.zeros(n, device=d)
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
        return torch.cat([gv, gw, spill, com, self.hand_glass_distance()[:, None]], -1)

    def hand_points_w(self):
        """(N, 10, 3): the hand bodies' origins and the ring centre (world)."""
        return torch.cat([self.robot.data.body_pos_w.torch[:, self._hand_bodies], self.grip_point_w()[:, None]], 1)

    def hand_glass_distance(self):
        """Nearest hand point to the glass surface (m; the glass as its upright collider cylinder, in the glass frame)."""
        gp, gq = self.glass_pose_w()
        pts = self.hand_points_w()
        loc = mu.quat_apply_inverse(gq[:, None].expand(-1, pts.shape[1], -1).reshape(-1, 4), (pts - gp[:, None]).reshape(-1, 3)).view(self.num_envs, -1, 3)
        radial = loc[..., :2].norm(dim=-1) - (R_IN + 0.004)
        above = (loc[..., 2] - (BOTTOM + CAVITY)).clamp_min(0.0)
        below = (-loc[..., 2]).clamp_min(0.0)
        d = torch.sqrt(radial.clamp_min(0.0) ** 2 + (above + below) ** 2)
        return d.min(-1).values

    def constraint_obs(self):
        """Ring-centre linear Jacobian and palm angular Jacobian (world, arm + waist joints), 3 x N_J each, flattened row-major."""
        n = self.num_envs
        if not getattr(self, "_ready", False):
            return torch.zeros(n, N_CONS, device=self.device)
        J = self.robot.data.body_link_jacobian_w.torch[:, self.palm - 1][:, :, self._arm_ids]  # fixed base: row = body - 1
        pp, _ = self.palm_pose_w()
        r = self.grip_point_w() - pp
        rx = torch.zeros(n, 3, 3, device=self.device)
        rx[:, 0, 1], rx[:, 0, 2], rx[:, 1, 0], rx[:, 1, 2], rx[:, 2, 0], rx[:, 2, 1] = -r[:, 2], r[:, 1], r[:, 2], -r[:, 0], -r[:, 1], r[:, 0]
        Jg = J[:, :3] - rx @ J[:, 3:]
        return torch.cat([Jg.reshape(n, -1), J[:, 3:].reshape(n, -1)], -1)

    def open_dir(self):
        _, pq = self.palm_pose_w()
        u = mu.quat_apply(pq, torch.tensor([0.7071, 0.7071, 0.0], device=self.device).expand(self.num_envs, -1))
        u[:, 2] = 0.0
        return u / u.norm(dim=-1, keepdim=True).clamp_min(1e-6)

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
        od = self.open_dir()
        if OBS_NOISE:  # sensor noise: encoders, camera glass pose (target given in the glass frame inherits it)
            rn = lambda *sh: torch.randn(*sh, device=self.device)  # noqa: E731
            q, qd = q + 0.002 * rn(n, N_J), qd + 0.02 * rn(n, N_J)
            gb_n = gb + 0.003 * rn(n, 3)
            tgt = tgt - (gb_n - gb)
            gb = gb_n
            up = up + 0.005 * rn(n, 3)
            up = up / up.norm(dim=-1, keepdim=True)
            ring = ring + 0.001 * rn(n, 3)
        return torch.cat([q - q0, qd, self.hand_closed.float()[:, None], self.grasped.float()[:, None], self.released.float()[:, None],
                          ring, gb, up, tgt, od, self.policy_action], -1)

    def clean_state(self):
        """Noise-free ring centre, glass base (env frame) and glass up vector, (n, 9): for evaluation metrics only, never the policy."""
        n = self.num_envs
        up = mu.quat_apply(self.glass_pose_w()[1], torch.tensor([0.0, 0.0, 1.0], device=self.device).expand(n, -1))
        return torch.cat([self.grip_point_w() - self.scene.env_origins, self.glass_base_e(), up], -1)

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
        if not WELD or os.environ.get("G1P_NO_WELD") == "1":  # friction grasp: the fingers alone hold the glass
            grab = torch.zeros_like(grab)
        if grab.any():
            self.set_weld(ids[grab], True)
        # open: a held glass only on the table (release gate); a free hand opens
        want_open = ~close & self.hand_closed
        gate = ~self.grasped | self.on_table()
        do_open = want_open & gate
        if (do_open & self.grasped).any():
            self.released |= do_open & self.grasped
            if WELD:
                self.set_weld(ids[do_open & self.grasped], False)
        if do_open.any():
            self.set_hand(ids[do_open], False)
        self.hand_closed = (self.hand_closed | close) & ~do_open
        if not WELD:
            self.grasped = self._held_now()

    def _held_now(self):
        """Friction grasp: the glass counts as held while the closed hand has it in its ring."""
        return self.hand_closed & (self.close_steps >= CLOSE_STEPS) & self.in_ring()

    def _post_physics(self):
        if not WELD and getattr(self, "_ready", False):
            self.grasped = self._held_now()

    def step(self, action):
        action = torch.as_tensor(action, device=self.device, dtype=torch.float32)
        # latency: envs drawn delayed this episode apply the previous command (the observation shows what the policy sent)
        applied = torch.where(self.delayed[:, None], self.policy_action, action)
        self.policy_action = action.clone()
        action = applied
        free_before = ~self.grasped.clone()  # before the gripper command (a grasp or release this step: not a push)
        self._gripper(action[:, GRIP])
        self._prev = self._pos_state()
        g_before = self._prev["gpos"].clone()  # a copy: _reset_idx overwrites _prev of finished envs with the next episode's start
        obs, rew, term, trunc, extras = super().step(action)
        done = term | trunc
        spill = self.spilled_fraction()
        d_spill = torch.where(done, self._final_spill - self.prev_spill, spill - self.prev_spill).clamp_min(0.0)
        self.prev_spill = torch.where(done, torch.zeros_like(spill), spill)
        went_down = done & self.termination_manager.get_term("down")
        v2 = torch.where(done, self._final_speed2, self.glass_vel().square().sum(-1))
        dist = torch.where(done, self._final_dist, self.hand_glass_distance())
        free = torch.where(done, self._final_free, ~self.grasped)
        contact = (free & (dist < CONTACT_D)).float()
        g_after = torch.where(done[:, None], self._final_gpos, self.glass_base_e())
        push = torch.where(free_before & free, (g_after[:, :2] - g_before[:, :2]).norm(dim=-1), torch.zeros_like(dist))
        self.push_m = push
        self.costs = torch.stack([d_spill, went_down.float() + PUSH_K * push, v2, contact], -1)
        extras["final_observation"] = self.final_obs if self.final_obs is not None else obs["policy"]
        extras["costs"] = self.costs
        extras["success"] = rew > 0.5
        now = self.clean_state()  # after the reset of finished envs: their next episode's start
        extras["clean"] = torch.cat([torch.where(done[:, None], self._final_clean, now), now], -1)  # (n, 18): this step, next start
        return obs, rew, term, trunc, extras

    def _reset_idx(self, env_ids):
        ids = torch.arange(self.num_envs, device=self.device)[env_ids] if isinstance(env_ids, slice) else torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        if self._ready:
            self.final_obs = self.observation_manager.compute_group("policy").clone()
            self._final_spill = self.spilled_fraction().clone()
            self._final_speed2 = self.glass_vel().square().sum(-1)
            self._final_dist = self.hand_glass_distance()
            self._final_free = ~self.grasped
            self._final_gpos = self.glass_base_e().clone()
            self._final_clean = self.clean_state().clone()
        else:
            self._final_speed2 = torch.zeros(self.num_envs, device=self.device)
        super()._reset_idx(env_ids)
        self.hand_closed[ids] = False
        self.close_steps[ids] = 0
        self.released[ids] = False
        self.policy_action[ids] = 0.0
        self.delayed[ids] = torch.rand(len(ids), device=self.device) < ACT_DELAY_P
        if self._ready and self.start_bank is not None:
            sel = ids[torch.rand(len(ids), device=self.device) < START_FRACTION]
            if len(sel):
                self._bank_reset(sel)
        if self._ready:
            cur = self._pos_state()
            for k in self._prev:
                self._prev[k][ids] = cur[k][ids]

    def _bank_reset(self, sel):
        """Put envs ``sel`` into random start-bank states (arm joints, hand, glass pose, weld, target), at rest, water settled."""
        b = self.start_bank
        k = torch.randint(0, len(b["type"]), (len(sel),), device=self.device)
        q = b["q"][k]
        self.robot.write_joint_position_to_sim_index(position=q, joint_ids=self._arm_ids, env_ids=sel)
        self.robot.write_joint_velocity_to_sim_index(velocity=torch.zeros_like(q), joint_ids=self._arm_ids, env_ids=sel)
        closed = b["closed"][k].bool()
        hand = torch.where(closed[:, None], torch.tensor(HAND_CLOSED, device=self.device), torch.tensor(HAND_OPEN, device=self.device))
        hids = torch.as_tensor(self.hand_ids, device=self.device)
        # friction grasp: the fingers start where they were in the demo (on the glass), the targets squeeze (the closed pose)
        hand_pos = b["hand_q"][k] if "hand_q" in b else hand
        self.robot.write_joint_position_to_sim_index(position=hand_pos, joint_ids=hids, env_ids=sel)
        self.robot.write_joint_velocity_to_sim_index(velocity=torch.zeros_like(hand_pos), joint_ids=hids, env_ids=sel)
        self.robot.set_joint_position_target_index(target=hand, joint_ids=hids, env_ids=sel)
        term = self.action_manager.get_term("right_arm")
        term._q_des[sel] = q
        up = b["up"][k] / b["up"][k].norm(dim=-1, keepdim=True)
        z = torch.tensor([0.0, 0.0, 1.0], device=self.device).expand(len(sel), -1)
        axis = torch.cross(z, up, dim=-1)
        s_ = axis.norm(dim=-1, keepdim=True)
        ang = torch.atan2(s_, (z * up).sum(-1, keepdim=True))
        axis = torch.where(s_ > 1e-6, axis / s_.clamp_min(1e-9), torch.tensor([1.0, 0.0, 0.0], device=self.device).expand(len(sel), -1))
        quat = torch.cat([axis * torch.sin(ang / 2), torch.cos(ang / 2)], -1)  # xyzw, z -> up
        pos = b["glass"][k] + self.scene.env_origins[sel]
        self.glass.write_root_pose_to_sim_index(root_pose=torch.cat([pos, quat], -1), env_ids=sel)
        self.glass.write_root_velocity_to_sim_index(root_velocity=torch.zeros(len(sel), 6, device=self.device), env_ids=sel)
        self.target_xy[sel] = b["target"][k]
        self.scene.write_data_to_sim()
        self.sim.forward()
        self.robot.update(0.0)
        self.glass.update(0.0)
        held = b["held"][k].bool()
        if held.any() and WELD:
            self.set_weld(sel[held], True)  # relative pose = the restored palm-glass pose
        if not WELD:
            self.grasped[sel] = held
        self.hand_closed[sel] = closed
        self.close_steps[sel] = torch.where(closed, torch.full_like(self.close_steps[sel], CLOSE_STEPS), torch.zeros_like(self.close_steps[sel]))
        gp, gq = self.glass_pose_w()
        if self.water_sim is not None:
            self.water_sim.reset(sel, gp.clone(), gq.clone())
        self.prev_spill[sel] = 0.0
        if self._prev is not None:
            cur = self._pos_state()
            for key in self._prev:
                self._prev[key][sel] = cur[key][sel]
