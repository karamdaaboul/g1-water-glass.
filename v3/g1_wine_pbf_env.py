"""G1 wine glass with Warp position-based-fluid water (v3, 2026-11-19).

Same robot and glass as g1_wine_env.py (glass welded inside the Dex3 hand, palm-up home, damped drives, first step held), but the
water is pbf_water.PBFWater instead of Newton's implicit MPM: the MPM water at RL speed spilled only past ~20 deg where real water
spills at ~5 deg (tilt_check.py); the PBF water matches hydrostatics in pbf_validate.py (8 cm bowl, 1 cm free: spills from
~15-20 deg tilt and ~2 m/s^2 lateral acceleration) at 247 env-steps/s for 64 glasses.

Physics: MuJoCo-Warp only (no coupler). Water: one-way, driven by the glass pose; advanced once per policy step inside
scene.update (after the robot's physics, before terminations/rewards), from the glass pose at the start of the step to the pose
after it (linear position / slerp per water substep). Resets: robot -> FK -> water to its pre-settled state in the glass.
G1V_SHOW_WATER=1 draws env 0's water with Isaac Lab VisualizationMarkers (videos).
"""
from __future__ import annotations

import os
import sys

import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
import torch
from isaaclab.assets import AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.utils import math as math_utils
from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from g1_wine_env import (  # noqa: E402
    BOTTOM, CAVITY, DECIMATION, FREE, H_GLASS, R_IN, R_OUT, RIGHT_ARM, RIGHT_HAND, SIM_DT, GRIPPER_POSE, HAND_SCALE, g1_cfg,
)
from g1_glass_env import JointVelocityServoActionCfg  # noqa: E402
from pbf_water import PBFWater  # noqa: E402

PBF_SPACING = float(os.environ.get("G1V_PBF_SPACING", "0.005"))
PBF_SUBSTEPS = int(os.environ.get("G1V_PBF_SUBSTEPS", "16"))
PBF_ITERS = int(os.environ.get("G1V_PBF_ITERS", "3"))


@configclass
class G1WinePbfSceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(prim_path="/World/GroundPlane", spawn=sim_utils.GroundPlaneCfg())
    robot = g1_cfg()
    light = AssetBaseCfg(prim_path="/World/light", spawn=sim_utils.DistantLightCfg(intensity=2500.0))


@configclass
class ActionsCfg:
    right_arm = JointVelocityServoActionCfg(asset_name="robot", joint_names=RIGHT_ARM, max_vel=1.0, hold_first_step=True)


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        arm_q = ObsTerm(func=mdp.joint_pos_rel, params={"asset_cfg": SceneEntityCfg("robot", joint_names=RIGHT_ARM)})
        last_action = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventsCfg:
    reset_robot = EventTerm(func=mdp.reset_joints_by_offset, mode="reset",
                            params={"asset_cfg": SceneEntityCfg("robot"), "position_range": (0.0, 0.0), "velocity_range": (0.0, 0.0)})


@configclass
class RewardsCfg:
    alive = RewTerm(func=mdp.is_alive, weight=0.0)


def spilled_too_much(env, limit: float = 0.3):
    return env.spilled_fraction() > limit


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    spilled = DoneTerm(func=spilled_too_much, params={"limit": 0.3}) if os.environ.get("G1V_NO_SPILL_TERM") != "1" else None


@configclass
class G1WinePbfEnvCfg(ManagerBasedRLEnvCfg):
    scene: G1WinePbfSceneCfg = G1WinePbfSceneCfg(num_envs=4, env_spacing=2.5)
    actions: ActionsCfg = ActionsCfg()
    observations: ObservationsCfg = ObservationsCfg()
    events: EventsCfg = EventsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    def __post_init__(self):
        self.decimation = DECIMATION
        self.episode_length_s = 12.0
        self.sim = sim_utils.SimulationCfg(dt=SIM_DT, physics=NewtonCfg(
            solver_cfg=MJWarpSolverCfg(solver="newton", integrator="implicitfast", njmax=256, nconmax=64, iterations=100, ls_iterations=50),
            num_substeps=3, use_cuda_graph=True))
        self.sim.use_newton_actuators = True


def set_mpm_capacity(cfg, **_):
    """No MPM in this env (kept for scripts written against the MPM envs)."""


class G1WinePbfEnv(ManagerBasedRLEnv):
    cfg: G1WinePbfEnvCfg

    def __init__(self, cfg, **kwargs):
        self._cup_body = None
        self._hand_ids = None
        self.water_sim = PBFWater(cfg.scene.num_envs, R_IN, R_OUT, BOTTOM, H_GLASS, CAVITY - FREE, cfg.sim.device,
                                  spacing=PBF_SPACING, substeps=PBF_SUBSTEPS, iterations=PBF_ITERS)
        self.water_sim.presettle(CAVITY - FREE)
        self._pose0 = None
        self._just_reset = torch.zeros(cfg.scene.num_envs, dtype=torch.bool, device=cfg.sim.device)
        self._markers = None
        super().__init__(cfg, **kwargs)
        update = self.scene.update

        def update_with_water(dt):  # after the robot's physics, before terminations and rewards
            update(dt)
            self._advance_water()

        self.scene.update = update_with_water
        if os.environ.get("G1V_SHOW_WATER") == "1":
            from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg  # noqa: PLC0415
            self._markers = VisualizationMarkers(VisualizationMarkersCfg(prim_path="/Visuals/Water", markers={
                "drop": sim_utils.SphereCfg(radius=0.5 * PBF_SPACING, visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.15, 0.45, 0.95)))}))

    @property
    def robot(self):
        return self.scene["robot"]

    @property
    def cup_body(self):
        if self._cup_body is None:
            self._cup_body = self.robot.body_names.index("Cup")
        return self._cup_body

    @property
    def hand_ids(self):
        if self._hand_ids is None:
            self._hand_ids = [self.robot.joint_names.index(j) for j in RIGHT_HAND]
        return self._hand_ids

    def cup_pose_w(self):
        return self.robot.data.body_pos_w.torch[:, self.cup_body], self.robot.data.body_quat_w.torch[:, self.cup_body]

    def step(self, action):
        p, q = self.cup_pose_w()
        self._pose0 = (p.clone(), q.clone())
        out = super().step(action)
        if self._markers is not None:
            self._markers.visualize(translations=self.water_sim.positions()[0])
        return out

    def _advance_water(self):
        if self._pose0 is None:
            return
        p1, q1 = self.cup_pose_w()
        p0, q0 = self._pose0
        self.water_sim.step(p0, q0, p1.clone(), q1.clone(), self.step_dt)

    def _reset_idx(self, env_ids):
        super()._reset_idx(env_ids)
        ids = torch.arange(self.num_envs, device=self.device)[env_ids] if isinstance(env_ids, slice) else torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        grip = torch.tensor([HAND_SCALE * v for v in GRIPPER_POSE], device=self.device).expand(len(ids), -1)
        self.robot.set_joint_position_target_index(target=grip, joint_ids=self.hand_ids, env_ids=ids)
        self.scene.write_data_to_sim()
        self.sim.forward()
        self.robot.update(0.0)
        p, q = self.cup_pose_w()
        self.water_sim.reset(ids, p.clone(), q.clone())

    # water measures in the glass frame (the PBF classifies in/out of the bowl every step)
    def spilled_fraction(self):
        return self.water_sim.spilled_fraction()

    def water_points_w(self):
        return self.water_sim.positions()

    def water_in_cup(self):
        import warp as wp  # noqa: PLC0415
        return wp.to_torch(self.water_sim.inside).view(self.num_envs, -1).bool()

    def cup_tilt(self):
        _, quat = self.cup_pose_w()
        up = math_utils.quat_apply(quat, torch.tensor([0.0, 0.0, 1.0], device=self.device).expand(quat.shape[0], -1))
        return torch.acos(up[:, 2].clamp(-1.0, 1.0))


# names the v3 scripts import
G1InHandEnv, G1InHandEnvCfg = G1WinePbfEnv, G1WinePbfEnvCfg
