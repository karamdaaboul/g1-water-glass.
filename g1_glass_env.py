"""G1 water-glass task on Isaac Lab main + Newton (MuJoCo-Warp): stage 1, the robot only.

A fixed-base Unitree G1 (29 DoF + Dex3 hands, isaaclab_assets G1_29DOF_CFG) whose right arm (7 joints) is driven by joint
velocity commands, as AGXCave drives the excavator. The glass and the water are added in later stages of this file.
"""
from __future__ import annotations

import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab_assets.robots.unitree import G1_29DOF_CFG
from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg

import torch
from isaaclab.envs.mdp.actions.actions_cfg import JointActionCfg
from isaaclab.envs.mdp.actions.joint_actions import JointAction


class JointVelocityServoAction(JointAction):
    """Velocity commands to a position servo, as a hydraulic machine's joystick: the policy's action a in [-1, 1] per joint is a
    velocity command v = a * max_vel; every physics step the servo target integrates it, q_des += v dt (clipped to the soft
    joint limits), and the PD actuator tracks q_des with velocity target v. Reset: q_des = the current joint position."""

    cfg: "JointVelocityServoActionCfg"

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._max_vel = torch.as_tensor(cfg.max_vel, device=self.device, dtype=torch.float32).expand(self.action_dim).clone()
        self._q_des = self._asset.data.joint_pos.torch[:, self._joint_ids].clone()
        lim = self._asset.data.soft_joint_pos_limits.torch[:, self._joint_ids]
        self._q_lo, self._q_hi = lim[..., 0].clone(), lim[..., 1].clone()

    def process_actions(self, actions: torch.Tensor):
        self._raw_actions[:] = actions
        self._processed_actions[:] = actions.clamp(-1.0, 1.0) * self._max_vel

    def apply_actions(self):
        # Isaac Lab main: with physics-owned decimation (Newton) apply_action runs once per policy step and the physics runs all
        # substeps in one call, so the target advances one policy step; else once per physics step. The env sets the flag after
        # the managers are built, so it is read here.
        self._dt = self._env.step_dt if getattr(self._env, "_physics_handles_decimation", False) else self._env.physics_dt
        v = self._processed_actions
        if getattr(self.cfg, "hold_first_step", False):
            # First step of an episode: hold still. The reset teleports the arm, and an MPM collider whose velocity is a finite
            # difference of poses would see that jump as a huge velocity (water blasted out of the glass). The observation still
            # shows the policy's action; physically this is a one-step start delay.
            v = torch.where((self._env.episode_length_buf == 0)[:, None], torch.zeros_like(v), v)
        self._q_des = torch.minimum(torch.maximum(self._q_des + v * self._dt, self._q_lo), self._q_hi)
        self._asset.set_joint_position_target_index(target=self._q_des, joint_ids=self._joint_ids)
        self._asset.set_joint_velocity_target_index(target=v, joint_ids=self._joint_ids)

    def reset(self, env_ids=None) -> None:
        super().reset(env_ids)
        q = self._asset.data.joint_pos.torch[:, self._joint_ids]
        if env_ids is None:
            self._q_des[:] = q
        else:
            self._q_des[env_ids] = q[env_ids]


@configclass
class JointVelocityServoActionCfg(JointActionCfg):
    class_type: type = JointVelocityServoAction
    max_vel: float | tuple = 1.5  # rad/s at |a| = 1
    hold_first_step: bool = False  # zero motion on the first step of each episode (see apply_actions)


RIGHT_ARM = ["right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint", "right_elbow_joint",
             "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint"]
SIM_DT = 1.0 / 120.0
DECIMATION = 8  # 15 Hz policy, as AGXCave


def g1_fixed_cfg():
    cfg = G1_29DOF_CFG.copy()
    cfg.spawn.fix_root_link = True
    cfg.prim_path = "{ENV_REGEX_NS}/Robot"
    cfg.init_state = cfg.init_state.replace(pos=(0.0, 0.0, 0.80))
    return cfg


@configclass
class G1GlassSceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg())
    robot = g1_fixed_cfg()
    light = AssetBaseCfg(prim_path="/World/light", spawn=sim_utils.DistantLightCfg(intensity=2500.0))


@configclass
class ActionsCfg:
    right_arm = JointVelocityServoActionCfg(asset_name="robot", joint_names=RIGHT_ARM, max_vel=1.5)


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        arm_q = ObsTerm(func=mdp.joint_pos_rel, params={"asset_cfg": SceneEntityCfg("robot", joint_names=RIGHT_ARM)})
        arm_qd = ObsTerm(func=mdp.joint_vel_rel, params={"asset_cfg": SceneEntityCfg("robot", joint_names=RIGHT_ARM)})
        last_action = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class RewardsCfg:
    alive = RewTerm(func=mdp.is_alive, weight=0.0)


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)


@configclass
class G1GlassEnvCfg(ManagerBasedRLEnvCfg):
    scene: G1GlassSceneCfg = G1GlassSceneCfg(num_envs=4, env_spacing=2.5)
    actions: ActionsCfg = ActionsCfg()
    observations: ObservationsCfg = ObservationsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    def __post_init__(self):
        self.decimation = DECIMATION
        self.episode_length_s = 10.0
        self.sim = sim_utils.SimulationCfg(
            dt=SIM_DT,
            physics=NewtonCfg(solver_cfg=MJWarpSolverCfg(solver="newton", integrator="implicitfast", njmax=200, nconmax=60,
                                                         iterations=100, ls_iterations=50),
                              num_substeps=1, use_cuda_graph=True))
