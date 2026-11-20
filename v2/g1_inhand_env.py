"""G1 water-glass v2b (2026-11-18): the glass IN the right Dex3 hand, fingers wrapped around it, held by a fixed joint.

User correction: "the G1 should carry the glass in the hand". A friction-only grasp did not work with the Dex3 in Newton
(probe_grasp.py: closing pushes the glass out of the ring; probe_place.py: placements that touch the finger meshes give NaN), so the
glass is welded to the palm INSIDE the grip (no slip) with the fingers closed around it in Isaac Lab's Dex3 gripper pose
(tri-hand retargeter: index/middle 1.0/1.0, thumb 0/-0.4/-0.7). Scene as Isaac Lab's fixed-base G1 manipulation env (G1_29DOF_CFG +
Dex3, pelvis fixed). Structure and all water fixes as v1 (g1_water_env.py): the glass is a body of the robot (fixed joint to the
palm) proxied into the MPM water; finite-difference collider velocity, one MPM substep, FK before the MPM history reset, damped
drives, the first step of an episode held. Home: palm z axis up (find_home.py), so the glass is upright.
"""
from __future__ import annotations

import math
import os
import sys

import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
import torch
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
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
from isaaclab_assets.robots.unitree import G1_29DOF_CFG
from isaaclab_newton.assets import MPMObjectCfg
from isaaclab_newton.physics import MJWarpSolverCfg, MPMSolverCfg, NewtonCfg, NewtonCollisionPipelineCfg
from isaaclab_newton.physics.coupling import CouplerEntryCfg, CouplerProxyCfg, CouplerProxyMappingCfg
from isaaclab_newton.sim.schemas import MujocoJointCfg, NewtonArticulationCfg
from isaaclab_newton.sim.spawners.mpm import MPMParticleMaterialCfg, MPMPointsCfg

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from g1_glass_env import RIGHT_ARM, JointVelocityServoActionCfg  # noqa: E402

SIM_DT, DECIMATION = 1.0 / 120.0, 8  # 15 Hz policy
HOME = (1.068, -0.07, 0.058, 0.5, 1.258, -1.44, -0.027)  # palm z up, GRIP at P_HOME (find_home.py --grip 0.065 0.032 0)
GRIP = tuple(float(x) for x in os.environ.get("G1I_GRIP", "0.065,0.032").split(","))  # glass axis in the palm frame (ring centre)
P_HOME = (0.30, -0.12, 0.85)  # glass axis at mid-height at home, env frame
RIGHT_HAND = ["right_hand_index_0_joint", "right_hand_index_1_joint", "right_hand_middle_0_joint", "right_hand_middle_1_joint",
              "right_hand_thumb_0_joint", "right_hand_thumb_1_joint", "right_hand_thumb_2_joint"]
GRIPPER_POSE = (1.0, 1.0, 1.0, 1.0, 0.0, -0.4, -0.7)  # Isaac Lab tri-hand retargeter, index + middle + thumb closed
HAND_SCALE = float(os.environ.get("G1I_HAND_SCALE", "1.0"))  # fraction of the gripper pose (visual fit around the glass)
R_OUT, WALL, CAVITY, BOTTOM = float(os.environ.get("G1I_R_OUT", "0.025")), 0.006, 0.08, 0.012
WATER_H = float(os.environ.get("G1I_WATER_H", "0.045"))  # water depth; free space above it = CAVITY - WATER_H (3.5 cm default)
R_IN, H_GLASS = R_OUT - WALL, BOTTOM + CAVITY
GLASS_MASS = 0.12
VOXEL, SPACING = 0.01, 0.005
SPILL_MARGIN, COLLIDER_MARGIN = 0.01, 0.004
GROUND_PATTERN = r".*/GroundPlane/CollisionPlane$"
WATER = MPMParticleMaterialCfg(density=1000.0, viscosity=1.0e-3, friction=0.0, damping=1.0e-3, yield_pressure=1.0e15,
                               tensile_yield_ratio=1.0)  # Isaac Lab teapot demo's water


def water_points():
    """Particles (glass frame, base at the origin): 0.5 cm lattice in the cylinder r <= R_IN - 4 mm, 4 mm above the bottom."""
    m = 0.004
    g = [-R_IN + m + 0.5 * SPACING + SPACING * i for i in range(int((2 * (R_IN - m)) / SPACING))]
    zs = [BOTTOM + m + 0.5 * SPACING + SPACING * k for k in range(int((WATER_H - m) / SPACING))]
    return [(x, y, z) for z in zs for y in g for x in g if x * x + y * y <= (R_IN - m) ** 2]


PARTICLES_PER_ENV = len(water_points())


def glass_mesh(n: int = 32):
    """Hollow round tumbler as one closed surface (base centre at the origin): points, face counts, indices (outward)."""
    ang = [2 * math.pi * k / n for k in range(n)]
    ring = lambda r, z: [(r * math.cos(a), r * math.sin(a), z) for a in ang]  # noqa: E731
    pts = ring(R_OUT, 0.0) + ring(R_OUT, H_GLASS) + ring(R_IN, H_GLASS) + ring(R_IN, BOTTOM) + [(0, 0, 0), (0, 0, BOTTOM)]
    co, ci = 4 * n, 4 * n + 1
    faces = []
    for i in range(n):
        j = (i + 1) % n
        faces += [(i, j, n + j, n + i), (n + i, n + j, 2 * n + j, 2 * n + i), (2 * n + i, 2 * n + j, 3 * n + j, 3 * n + i)]
        faces += [(co, j, i), (ci, 3 * n + i, 3 * n + j)]
    return pts, [len(f) for f in faces], [v for f in faces for v in f]


def weld_local_pos():
    """Glass base in the palm frame: the glass axis through GRIP, its mid-height at the palm plane (between index and middle)."""
    return (GRIP[0], GRIP[1], -0.5 * H_GLASS)


def _spawn_glass_in_hand(glass_path: str, palm_path: str, stage) -> None:
    from pxr import Gf, Sdf, UsdGeom, UsdPhysics  # noqa: PLC0415

    pos = weld_local_pos()
    xf = UsdGeom.Xform.Define(stage, glass_path)
    xf.AddTranslateOp().Set(Gf.Vec3d(*pos))
    prim = stage.GetPrimAtPath(glass_path)
    UsdPhysics.RigidBodyAPI.Apply(prim)
    UsdPhysics.MassAPI.Apply(prim).CreateMassAttr(GLASS_MASS)
    pts, counts, idx = glass_mesh()
    mesh = UsdGeom.Mesh.Define(stage, f"{glass_path}/glass")
    mesh.CreatePointsAttr([Gf.Vec3f(*v) for v in pts])
    mesh.CreateFaceVertexCountsAttr(counts)
    mesh.CreateFaceVertexIndicesAttr(idx)
    mesh.CreateDisplayColorAttr([Gf.Vec3f(0.80, 0.90, 0.97)])
    UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
    UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim()).CreateApproximationAttr("none")
    mesh.GetPrim().CreateAttribute("newton:contactMargin", Sdf.ValueTypeNames.Float).Set(COLLIDER_MARGIN)
    joint = UsdPhysics.FixedJoint.Define(stage, f"{glass_path}/FixedJoint")
    joint.CreateBody0Rel().SetTargets([Sdf.Path(palm_path)])
    joint.CreateBody1Rel().SetTargets([Sdf.Path(glass_path)])
    joint.CreateLocalPos0Attr().Set(Gf.Vec3f(*pos))
    joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0))
    joint.CreateLocalRot0Attr().Set(Gf.Quatf(1.0))
    joint.CreateLocalRot1Attr().Set(Gf.Quatf(1.0))


def _spawn_g1_with_glass(prim_path, cfg, translation=None, orientation=None, **kwargs):
    from pxr import Usd  # noqa: PLC0415

    robot_prim = sim_utils.spawn_from_usd(prim_path, cfg, translation, orientation, **kwargs)
    stage = robot_prim.GetStage()
    for root in sim_utils.find_matching_prims(prim_path, stage=stage):
        palm = next((p for p in Usd.PrimRange(root) if p.GetName() == "right_hand_palm_link"), None)
        if palm is None:
            raise ValueError(f"no right_hand_palm_link under {root.GetPath()}")
        _spawn_glass_in_hand(f"{root.GetPath()}/Cup", str(palm.GetPath()), stage)
    return robot_prim


@configclass
class G1WithGlassCfg(sim_utils.UsdFileCfg):
    func = _spawn_g1_with_glass


def g1_cfg():
    cfg = G1_29DOF_CFG.copy()
    spawn = G1WithGlassCfg(**{k: getattr(cfg.spawn, k) for k in ("usd_path", "activate_contact_sensors", "rigid_props")})
    spawn.fix_root_link = True
    spawn.articulation_props = [NewtonArticulationCfg(self_collision_enabled=False)]
    spawn.joint_drive_props = [MujocoJointCfg(actuatorgravcomp=True)]
    cfg.spawn = spawn
    cfg.prim_path = "{ENV_REGEX_NS}/Robot"
    joint_pos = dict(cfg.init_state.joint_pos)
    joint_pos.update(dict(zip(RIGHT_ARM, HOME)))
    joint_pos.update(dict(zip(RIGHT_HAND, (HAND_SCALE * v for v in GRIPPER_POSE))))
    cfg.init_state = cfg.init_state.replace(pos=(0.0, 0.0, 0.80), joint_pos=joint_pos)
    for name, (kp, kd, arm) in {"arms": (3000.0, 100.0, 0.03), "waist": (5000.0, 300.0, 0.03), "feet": (20.0, 2.0, 0.03),
                                "hands": (20.0, 2.0, 0.03)}.items():
        cfg.actuators[name] = cfg.actuators[name].replace(stiffness=kp, damping=kd, armature=arm)
    return cfg


def glass_home_base():
    return (P_HOME[0], P_HOME[1], P_HOME[2] - 0.5 * H_GLASS)


TRAY_CENTER, TRAY_SIZE, TRAY_WALL_H = (0.30, -0.10, 0.40), (0.90, 1.00), 0.08  # catch tray for spilled water (MPM only)


def _tray_parts():
    (cx, cy, cz), (sx, sy), h, t = TRAY_CENTER, TRAY_SIZE, TRAY_WALL_H, 0.02
    return {"TrayFloor": ((sx, sy, t), (cx, cy, cz)),
            "TrayFront": ((t, sy, h), (cx + 0.5 * sx, cy, cz + 0.5 * h)), "TrayBack": ((t, sy, h), (cx - 0.5 * sx, cy, cz + 0.5 * h)),
            "TrayLeft": ((sx, t, h), (cx, cy + 0.5 * sy, cz + 0.5 * h)), "TrayRight": ((sx, t, h), (cx, cy - 0.5 * sy, cz + 0.5 * h))}


def _tray_box(name):
    from isaaclab.sim.schemas import UsdPhysicsCollisionCfg, UsdPhysicsRigidBodyCfg
    size, pos = _tray_parts()[name]
    return RigidObjectCfg(prim_path=f"{{ENV_REGEX_NS}}/{name}", init_state=RigidObjectCfg.InitialStateCfg(pos=pos),
                          spawn=sim_utils.CuboidCfg(size=size, rigid_props=UsdPhysicsRigidBodyCfg(rigid_body_enabled=True, kinematic_enabled=True),
                                                    collision_props=[UsdPhysicsCollisionCfg(collision_enabled=True)],
                                                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.55, 0.55, 0.5))))


@configclass
class G1InHandSceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(prim_path="/World/GroundPlane", spawn=sim_utils.GroundPlaneCfg())
    robot = g1_cfg()
    water = MPMObjectCfg(prim_path="{ENV_REGEX_NS}/Water", init_state=MPMObjectCfg.InitialStateCfg(pos=glass_home_base()),
                         spawn=MPMPointsCfg(positions=water_points(), mass=1000.0 * SPACING**3, radius=0.62 * SPACING, material=WATER,
                                            visual_color=(0.2, 0.45, 0.95)))
    tray_floor = _tray_box("TrayFloor")
    tray_front = _tray_box("TrayFront")
    tray_back = _tray_box("TrayBack")
    tray_left = _tray_box("TrayLeft")
    tray_right = _tray_box("TrayRight")
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
    spilled = DoneTerm(func=spilled_too_much, params={"limit": 0.3})


@configclass
class G1InHandEnvCfg(ManagerBasedRLEnvCfg):
    scene: G1InHandSceneCfg = G1InHandSceneCfg(num_envs=4, env_spacing=2.5)
    actions: ActionsCfg = ActionsCfg()
    observations: ObservationsCfg = ObservationsCfg()
    events: EventsCfg = EventsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    def __post_init__(self):
        self.decimation = DECIMATION
        self.episode_length_s = 12.0
        self.sim = sim_utils.SimulationCfg(dt=SIM_DT, physics=NewtonCfg(
            solver_cfg=CouplerProxyCfg(
                entries=[
                    CouplerEntryCfg(name="arm", solver_cfg=MJWarpSolverCfg(use_mujoco_contacts=False, integrator="implicitfast",
                                                                           njmax=256, nconmax=512),
                                    bodies=[r"{ENV_REGEX_NS}/Robot"], include_body_shapes=False, substeps=3),
                    CouplerEntryCfg(name="water", solver_cfg=MPMSolverCfg(
                        voxel_size=VOXEL, grid_type="sparse", grid_padding=0, strain_basis="P0", transfer_scheme="apic",
                        max_iterations=24, tolerance=1.0e-4, warmstart_mode="auto", velocity_basis="Q1", collider_basis="pic27",
                        collider_velocity_mode="finite_difference", solver="auto", separate_worlds=True, project_outside_colliders=False),
                        bodies=[rf"{{ENV_REGEX_NS}}/{n}" for n in _tray_parts()], shape_label_patterns=[GROUND_PATTERN], all_particles=True,
                        include_static_shapes=False, include_child_joints=False, substeps=1, in_place=True),
                ],
                proxies=[CouplerProxyMappingCfg(source="arm", destination="water", bodies=[r"{ENV_REGEX_NS}/Robot/Cup"], mode="staggered",
                                                mass_scale=100.0, proxy_relaxation=0.0, collision_pipeline=None)],
                iterations=1),
            collision_cfg=NewtonCollisionPipelineCfg(soft_contact_max=0), num_substeps=1, use_cuda_graph=True))
        self.sim.use_newton_actuators = True
        set_mpm_capacity(self)


def set_mpm_capacity(cfg, active=1536, leaf=256, lower=16, worlds_per_upper=8, min_upper=32):
    w = max(2, int(cfg.scene.num_envs))
    mpm = cfg.sim.physics.solver_cfg.entries[1].solver_cfg
    mpm.max_upper_node_count = max(min_upper, math.ceil(w / worlds_per_upper))
    mpm.max_lower_node_count = max(lower * w, mpm.max_upper_node_count)
    mpm.max_leaf_node_count = max(leaf * w, mpm.max_lower_node_count)
    mpm.max_active_cell_count = max(max(active, PARTICLES_PER_ENV) * w, mpm.max_leaf_node_count)


class G1InHandEnv(ManagerBasedRLEnv):
    """v1's reset contract (FK before the MPM history reset) and spill measure, for the round glass in the hand."""

    cfg: G1InHandEnvCfg

    def __init__(self, cfg, **kwargs):
        self._cup_body = None
        self._hand_ids = None
        super().__init__(cfg, **kwargs)

    @property
    def robot(self):
        return self.scene["robot"]

    @property
    def water(self):
        return self.scene["water"]

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

    def _reset_idx(self, env_ids):
        super()._reset_idx(env_ids)
        import newton  # noqa: PLC0415
        import warp as wp  # noqa: PLC0415
        from isaaclab_newton.physics.mpm_solver import reset_mpm_solver_state  # noqa: PLC0415

        ids = torch.arange(self.num_envs, device=self.device)[env_ids] if isinstance(env_ids, slice) else torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        grip = torch.tensor([HAND_SCALE * v for v in GRIPPER_POSE], device=self.device).expand(len(ids), -1)
        self.robot.set_joint_position_target_index(target=grip, joint_ids=self.hand_ids, env_ids=ids)  # fingers stay closed
        self.water.reset(ids)
        self.scene.write_data_to_sim()
        self.sim.forward()  # FK before the MPM history reset (v1: else the reset teleport blasts the water out)
        backend = self.sim.backend_manager.backend
        mask = torch.zeros(self.num_envs + 1, dtype=torch.bool, device=self.device)
        mask[ids] = True
        reset_mpm_solver_state(backend, state=backend.state_0, world_mask=wp.from_torch(mask, dtype=wp.bool),
                               flags=newton.StateFlags.BODY | newton.StateFlags.PARTICLE)

    def cup_pose_w(self):
        return self.robot.data.body_pos_w.torch[:, self.cup_body], self.robot.data.body_quat_w.torch[:, self.cup_body]

    def water_in_cup(self):
        pos, quat = self.cup_pose_w()
        pts = self.water.data.particle_pos_w.torch
        local = math_utils.quat_apply_inverse(quat[:, None].expand(-1, pts.shape[1], -1), pts - pos[:, None])
        r = local[..., :2].norm(dim=-1)
        return (r <= R_IN + SPILL_MARGIN) & (local[..., 2] >= BOTTOM - SPILL_MARGIN) & (local[..., 2] <= H_GLASS + SPILL_MARGIN)

    def spilled_fraction(self):
        return 1.0 - self.water_in_cup().float().mean(-1)

    def cup_tilt(self):
        _, quat = self.cup_pose_w()
        up = math_utils.quat_apply(quat, torch.tensor([0.0, 0.0, 1.0], device=self.device).expand(quat.shape[0], -1))
        return torch.acos(up[:, 2].clamp(-1.0, 1.0))
