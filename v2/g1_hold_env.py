"""G1 water-glass v2 (2026-11-18): the G1 HOLDS the glass in its right Dex3 hand (user correction: "the G1 should carry the glass in
the hand"). Scene as Isaac Lab's fixed-base G1 manipulation env (IsaacContrib-PickPlace-FixedBaseUpperBodyIK-G1-Abs: G1_29DOF_CFG
with Dex3 hands, pelvis fixed), on Newton because the water is MPM.

- Glass: a free rigid body (round tumbler, hollow watertight mesh), no weld: thumb, index and middle finger close around it and
  hold it by contact friction. Its axis is the palm z axis, so the arm's home pose has the palm z axis vertical (find_home.py).
- Contacts: the MuJoCo-Warp entry owns the robot and the glass but only the right hand's shapes and the glass's (shape label
  patterns), so the fingers grip the glass while the legs, which pass through the spill tray, collide with nothing.
- Water: MPM particles in a cylinder inside the glass; one-way coupling (the glass moves the water); the water's mass is added
  to the glass so the fingers carry the full load. Reset order: joints -> FK -> glass placed in the hand -> FK -> water ->
  MPM history (the v1 reset-teleport fix).
All water-physics fixes of v1 (g1_water_env.py) are kept: finite-difference collider velocity with one MPM substep, walls >= 1
voxel, damped drives, step-mean velocities, the first step of an episode held.
"""
from __future__ import annotations

import math
import os
import sys

import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
import numpy as np
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
from isaaclab.sim.spawners.spawner_cfg import SpawnerCfg
from isaaclab.sim.utils import clone
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
# Arm home: palm z axis = world up, grip point at P_HOME (find_home.py). Overridable for the grasp probe.
HOME = tuple(float(x) for x in os.environ.get("G1H_HOME", "1.068,-0.07,0.058,0.5,1.258,-1.44,-0.027").split(","))
GRIP = tuple(float(x) for x in os.environ.get("G1H_GRIP", "0.065,0.032,0.0").split(","))  # glass axis point at mid-height, palm frame
P_HOME = tuple(float(x) for x in os.environ.get("G1H_P_HOME", "0.30,-0.12,0.85").split(","))  # GRIP at home, env frame
RIGHT_HAND = ["right_hand_index_0_joint", "right_hand_index_1_joint", "right_hand_middle_0_joint", "right_hand_middle_1_joint",
              "right_hand_thumb_0_joint", "right_hand_thumb_1_joint", "right_hand_thumb_2_joint"]
HAND_OPEN = (0.08, 0.09, 0.08, 0.09, 0.0, 0.0, -0.09)
# Isaac Lab's empirical Dex3 gripper pose (tri-hand retargeter: index/middle 1.0/1.0, thumb 0/-0.4/-0.7), a little further.
HAND_CLOSED = (1.2, 1.2, 1.2, 1.2, 0.0, -0.5, -0.85)
HAND_GRASP = tuple(float(x) for x in os.environ.get("G1H_GRASP", ",".join(map(str, HAND_OPEN))).split(","))  # contact pose
SQUEEZE = float(os.environ.get("G1H_SQUEEZE", "0.08"))  # rad past contact, towards closing: grip force = kp x this
# Round tumbler sized for the ring the Dex3 thumb and fingers form (view_hand.py): 5.5 cm outside, 6 mm walls (4.3 cm inside),
# 8 cm deep, 12 mm bottom; water 4.5 cm (~65 ml). The MPM collider is thickened by COLLIDER_MARGIN.
R_IN, WALL, CAVITY, BOTTOM, WATER_H = (float(os.environ.get("G1H_R_IN", "0.0215")), 0.006, 0.08, 0.012, 0.045)
R_OUT, H_GLASS = R_IN + WALL, BOTTOM + CAVITY
GLASS_MASS = 0.12
VOXEL, SPACING = 0.01, 0.005  # 2 particles per voxel axis, as v1
WATER_MASS_P = 1000.0 * SPACING**3
SPILL_MARGIN, COLLIDER_MARGIN, GRIP_FRICTION = 0.01, 0.004, 1.0
GROUND_PATTERN = r".*/GroundPlane/CollisionPlane$"
WATER = MPMParticleMaterialCfg(density=1000.0, viscosity=1.0e-3, friction=0.0, damping=1.0e-3, yield_pressure=1.0e15,
                               tensile_yield_ratio=1.0)  # Isaac Lab teapot demo's water


def water_points():
    """Particle positions (glass frame, base at the origin): a 0.5 cm lattice inside the cylinder r <= R_IN - 4 mm."""
    m = 0.004
    g = np.arange(-R_IN + m + 0.5 * SPACING, R_IN - m, SPACING)
    zs = np.arange(BOTTOM + m + 0.5 * SPACING, BOTTOM + WATER_H, SPACING)
    pts = [(x, y, z) for z in zs for y in g for x in g if x * x + y * y <= (R_IN - m) ** 2]
    return pts


PARTICLES_PER_ENV = len(water_points())
WATER_MASS = PARTICLES_PER_ENV * WATER_MASS_P


def glass_mesh(n: int = 32):
    """Hollow round tumbler as one closed surface (base centre at the origin): points, face counts, indices (outward)."""
    ang = [2 * math.pi * k / n for k in range(n)]
    ring = lambda r, z: [(r * math.cos(a), r * math.sin(a), z) for a in ang]  # noqa: E731
    pts = ring(R_OUT, 0.0) + ring(R_OUT, H_GLASS) + ring(R_IN, H_GLASS) + ring(R_IN, BOTTOM) + [(0, 0, 0), (0, 0, BOTTOM)]
    co, ci = 4 * n, 4 * n + 1  # centres of the outer and inner bottom
    faces = []
    for i in range(n):
        j = (i + 1) % n
        faces += [(i, j, n + j, n + i), (n + i, n + j, 2 * n + j, 2 * n + i), (2 * n + i, 2 * n + j, 3 * n + j, 3 * n + i)]
        faces += [(co, j, i), (ci, 3 * n + i, 3 * n + j)]  # outer bottom (-z), inner bottom (+z)
    return pts, [len(f) for f in faces], [v for f in faces for v in f]


@clone
def spawn_glass(prim_path, cfg, translation=None, orientation=None, **kwargs):
    from pxr import Gf, Sdf, UsdGeom, UsdPhysics, UsdShade  # noqa: PLC0415

    stage = sim_utils.get_current_stage()
    xf = UsdGeom.Xform.Define(stage, prim_path)
    if translation is not None:
        xf.AddTranslateOp().Set(Gf.Vec3d(*translation))
    if orientation is not None:
        xf.AddOrientOp().Set(Gf.Quatf(orientation[3], Gf.Vec3f(*orientation[:3])))
    prim = stage.GetPrimAtPath(prim_path)
    UsdPhysics.RigidBodyAPI.Apply(prim)
    UsdPhysics.MassAPI.Apply(prim).CreateMassAttr(cfg.mass)
    if cfg.disable_gravity:  # probes only: PhysX's attribute (Newton's importer reads it) and MuJoCo's body gravity compensation
        prim.AddAppliedSchema("PhysxRigidBodyAPI")
        prim.CreateAttribute("physxRigidBody:disableGravity", Sdf.ValueTypeNames.Bool).Set(True)
        prim.CreateAttribute("mjc:gravcomp", Sdf.ValueTypeNames.Float).Set(1.0)
    pts, counts, idx = glass_mesh()
    mesh = UsdGeom.Mesh.Define(stage, f"{prim_path}/glass")
    mesh.CreatePointsAttr([Gf.Vec3f(*v) for v in pts])
    mesh.CreateFaceVertexCountsAttr(counts)
    mesh.CreateFaceVertexIndicesAttr(idx)
    mesh.CreateDisplayColorAttr([Gf.Vec3f(0.80, 0.90, 0.97)])
    mesh.CreateDisplayOpacityAttr([0.45])
    UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
    UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim()).CreateApproximationAttr("none")
    mesh.GetPrim().CreateAttribute("newton:contactMargin", Sdf.ValueTypeNames.Float).Set(COLLIDER_MARGIN)
    mat = UsdShade.Material.Define(stage, f"{prim_path}/grip_material")
    pm = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
    pm.CreateStaticFrictionAttr(cfg.friction)
    pm.CreateDynamicFrictionAttr(cfg.friction)
    pm.CreateRestitutionAttr(0.0)
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(mat, UsdShade.Tokens.weakerThanDescendants, "physics")
    return prim


@configclass
class GlassSpawnCfg(SpawnerCfg):
    func: object = spawn_glass
    mass: float = GLASS_MASS
    friction: float = GRIP_FRICTION
    disable_gravity: bool = os.environ.get("G1H_GLASS_NOGRAV") == "1"


def g1_cfg():
    cfg = G1_29DOF_CFG.copy()
    cfg.spawn = cfg.spawn.replace(fix_root_link=True, articulation_props=[NewtonArticulationCfg(self_collision_enabled=False)],
                                  joint_drive_props=[MujocoJointCfg(actuatorgravcomp=True)])
    cfg.prim_path = "{ENV_REGEX_NS}/Robot"
    joint_pos = dict(cfg.init_state.joint_pos)
    joint_pos.update(dict(zip(RIGHT_ARM, HOME)))
    joint_pos.update(dict(zip(RIGHT_HAND, HAND_GRASP)))
    cfg.init_state = cfg.init_state.replace(pos=(0.0, 0.0, 0.80), joint_pos=joint_pos)
    # Damped drives (v1 gain study): arms, waist, feet; hands as the asset (kp 20, kd 2) with armature 0.03.
    for name, (kp, kd, arm) in {"arms": (3000.0, 100.0, 0.03), "waist": (5000.0, 300.0, 0.03), "feet": (20.0, 2.0, 0.03),
                                "hands": (20.0, 2.0, 0.03)}.items():
        cfg.actuators[name] = cfg.actuators[name].replace(stiffness=kp, damping=kd, armature=arm)
    return cfg


def glass_home_base():
    """Glass base at home (env frame): the grip point is the glass axis at mid-height; the palm z axis is up at home."""
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
class G1HoldSceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(prim_path="/World/GroundPlane", spawn=sim_utils.GroundPlaneCfg())
    robot = g1_cfg()
    glass = RigidObjectCfg(prim_path="{ENV_REGEX_NS}/Glass", spawn=GlassSpawnCfg(mass=GLASS_MASS + WATER_MASS),
                           init_state=RigidObjectCfg.InitialStateCfg(pos=glass_home_base()))
    water = MPMObjectCfg(prim_path="{ENV_REGEX_NS}/Water", init_state=MPMObjectCfg.InitialStateCfg(pos=glass_home_base()),
                         spawn=MPMPointsCfg(positions=water_points(), mass=WATER_MASS_P, radius=0.62 * SPACING, material=WATER,
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


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)


@configclass
class G1HoldEnvCfg(ManagerBasedRLEnvCfg):
    scene: G1HoldSceneCfg = G1HoldSceneCfg(num_envs=4, env_spacing=2.5)
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
                                                                           njmax=512, nconmax=512),
                                    bodies=[r"{ENV_REGEX_NS}/Robot", r"{ENV_REGEX_NS}/Glass"], include_body_shapes=False,
                                    shape_label_patterns=[r"{ENV_REGEX_NS}/Robot/right_hand/.*", r"{ENV_REGEX_NS}/Glass/.*"],
                                    substeps=int(os.environ.get("G1H_ARM_SUB", "3"))),
                    CouplerEntryCfg(name="water", solver_cfg=MPMSolverCfg(
                        voxel_size=VOXEL, grid_type="sparse", grid_padding=0, strain_basis="P0", transfer_scheme="apic",
                        max_iterations=24, tolerance=1.0e-4, warmstart_mode="auto", velocity_basis="Q1", collider_basis="pic27",
                        collider_velocity_mode="finite_difference", solver="auto", separate_worlds=True, project_outside_colliders=False),
                        bodies=[rf"{{ENV_REGEX_NS}}/{n}" for n in _tray_parts()], shape_label_patterns=[GROUND_PATTERN], all_particles=True,
                        include_static_shapes=False, include_child_joints=False, substeps=1, in_place=True),
                ],
                proxies=[CouplerProxyMappingCfg(source="arm", destination="water", bodies=[r"{ENV_REGEX_NS}/Glass"], mode="staggered",
                                                mass_scale=100.0, proxy_relaxation=0.0, collision_pipeline=None)],
                iterations=1),
            collision_cfg=NewtonCollisionPipelineCfg(soft_contact_max=0), num_substeps=1, use_cuda_graph=True))
        self.sim.use_newton_actuators = True
        set_mpm_capacity(self)


def set_mpm_capacity(cfg, active=1536, leaf=256, lower=16, worlds_per_upper=8, min_upper=32):
    """Sparse-grid caps for all worlds (v1). Call again after changing num_envs."""
    w = max(2, int(cfg.scene.num_envs))
    mpm = cfg.sim.physics.solver_cfg.entries[1].solver_cfg
    mpm.max_upper_node_count = max(min_upper, math.ceil(w / worlds_per_upper))
    mpm.max_lower_node_count = max(lower * w, mpm.max_upper_node_count)
    mpm.max_leaf_node_count = max(leaf * w, mpm.max_lower_node_count)
    mpm.max_active_cell_count = max(max(active, PARTICLES_PER_ENV) * w, mpm.max_leaf_node_count)


class G1HoldEnv(ManagerBasedRLEnv):
    """Glass placed in the hand at every reset, MPM resets in the v1 order, spill measure in the glass frame."""

    cfg: G1HoldEnvCfg

    def __init__(self, cfg, **kwargs):
        self._palm = None
        self._hand_ids = None
        super().__init__(cfg, **kwargs)

    @property
    def robot(self):
        return self.scene["robot"]

    @property
    def glass(self):
        return self.scene["glass"]

    @property
    def water(self):
        return self.scene["water"]

    @property
    def palm(self):
        if self._palm is None:
            self._palm = self.robot.body_names.index("right_hand_palm_link")
        return self._palm

    @property
    def hand_ids(self):
        if self._hand_ids is None:
            self._hand_ids = [self.robot.joint_names.index(j) for j in RIGHT_HAND]
        return self._hand_ids

    def hand_targets(self, ids, grasp=None):
        """Finger targets SQUEEZE rad past the contact pose, towards closing."""
        g = torch.tensor(HAND_GRASP if grasp is None else grasp, device=self.device)
        closing = torch.sign(torch.tensor(HAND_CLOSED, device=self.device) - torch.tensor(HAND_OPEN, device=self.device))
        t = (g + SQUEEZE * closing).expand(len(ids), -1)
        self.robot.set_joint_position_target_index(target=t, joint_ids=self.hand_ids, env_ids=ids)

    def glass_in_hand_pose(self):
        """World pose of the glass base in the grip: palm pose x (GRIP - H/2 along the palm z axis), glass frame = palm frame."""
        pp, pq = self.robot.data.body_pos_w.torch[:, self.palm], self.robot.data.body_quat_w.torch[:, self.palm]
        off = torch.tensor([GRIP[0], GRIP[1], GRIP[2] - 0.5 * H_GLASS], device=self.device).expand(self.num_envs, -1)
        return pp + math_utils.quat_apply(pq, off), pq

    def _reset_idx(self, env_ids):
        super()._reset_idx(env_ids)
        import newton  # noqa: PLC0415
        import warp as wp  # noqa: PLC0415
        from isaaclab_newton.physics.mpm_solver import reset_mpm_solver_state  # noqa: PLC0415

        ids = torch.arange(self.num_envs, device=self.device)[env_ids] if isinstance(env_ids, slice) else torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        self.hand_targets(ids)
        self.scene.write_data_to_sim()
        self.sim.forward()  # palm pose of the reset arm
        pos, quat = self.glass_in_hand_pose()
        self.glass.write_root_pose_to_sim_index(root_pose=torch.cat([pos, quat], -1)[ids], env_ids=ids)
        self.glass.write_root_velocity_to_sim_index(root_velocity=torch.zeros(len(ids), 6, device=self.device), env_ids=ids)
        self.water.reset(ids)
        self.scene.write_data_to_sim()
        self.sim.forward()  # FK before the MPM history reset (v1: else the reset teleport blasts the water out)
        backend = self.sim.backend_manager.backend
        mask = torch.zeros(self.num_envs + 1, dtype=torch.bool, device=self.device)
        mask[ids] = True
        reset_mpm_solver_state(backend, state=backend.state_0, world_mask=wp.from_torch(mask, dtype=wp.bool),
                               flags=newton.StateFlags.BODY | newton.StateFlags.PARTICLE)

    def glass_pose_w(self):
        return self.glass.data.root_pos_w.torch, self.glass.data.root_quat_w.torch

    def glass_in_palm(self):
        """Glass base position (palm frame) and the angle (rad) between the glass axis and the palm z axis: slip measures."""
        gp, gq = self.glass_pose_w()
        pp, pq = self.robot.data.body_pos_w.torch[:, self.palm], self.robot.data.body_quat_w.torch[:, self.palm]
        rel = math_utils.quat_apply_inverse(pq, gp - pp)
        z = torch.tensor([0.0, 0.0, 1.0], device=self.device).expand(self.num_envs, -1)
        ang = torch.acos((math_utils.quat_apply(gq, z) * math_utils.quat_apply(pq, z)).sum(-1).clamp(-1, 1))
        return rel, ang

    def water_in_glass(self):
        pos, quat = self.glass_pose_w()
        pts = self.water.data.particle_pos_w.torch
        local = math_utils.quat_apply_inverse(quat[:, None].expand(-1, pts.shape[1], -1), pts - pos[:, None])
        r = local[..., :2].norm(dim=-1)
        return (r <= R_IN + SPILL_MARGIN) & (local[..., 2] >= BOTTOM - SPILL_MARGIN) & (local[..., 2] <= H_GLASS + SPILL_MARGIN)

    def spilled_fraction(self):
        return 1.0 - self.water_in_glass().float().mean(-1)

    def glass_tilt(self):
        _, quat = self.glass_pose_w()
        up = math_utils.quat_apply(quat, torch.tensor([0.0, 0.0, 1.0], device=self.device).expand(quat.shape[0], -1))
        return torch.acos(up[:, 2].clamp(-1.0, 1.0))
