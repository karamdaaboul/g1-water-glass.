"""G1 water-glass task, stage 2 (2026-11-18): fixed-base G1 with a glass welded to the right palm and MPM water in it.

Isaac Lab main + Newton, following the patterns of Isaac Lab's UR10 particle push (a tool welded to a link and proxied into MPM)
and Franka Pour (per-env particle reset and spill test):
  rigid entry  "arm"    MuJoCo-Warp: the G1 (fixed base, gravity compensation, no self-collision) and its glass
  MPM entry    "water"  implicit MPM, water material of Isaac Lab's teapot demo; collides with the ground plane
  proxy        the glass body of the rigid entry is a collider in the MPM entry (two-way through the proxy, mass_scale)
The glass is one rigid body of five boxes (bottom + four walls), welded to right_hand_palm_link with a rotation that makes it
upright at the home arm pose. Spill = share of particles outside the glass cavity (glass frame, + margin).
"""
from __future__ import annotations

import math
import os

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
from isaaclab_assets.robots.unitree import G1_29DOF_CFG
from isaaclab_newton.assets import MPMObjectCfg
from isaaclab_newton.physics import MJWarpSolverCfg, MPMSolverCfg, NewtonCfg, NewtonCollisionPipelineCfg
from isaaclab_newton.physics.coupling import CouplerEntryCfg, CouplerProxyCfg, CouplerProxyMappingCfg
from isaaclab_newton.sim.schemas import MujocoJointCfg, NewtonArticulationCfg
from isaaclab_newton.sim.spawners.mpm import MPMGridCfg, MPMParticleMaterialCfg

from g1_glass_env import RIGHT_ARM, JointVelocityServoActionCfg

SIM_DT = 1.0 / 120.0
DECIMATION = 8  # 15 Hz policy
# Home pose of the right arm (probe g1_pose_probe.py, candidate A: forearm forward) and the palm pose measured there (env frame).
HOME = (-0.3, -0.2, 0.0, 1.2, 0.0, 0.0, 0.0)
PALM_HOME_POS = (0.21845, 0.20328, 0.74232)  # measured with gravity compensation (g1_home_measure.py)
PALM_HOME_QUAT_XYZW = (-0.37555, 0.23935, 0.60904, 0.65631)
CUP_OFFSET_WORLD = (0.07, 0.0, -0.06)  # glass base relative to the palm at home, world axes: 7 cm in front, 6 cm below
# Glass: inner 6 x 6 cm, cavity 9 cm, walls 6 mm, bottom 8 mm; water 5 cm deep.
INNER_W, CAVITY, WALL, BOTTOM, WATER_H = 0.06, 0.09, 0.012, 0.015, 0.05  # walls/bottom >= 1 voxel: thinner ones leaked water
CUP_MASS = 0.15
VOXEL, PPC = 0.01, 2.0
SPILL_MARGIN = 0.01
COLLIDER_MARGIN = 0.004
GROUND_PATTERN = r".*/GroundPlane/CollisionPlane$"
WATER = MPMParticleMaterialCfg(density=1000.0, viscosity=1.0e-3, friction=0.0, damping=1.0e-3, yield_pressure=1.0e15,
                               tensile_yield_ratio=1.0)  # Isaac Lab teapot demo's water


def _weld_local_pose():
    """Palm-frame pose (pos, quat xyzw) of the glass base that puts it upright at CUP_OFFSET_WORLD from the palm at home."""
    q = torch.tensor(PALM_HOME_QUAT_XYZW)
    q_inv = torch.cat([-q[:3], q[3:]])
    pos = math_utils.quat_apply(q_inv[None], torch.tensor(CUP_OFFSET_WORLD)[None])[0]
    return tuple(pos.tolist()), tuple(q_inv.tolist())


def cup_boxes():
    """(offset, half extents) of the bottom and four walls in the glass frame (origin at the outer base centre)."""
    ox = 0.5 * INNER_W + WALL
    rim = BOTTOM + CAVITY
    hw, hh = 0.5 * WALL, 0.5 * rim
    return (((0.0, 0.0, 0.5 * BOTTOM), (ox, ox, 0.5 * BOTTOM)),
            ((0.5 * INNER_W + hw, 0.0, hh), (hw, ox, hh)), ((-0.5 * INNER_W - hw, 0.0, hh), (hw, ox, hh)),
            ((0.0, 0.5 * INNER_W + hw, hh), (0.5 * INNER_W, hw, hh)), ((0.0, -0.5 * INNER_W - hw, hh), (0.5 * INNER_W, hw, hh)))


def cup_mesh():
    """Closed solid of the glass (origin at the outer base centre): points, face vertex counts, indices (14 quads, outward)."""
    ox, a, rim = 0.5 * INNER_W + WALL, 0.5 * INNER_W, BOTTOM + CAVITY
    sq = ((-1, -1), (1, -1), (1, 1), (-1, 1))
    O = [(x * ox, y * ox, 0.0) for x, y in sq]; T = [(x * ox, y * ox, rim) for x, y in sq]
    I = [(x * a, y * a, rim) for x, y in sq]; B = [(x * a, y * a, BOTTOM) for x, y in sq]
    pts = O + T + I + B  # indices: O 0-3, T 4-7, I 8-11, B 12-15
    faces = [(0, 3, 2, 1), (12, 13, 14, 15)]  # outer bottom (-z), inner bottom (+z)
    for i in range(4):
        j = (i + 1) % 4
        faces += [(i, j, 4 + j, 4 + i), (4 + i, 4 + j, 8 + j, 8 + i), (8 + i, 8 + j, 12 + j, 12 + i)]  # outer side, rim, inner side
    return pts, [4] * len(faces), [v for f in faces for v in f]


def _spawn_cup(cup_path: str, palm_path: str, stage) -> None:
    """One rigid body of five box colliders at cup_path, welded to the palm (as Isaac Lab's UR10 paddle)."""
    from pxr import Gf, Sdf, UsdGeom, UsdPhysics  # noqa: PLC0415  (after the app starts, as the UR10 example)

    pos, quat = _weld_local_pose()
    xf = UsdGeom.Xform.Define(stage, cup_path)
    xf.AddTranslateOp().Set(Gf.Vec3d(*pos))
    xf.AddOrientOp().Set(Gf.Quatf(quat[3], Gf.Vec3f(*quat[:3])))
    cup = stage.GetPrimAtPath(cup_path)
    UsdPhysics.RigidBodyAPI.Apply(cup)
    UsdPhysics.MassAPI.Apply(cup).CreateMassAttr(CUP_MASS)
    # One watertight solid (outer box minus the open cavity) as an exact triangle mesh: overlapping boxes merged into one MPM
    # collider leaked water at the corners (Isaac Lab's teapot demo also uses an exact open mesh for water).
    pts, counts, idx = cup_mesh()
    mesh = UsdGeom.Mesh.Define(stage, f"{cup_path}/glass")
    mesh.CreatePointsAttr([Gf.Vec3f(*v) for v in pts])
    mesh.CreateFaceVertexCountsAttr(counts)
    mesh.CreateFaceVertexIndicesAttr(idx)
    mesh.CreateDisplayColorAttr([Gf.Vec3f(0.75, 0.85, 0.95)])
    UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
    UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim()).CreateApproximationAttr("none")
    mesh.GetPrim().CreateAttribute("newton:contactMargin", Sdf.ValueTypeNames.Float).Set(COLLIDER_MARGIN)
    joint = UsdPhysics.FixedJoint.Define(stage, f"{cup_path}/FixedJoint")
    joint.CreateBody0Rel().SetTargets([Sdf.Path(palm_path)])
    joint.CreateBody1Rel().SetTargets([Sdf.Path(cup_path)])
    joint.CreateLocalPos0Attr().Set(Gf.Vec3f(*pos))
    joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0))
    joint.CreateLocalRot0Attr().Set(Gf.Quatf(quat[3], Gf.Vec3f(*quat[:3])))
    joint.CreateLocalRot1Attr().Set(Gf.Quatf(1.0))


def _spawn_g1_with_cup(prim_path, cfg, translation=None, orientation=None, **kwargs):
    """Spawn the G1 USD and author the welded glass inside each robot prototype."""
    from pxr import Usd  # noqa: PLC0415

    robot_prim = sim_utils.spawn_from_usd(prim_path, cfg, translation, orientation, **kwargs)
    stage = robot_prim.GetStage()
    for root in sim_utils.find_matching_prims(prim_path, stage=stage):
        palm = next((p for p in Usd.PrimRange(root) if p.GetName() == "right_hand_palm_link"), None)
        if palm is None:
            raise ValueError(f"no right_hand_palm_link under {root.GetPath()}")
        _spawn_cup(f"{root.GetPath()}/Cup", str(palm.GetPath()), stage)
    return robot_prim


@configclass
class G1WithCupCfg(sim_utils.UsdFileCfg):
    func = _spawn_g1_with_cup


def g1_with_cup_cfg():
    cfg = G1_29DOF_CFG.copy()
    spawn = G1WithCupCfg(**{k: getattr(cfg.spawn, k) for k in ("usd_path", "activate_contact_sensors", "rigid_props")})
    spawn.fix_root_link = True
    spawn.articulation_props = [NewtonArticulationCfg(self_collision_enabled=False)]
    spawn.joint_drive_props = [MujocoJointCfg(actuatorgravcomp=True)]
    cfg.spawn = spawn
    cfg.prim_path = "{ENV_REGEX_NS}/Robot"
    joint_pos = dict(cfg.init_state.joint_pos)
    joint_pos.update(dict(zip(RIGHT_ARM, HOME)))
    cfg.init_state = cfg.init_state.replace(pos=(0.0, 0.0, 0.80), joint_pos=joint_pos)
    # Drive gains. G1_29DOF_CFG has arms kp 3000 / kd 10 / armature 0.001 and waist kp 5000 / kd 5 (damping ratio ~0.03): the torso
    # rang under the arm's reaction torques (waist up to 0.6 rad/s) and shook the glass, 3.9 % spilled on a slow move that spills
    # nothing with damped drives (gain study 2026-11-18, debug_gains.py). Armature 0.03 = the asset's leg value (geared motors).
    # Overridable: G1W_ARM / G1W_WAIST / G1W_FEET = 'kp,kd,armature'.
    gains = {"arms": ("G1W_ARM", (3000.0, 100.0, 0.03)), "waist": ("G1W_WAIST", (5000.0, 300.0, 0.03)),
             "feet": ("G1W_FEET", (20.0, 2.0, 0.03))}
    for name, (var, default) in gains.items():
        kp, kd, arm = (float(x) for x in os.environ[var].split(",")) if os.environ.get(var) else default
        cfg.actuators[name] = cfg.actuators[name].replace(stiffness=kp, damping=kd, armature=arm)
    return cfg


def _cup_home_base_pos():
    return tuple(p + o for p, o in zip(PALM_HOME_POS, CUP_OFFSET_WORLD))


def water_cfg():
    """Water box in the cavity at the home glass pose (env frame); MPM init pos = glass base at home, upright."""
    margin = 0.004
    half = 0.5 * INNER_W - margin
    return MPMObjectCfg(
        prim_path="{ENV_REGEX_NS}/Water",
        init_state=MPMObjectCfg.InitialStateCfg(pos=_cup_home_base_pos()),
        spawn=MPMGridCfg(lower=(-half, -half, BOTTOM + margin), upper=(half, half, BOTTOM + WATER_H), voxel_size=VOXEL,
                         particles_per_cell=PPC, particle_placement="cell_center", jitter=0.0, material=WATER,
                         visual_color=(0.2, 0.45, 0.95)))


PARTICLES_PER_ENV = 1210  # measured: the MPM grid spawner with these settings emits 1,210 particles


def particles_per_env():
    margin = 0.004
    cells = [math.ceil((INNER_W - 2 * margin) / VOXEL), math.ceil((INNER_W - 2 * margin) / VOXEL), math.ceil((WATER_H - margin) / VOXEL)]
    return int(math.prod(cells) * PPC ** 3)


TRAY_CENTER, TRAY_SIZE, TRAY_WALL_H = (0.30, 0.15, 0.40), (0.90, 1.00), 0.08  # catch tray for spilled water (MPM only)


def _tray_box(name, size, pos):
    """Kinematic box seen only by the MPM entry (as the UR10 example's bin): keeps spilled water local."""
    from isaaclab.assets import RigidObjectCfg
    from isaaclab.sim.schemas import UsdPhysicsCollisionCfg, UsdPhysicsRigidBodyCfg
    return RigidObjectCfg(prim_path=f"{{ENV_REGEX_NS}}/{name}", init_state=RigidObjectCfg.InitialStateCfg(pos=pos),
                          spawn=sim_utils.CuboidCfg(size=size, rigid_props=UsdPhysicsRigidBodyCfg(rigid_body_enabled=True, kinematic_enabled=True),
                                                    collision_props=[UsdPhysicsCollisionCfg(collision_enabled=True)],
                                                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.55, 0.55, 0.5))))


def _tray_parts():
    (cx, cy, cz), (sx, sy), h, t = TRAY_CENTER, TRAY_SIZE, TRAY_WALL_H, 0.02
    return {"TrayFloor": ((sx, sy, t), (cx, cy, cz)),
            "TrayFront": ((t, sy, h), (cx + 0.5 * sx, cy, cz + 0.5 * h)), "TrayBack": ((t, sy, h), (cx - 0.5 * sx, cy, cz + 0.5 * h)),
            "TrayLeft": ((sx, t, h), (cx, cy + 0.5 * sy, cz + 0.5 * h)), "TrayRight": ((sx, t, h), (cx, cy - 0.5 * sy, cz + 0.5 * h))}


@configclass
class G1WaterSceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(prim_path="/World/GroundPlane", spawn=sim_utils.GroundPlaneCfg())
    robot = g1_with_cup_cfg()
    water = water_cfg()
    tray_floor = _tray_box("TrayFloor", *_tray_parts()["TrayFloor"])
    tray_front = _tray_box("TrayFront", *_tray_parts()["TrayFront"])
    tray_back = _tray_box("TrayBack", *_tray_parts()["TrayBack"])
    tray_left = _tray_box("TrayLeft", *_tray_parts()["TrayLeft"])
    tray_right = _tray_box("TrayRight", *_tray_parts()["TrayRight"])
    light = AssetBaseCfg(prim_path="/World/light", spawn=sim_utils.DistantLightCfg(intensity=2500.0))


@configclass
class ActionsCfg:
    right_arm = JointVelocityServoActionCfg(asset_name="robot", joint_names=RIGHT_ARM, max_vel=1.0, hold_first_step=True)


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
class EventsCfg:
    reset_robot = EventTerm(func=mdp.reset_joints_by_offset, mode="reset",
                            params={"asset_cfg": SceneEntityCfg("robot"), "position_range": (0.0, 0.0), "velocity_range": (0.0, 0.0)})


@configclass
class RewardsCfg:
    alive = RewTerm(func=mdp.is_alive, weight=0.0)


def spilled_too_much(env, limit: float = 0.3):
    """Terminate once more than ``limit`` of the water is outside the glass (also bounds the spread of spilled water)."""
    return env.spilled_fraction() > limit


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    spilled = DoneTerm(func=spilled_too_much, params={"limit": 0.3})


@configclass
class G1WaterEnvCfg(ManagerBasedRLEnvCfg):
    scene: G1WaterSceneCfg = G1WaterSceneCfg(num_envs=4, env_spacing=2.5)
    actions: ActionsCfg = ActionsCfg()
    observations: ObservationsCfg = ObservationsCfg()
    events: EventsCfg = EventsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    proxy_mass_scale: float = 100.0  # Franka Pour carries its 50 g cup with 1000

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
                        collider_velocity_mode=os.environ.get("G1W_COLLIDER_VEL", "finite_difference"), solver="auto", separate_worlds=True, project_outside_colliders=False),
                        bodies=[rf"{{ENV_REGEX_NS}}/{n}" for n in _tray_parts()], shape_label_patterns=[GROUND_PATTERN], all_particles=True, include_static_shapes=False,
                        include_child_joints=False, substeps=int(os.environ.get("G1W_MPM_SUB", "1")), in_place=True),
                ],
                proxies=[CouplerProxyMappingCfg(source="arm", destination="water", bodies=[r"{ENV_REGEX_NS}/Robot/Cup"],
                                                mode="staggered", mass_scale=self.proxy_mass_scale, proxy_relaxation=0.0, collision_pipeline=None)],
                iterations=1),
            collision_cfg=NewtonCollisionPipelineCfg(soft_contact_max=0), num_substeps=1, use_cuda_graph=True))
        self.sim.use_newton_actuators = True
        mpm = self.sim.physics.solver_cfg.entries[1].solver_cfg
        set_mpm_capacity(self)


def set_mpm_capacity(cfg, active=1536, leaf=256, lower=16, worlds_per_upper=8, min_upper=32):
    """Sparse-grid caps for all worlds, as Isaac Lab's UR10 particle push sizes them (per world: active cells >= particles;
    a leaf node covers 8^3 voxels, so 128 leaves allow spilled water to spread ~1 m^2; upper nodes are shared by nearby worlds).
    upper <= lower <= leaf <= active. Call again after changing num_envs."""
    w = max(2, int(cfg.scene.num_envs))
    mpm = cfg.sim.physics.solver_cfg.entries[1].solver_cfg
    mpm.max_upper_node_count = max(min_upper, math.ceil(w / worlds_per_upper))
    mpm.max_lower_node_count = max(lower * w, mpm.max_upper_node_count)
    mpm.max_leaf_node_count = max(leaf * w, mpm.max_lower_node_count)
    mpm.max_active_cell_count = max(max(active, PARTICLES_PER_ENV) * w, mpm.max_leaf_node_count)


class G1WaterEnv(ManagerBasedRLEnv):
    """Adds the MPM solver reset after particle writes (Franka Pour's contract) and the spill measure."""

    cfg: G1WaterEnvCfg

    def __init__(self, cfg, **kwargs):
        # Observation terms run once inside the base constructor (to size the groups), so everything they read is set before it.
        d = cfg.sim.device
        lo = torch.tensor([-0.5 * INNER_W, -0.5 * INNER_W, BOTTOM], device=d)
        hi = torch.tensor([0.5 * INNER_W, 0.5 * INNER_W, BOTTOM + CAVITY], device=d)
        self.cavity_lo, self.cavity_hi = lo - SPILL_MARGIN, hi + SPILL_MARGIN
        self._cup_body = None
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

    def _reset_idx(self, env_ids):
        super()._reset_idx(env_ids)
        import newton  # noqa: PLC0415
        import warp as wp  # noqa: PLC0415
        from isaaclab_newton.physics.mpm_solver import reset_mpm_solver_state  # noqa: PLC0415

        if isinstance(env_ids, slice):
            ids = torch.arange(self.num_envs, device=self.device)[env_ids]
        else:
            ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        self.water.reset(ids)
        # Forward kinematics BEFORE the MPM history reset: the reset copies the glass pose into the water solver's state and
        # caches it as the previous collider pose. Isaac Lab's step-time resets run FK only at the next physics step (and the
        # backend's forward() resets the solver before its FK), so without this the cache keeps the pre-teleport glass pose and
        # the next step's finite-difference collider velocity (jump / dt) blasts the water out of the glass.
        self.scene.write_data_to_sim()
        self.sim.forward()
        backend = self.sim.backend_manager.backend
        world_mask = torch.zeros(self.num_envs + 1, dtype=torch.bool, device=self.device)
        world_mask[ids] = True
        reset_mpm_solver_state(backend, state=backend.state_0, world_mask=wp.from_torch(world_mask, dtype=wp.bool),
                               flags=newton.StateFlags.BODY | newton.StateFlags.PARTICLE)

    def cup_pose_w(self):
        return self.robot.data.body_pos_w.torch[:, self.cup_body], self.robot.data.body_quat_w.torch[:, self.cup_body]

    def water_in_cup(self):
        """(num_envs, particles) bool: particle inside the glass cavity (+ margin), in the glass frame."""
        pos, quat = self.cup_pose_w()
        pts = self.water.data.particle_pos_w.torch
        local = math_utils.quat_apply_inverse(quat[:, None].expand(-1, pts.shape[1], -1), pts - pos[:, None])
        return ((local >= self.cavity_lo) & (local <= self.cavity_hi)).all(-1)

    def spilled_fraction(self):
        return 1.0 - self.water_in_cup().float().mean(-1)

    def cup_tilt(self):
        """Angle (rad) between the glass axis and world up."""
        _, quat = self.cup_pose_w()
        up = math_utils.quat_apply(quat, torch.tensor([0.0, 0.0, 1.0], device=self.device).expand(quat.shape[0], -1))
        return torch.acos(up[:, 2].clamp(-1.0, 1.0))
