"""G1 pick-and-place of a full NORMAL drinking glass (v4, 2026-11-19; the wine-glass version is g1_pp_env_wine.py: the user
wanted a normal glass). Straight-sided glass, 6 cm across, 12 cm tall, held around its body by the Dex3 (as Unitree's
Isaac-PickPlace-Cylinder-G129-Dex3 task holds a cylinder).

Scene as Isaac Lab's fixed-base G1 manipulation env (G1_29DOF_CFG + Dex3, pelvis fixed) plus a table (top at TABLE_TOP) and a free
wine glass standing on it (foot disc + stem + 8 cm bowl), full of Warp PBF water (v3/pbf_water.py, validated against hydrostatics).
- Grasp: a palm-glass weld (USD FixedJoint with MjcEqualityWeldAPI -> MuJoCo weld equality), switched off after start-up; grasp()
  sets its relative pose to the current palm-glass pose (no jump) and switches it on; release() switches it off. A friction grasp
  with the Dex3 failed in Newton (v2/probe_grasp.py, probe_place.py), so the fingers are visual (robot collisions off) and the weld
  holds the glass while the hand is closed around the stem.
- Contacts: only the glass foot (cylinder) collides, with the table and the ground (MuJoCo-Warp contacts).
- Water: one-way, driven by the glass pose each policy step (inside scene.update, before terminations); spill = water outside
  the bowl. Resets: robot -> FK -> glass on the table -> water to its pre-settled state.
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
from isaaclab.sim.schemas import UsdPhysicsCollisionCfg
from isaaclab.sim.spawners.spawner_cfg import SpawnerCfg
from isaaclab.sim.utils import clone
from isaaclab.utils import configclass
from isaaclab.utils import math as math_utils
from isaaclab_assets.robots.unitree import G1_29DOF_CFG
from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg
from isaaclab_newton.sim.schemas import MujocoJointCfg, NewtonArticulationCfg

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "v3"))
from g1_glass_env import RIGHT_ARM, JointVelocityServoActionCfg  # noqa: E402
from pbf_water import PBFWater  # noqa: E402

SIM_DT, DECIMATION = 1.0 / 120.0, 8  # 15 Hz policy
READY = (1.068, -0.07, 0.058, 0.5, 1.258, -1.44, -0.027)  # palm z up, grip point at (0.30, -0.12, 0.85) (v2/find_home.py)
GRIP = (0.065, 0.032)  # stem axis in the palm frame (centre of the Dex3 ring)
RIGHT_HAND = ["right_hand_index_0_joint", "right_hand_index_1_joint", "right_hand_middle_0_joint", "right_hand_middle_1_joint",
              "right_hand_thumb_0_joint", "right_hand_thumb_1_joint", "right_hand_thumb_2_joint"]
HAND_OPEN = (0.08, 0.09, 0.08, 0.09, 0.0, 0.0, -0.09)
HAND_CLOSED = tuple(float(os.environ.get("G1P_HAND_SCALE", "0.85")) * v for v in (1.0, 1.0, 1.0, 1.0, 0.0, -0.4, -0.7))  # gripper pose, around the glass
# Wine glass (glass frame: origin at the bowl's outer base centre, z up; stem below, foot at the bottom)
R_IN, WALL, CAVITY, BOTTOM = 0.026, 0.004, 0.114, 0.006  # normal glass: 6 cm across, 12 cm tall
R_OUT, H_GLASS = R_IN + WALL, BOTTOM + CAVITY
FREE = float(os.environ.get("G1P_FREE", "0.01"))  # free space above the settled water (difficulty knob)
GRASP_H = 0.055  # the hand holds the glass this high above its base (the Dex3 ring centre)
GLASS_MASS = 0.20
# Start and target spots of the glass on the table (env frame); the target is at least MIN_MOVE from the start.
START_LO, START_HI = (0.30, -0.20), (0.40, -0.02)
TARGET_LO, TARGET_HI = (0.26, -0.30), (0.40, 0.00)
MIN_MOVE = 0.12
TABLE_TOP = 0.75
TABLE_CENTER, TABLE_SIZE = (0.36, -0.12), (0.40, 0.70)
GLASS_Z = TABLE_TOP  # glass base height when it stands on the table
PBF_SPACING = float(os.environ.get("G1P_PBF_SPACING", "0.005"))


def _ring(r, z, n):
    return [(r * math.cos(2 * math.pi * k / n), r * math.sin(2 * math.pi * k / n), z) for k in range(n)]


def bowl_mesh(n: int = 32):
    pts = _ring(R_OUT, 0.0, n) + _ring(R_OUT, H_GLASS, n) + _ring(R_IN, H_GLASS, n) + _ring(R_IN, BOTTOM, n) + [(0, 0, 0), (0, 0, BOTTOM)]
    co, ci = 4 * n, 4 * n + 1
    faces = []
    for i in range(n):
        j = (i + 1) % n
        faces += [(i, j, n + j, n + i), (n + i, n + j, 2 * n + j, 2 * n + i), (2 * n + i, 2 * n + j, 3 * n + j, 3 * n + i)]
        faces += [(co, j, i), (ci, 3 * n + i, 3 * n + j)]
    return pts, [len(f) for f in faces], [v for f in faces for v in f]


def cylinder_mesh(r, z0, z1, n: int = 24):
    pts = _ring(r, z0, n) + _ring(r, z1, n) + [(0, 0, z0), (0, 0, z1)]
    faces = []
    for i in range(n):
        j = (i + 1) % n
        faces += [(i, j, n + j, n + i), (2 * n, j, i), (2 * n + 1, n + i, n + j)]
    return pts, [len(f) for f in faces], [v for f in faces for v in f]


def _mesh(stage, path, mesh, color):
    from pxr import Gf, UsdGeom  # noqa: PLC0415
    pts, counts, idx = mesh
    m = UsdGeom.Mesh.Define(stage, path)
    m.CreatePointsAttr([Gf.Vec3f(*v) for v in pts])
    m.CreateFaceVertexCountsAttr(counts)
    m.CreateFaceVertexIndicesAttr(idx)
    m.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    return m


@clone
def spawn_wine_glass(prim_path, cfg, translation=None, orientation=None, **kwargs):
    """Free drinking glass: visual walls, a colliding base disc, mass at the water (the grasp weld is added to the builder)."""
    from pxr import Gf, UsdGeom, UsdPhysics, UsdShade  # noqa: PLC0415

    stage = sim_utils.get_current_stage()
    xf = UsdGeom.Xform.Define(stage, prim_path)
    if translation is not None:
        xf.AddTranslateOp().Set(Gf.Vec3d(*translation))
    if orientation is not None:
        xf.AddOrientOp().Set(Gf.Quatf(orientation[3], Gf.Vec3f(*orientation[:3])))
    prim = stage.GetPrimAtPath(prim_path)
    UsdPhysics.RigidBodyAPI.Apply(prim)
    mass = UsdPhysics.MassAPI.Apply(prim)
    mass.CreateMassAttr(cfg.mass)
    mass.CreateCenterOfMassAttr(Gf.Vec3f(0.0, 0.0, 0.05))  # mostly the water
    mass.CreateDiagonalInertiaAttr(Gf.Vec3f(6e-4, 6e-4, 2e-4))
    _mesh(stage, f"{prim_path}/glass_visual", bowl_mesh(), (0.78, 0.86, 0.93))
    foot = UsdGeom.Cylinder.Define(stage, f"{prim_path}/foot")
    foot.CreateRadiusAttr(R_OUT)
    foot.CreateHeightAttr(BOTTOM)
    foot.CreateAxisAttr("Z")
    foot.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, 0.5 * BOTTOM))
    foot.CreateDisplayColorAttr([Gf.Vec3f(0.72, 0.80, 0.88)])
    UsdPhysics.CollisionAPI.Apply(foot.GetPrim())
    mat = UsdShade.Material.Define(stage, f"{prim_path}/foot_material")
    pm = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
    pm.CreateStaticFrictionAttr(0.8)
    pm.CreateDynamicFrictionAttr(0.8)
    UsdShade.MaterialBindingAPI.Apply(foot.GetPrim()).Bind(mat, UsdShade.Tokens.weakerThanDescendants, "physics")
    return prim


@configclass
class WineGlassSpawnCfg(SpawnerCfg):
    func: object = spawn_wine_glass
    mass: float = GLASS_MASS


def g1_cfg():
    cfg = G1_29DOF_CFG.copy()
    cfg.spawn = cfg.spawn.replace(fix_root_link=True, articulation_props=[NewtonArticulationCfg(self_collision_enabled=False)],
                                  joint_drive_props=[MujocoJointCfg(actuatorgravcomp=True)],
                                  collision_props=[UsdPhysicsCollisionCfg(collision_enabled=False)])  # visual fingers, no robot contacts
    cfg.prim_path = "{ENV_REGEX_NS}/Robot"
    joint_pos = dict(cfg.init_state.joint_pos)
    joint_pos.update(dict(zip(RIGHT_ARM, READY)))
    joint_pos.update(dict(zip(RIGHT_HAND, HAND_OPEN)))
    cfg.init_state = cfg.init_state.replace(pos=(0.0, 0.0, 0.80), joint_pos=joint_pos)
    for name, (kp, kd, arm) in {"arms": (3000.0, 100.0, 0.03), "waist": (5000.0, 300.0, 0.03), "feet": (20.0, 2.0, 0.03),
                                "hands": (20.0, 2.0, 0.03)}.items():
        cfg.actuators[name] = cfg.actuators[name].replace(stiffness=kp, damping=kd, armature=arm)
    return cfg


@configclass
class G1PPSceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(prim_path="/World/GroundPlane", spawn=sim_utils.GroundPlaneCfg())
    robot = g1_cfg()  # before the glass: the glass spawner looks up the palm for the grasp weld
    glass = RigidObjectCfg(prim_path="{ENV_REGEX_NS}/Glass", spawn=WineGlassSpawnCfg(),
                           init_state=RigidObjectCfg.InitialStateCfg(pos=(0.36, -0.05, GLASS_Z)))
    table = AssetBaseCfg(prim_path="{ENV_REGEX_NS}/Table",
                         init_state=AssetBaseCfg.InitialStateCfg(pos=(TABLE_CENTER[0], TABLE_CENTER[1], TABLE_TOP - 0.025)),
                         spawn=sim_utils.CuboidCfg(size=(TABLE_SIZE[0], TABLE_SIZE[1], 0.05), collision_props=[UsdPhysicsCollisionCfg(collision_enabled=True)],
                                                   visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.55, 0.42, 0.30))))
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


def _add_grasp_welds(_event) -> None:
    """MODEL_INIT: one MuJoCo weld equality per world, palm (body1) - glass (body2), disabled, relpose = the nominal grasp
    (stem axis through GRIP, stem middle at the palm plane). Added to the builder because Isaac Lab parses each USD asset
    separately and drops a joint between two assets."""
    import re  # noqa: PLC0415
    import warp as wp  # noqa: PLC0415
    from isaaclab.sim import SimulationContext  # noqa: PLC0415
    from newton._src.solvers.mujoco.enums import EqType  # noqa: PLC0415
    from newton._src.solvers.mujoco.equality import _add_equality_constraint, _register_equality_constraint_attributes  # noqa: PLC0415

    builder = SimulationContext.instance().backend_manager.builder
    labels = list(builder.body_label)
    palms = {int(m.group(1)): i for i, l in enumerate(labels) if (m := re.match(r"/World/envs/env_(\d+)/Robot/.*right_hand_palm_link$", l))}
    glasses = {int(m.group(1)): i for i, l in enumerate(labels) if (m := re.match(r"/World/envs/env_(\d+)/Glass$", l))}
    if not palms or palms.keys() != glasses.keys():
        raise RuntimeError(f"grasp welds: palms {len(palms)} vs glasses {len(glasses)} in the builder")
    _register_equality_constraint_attributes(builder)
    relpose = wp.transform(wp.vec3(GRIP[0], GRIP[1], -GRASP_H), wp.quat_identity())
    world = builder.body_world if hasattr(builder, "body_world") else None
    for e in sorted(palms):
        w = int(world[glasses[e]]) if world is not None else e
        builder._current_world = w
        _add_equality_constraint(builder, constraint_type=EqType.WELD, body1=palms[e], body2=glasses[e], anchor=wp.vec3(0.0, 0.0, 0.0),
                                 relpose=relpose, torquescale=1.0, label=f"/World/envs/env_{e}/GraspWeld", enabled=False)
    builder._current_world = -1


def register_grasp_welds(env, env_ids) -> None:
    """prestartup event: after the scene exists, before the simulation starts (the builder is finalized at MODEL_INIT)."""
    from isaaclab.physics import PhysicsEvent  # noqa: PLC0415
    from isaaclab.sim import SimulationContext  # noqa: PLC0415
    SimulationContext.instance().backend_manager.register_callback(_add_grasp_welds, PhysicsEvent.MODEL_INIT, name="grasp_welds")


@configclass
class EventsCfg:
    grasp_welds = EventTerm(func=register_grasp_welds, mode="prestartup")
    reset_robot = EventTerm(func=mdp.reset_joints_by_offset, mode="reset",
                            params={"asset_cfg": SceneEntityCfg("robot"), "position_range": (0.0, 0.0), "velocity_range": (0.0, 0.0)})


@configclass
class RewardsCfg:
    alive = RewTerm(func=mdp.is_alive, weight=0.0)


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)


@configclass
class G1PPEnvCfg(ManagerBasedRLEnvCfg):
    scene: G1PPSceneCfg = G1PPSceneCfg(num_envs=4, env_spacing=2.5)
    actions: ActionsCfg = ActionsCfg()
    observations: ObservationsCfg = ObservationsCfg()
    events: EventsCfg = EventsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    def __post_init__(self):
        self.decimation = DECIMATION
        self.episode_length_s = 15.0
        self.sim = sim_utils.SimulationCfg(dt=SIM_DT, physics=NewtonCfg(
            solver_cfg=MJWarpSolverCfg(solver="newton", integrator="implicitfast", njmax=256, nconmax=64, iterations=100, ls_iterations=50),
            num_substeps=3, use_cuda_graph=True))
        self.sim.use_newton_actuators = True


class G1PPEnv(ManagerBasedRLEnv):
    cfg: G1PPEnvCfg

    def __init__(self, cfg, **kwargs):
        self._palm = None
        self._hand_ids = None
        n, d = cfg.scene.num_envs, cfg.sim.device
        self.water_sim = PBFWater(n, R_IN, R_OUT, BOTTOM, H_GLASS, CAVITY - FREE, d, spacing=PBF_SPACING,
                                  substeps=int(os.environ.get("G1P_PBF_SUBSTEPS", "16")), iterations=int(os.environ.get("G1P_PBF_ITERS", "6")))  # 3 let a tall column bounce (spill 5-9 %)
        self.water_sim.presettle(CAVITY - FREE)
        self._pose0 = None
        self.grasped = torch.zeros(n, dtype=torch.bool, device=d)
        self.glass_start = torch.tensor([[0.36, -0.05]], device=d).repeat(n, 1)
        self.target_xy = torch.tensor([[0.30, -0.25]], device=d).repeat(n, 1)
        self._target_markers = None
        self._weld_idx = None
        self._markers = None
        super().__init__(cfg, **kwargs)
        update = self.scene.update

        def update_with_water(dt):
            update(dt)
            self._advance_water()

        self.scene.update = update_with_water
        self._find_welds()
        self.set_weld(torch.arange(self.num_envs, device=self.device), False)
        if os.environ.get("G1P_SHOW_WATER") == "1":
            from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg  # noqa: PLC0415
            self._target_markers = VisualizationMarkers(VisualizationMarkersCfg(prim_path="/Visuals/Target", markers={
                "spot": sim_utils.CylinderCfg(radius=R_OUT + 0.01, height=0.002, visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 0.75, 0.25)))}))
            from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg  # noqa: PLC0415
            self._markers = VisualizationMarkers(VisualizationMarkersCfg(prim_path="/Visuals/Water", markers={
                "drop": sim_utils.SphereCfg(radius=0.5 * PBF_SPACING, visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.15, 0.45, 0.95)))}))

    # --- handles ---
    @property
    def robot(self):
        return self.scene["robot"]

    @property
    def glass(self):
        return self.scene["glass"]

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

    def glass_pose_w(self):
        return self.glass.data.root_pos_w.torch, self.glass.data.root_quat_w.torch

    def palm_pose_w(self):
        return self.robot.data.body_pos_w.torch[:, self.palm], self.robot.data.body_quat_w.torch[:, self.palm]

    def grip_point_w(self):
        """Centre of the Dex3 ring (where the stem axis goes), world frame."""
        pp, pq = self.palm_pose_w()
        return pp + math_utils.quat_apply(pq, torch.tensor([GRIP[0], GRIP[1], 0.0], device=self.device).expand(self.num_envs, -1))

    def target_pos_w(self):
        """Target spot on the table (world), where the glass foot should stand."""
        p = torch.zeros(self.num_envs, 3, device=self.device)
        p[:, :2], p[:, 2] = self.target_xy, TABLE_TOP + 0.001
        return p + self.scene.env_origins

    def stem_mid_w(self):  # (name kept from the wine glass) the grasp point: glass axis at GRASP_H
        gp, gq = self.glass_pose_w()
        return gp + math_utils.quat_apply(gq, torch.tensor([0.0, 0.0, GRASP_H], device=self.device).expand(self.num_envs, -1))

    # --- grasp weld ---
    def _find_welds(self):
        m = self.sim.backend_manager.backend.model
        labels = list(m.mujoco.equality_constraint_label)
        world = m.mujoco.equality_constraint_world.numpy()
        idx = [-1] * self.num_envs
        for k, lab in enumerate(labels):
            if lab.endswith("GraspWeld") and 0 <= world[k] < self.num_envs:
                idx[int(world[k])] = k
        if min(idx) < 0:
            raise RuntimeError(f"grasp welds not found for every env: {idx[:8]}... (labels {labels[:4]})")
        self._weld_idx = torch.tensor(idx, device=self.device, dtype=torch.long)

    def set_weld(self, ids: torch.Tensor, on: bool):
        """Switch the grasp weld of envs `ids`; switching on takes the current palm-glass pose as the weld's relative pose."""
        import warp as wp  # noqa: PLC0415
        from newton import ModelFlags  # noqa: PLC0415

        if len(ids) == 0:
            return
        backend = self.sim.backend_manager.backend
        m = backend.model
        k = self._weld_idx[ids]
        enabled = wp.to_torch(m.mujoco.equality_constraint_enabled)
        if on:
            pp, pq = self.palm_pose_w()
            gp, gq = self.glass_pose_w()
            rel_p = math_utils.quat_apply_inverse(pq[ids], gp[ids] - pp[ids])
            rel_q = math_utils.quat_mul(math_utils.quat_conjugate(pq[ids]), gq[ids])
            relpose = wp.to_torch(m.mujoco.equality_constraint_relpose).view(-1, 7)
            relpose[k] = torch.cat([rel_p, rel_q], -1)
        enabled[k] = on
        self.grasped[ids] = on
        backend.model_changes.add(int(ModelFlags.CONSTRAINT_PROPERTIES))
        backend.notify_model_changes()

    def set_hand(self, ids: torch.Tensor, closed: bool):
        t = torch.tensor(HAND_CLOSED if closed else HAND_OPEN, device=self.device).expand(len(ids), -1)
        self.robot.set_joint_position_target_index(target=t, joint_ids=self.hand_ids, env_ids=ids)

    # --- water ---
    def step(self, action):
        p, q = self.glass_pose_w()
        self._pose0 = (p.clone(), q.clone())
        out = super().step(action)
        if self._markers is not None:
            self._markers.visualize(translations=self.water_sim.positions()[0])
        if self._target_markers is not None:
            self._target_markers.visualize(translations=self.target_pos_w()[:1])
        return out

    def _advance_water(self):
        if self._pose0 is None:
            return
        p1, q1 = self.glass_pose_w()
        self.water_sim.step(self._pose0[0], self._pose0[1], p1.clone(), q1.clone(), self.step_dt)

    def spilled_fraction(self):
        return self.water_sim.spilled_fraction()

    def glass_tilt(self):
        _, q = self.glass_pose_w()
        up = math_utils.quat_apply(q, torch.tensor([0.0, 0.0, 1.0], device=self.device).expand(q.shape[0], -1))
        return torch.acos(up[:, 2].clamp(-1.0, 1.0))

    # --- reset ---
    def _reset_idx(self, env_ids):
        super()._reset_idx(env_ids)
        ids = torch.arange(self.num_envs, device=self.device)[env_ids] if isinstance(env_ids, slice) else torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        if self._weld_idx is not None:
            self.set_weld(ids, False)
        self.set_hand(ids, False)
        k = len(ids)
        lo, hi = torch.tensor(START_LO, device=self.device), torch.tensor(START_HI, device=self.device)
        tlo, thi = torch.tensor(TARGET_LO, device=self.device), torch.tensor(TARGET_HI, device=self.device)
        start = lo + (hi - lo) * torch.rand(k, 2, device=self.device)
        tgt = tlo + (thi - tlo) * torch.rand(k, 2, device=self.device)
        for _ in range(20):  # resample targets closer than MIN_MOVE to the start
            bad = (tgt - start).norm(dim=-1) < MIN_MOVE
            if not bad.any():
                break
            tgt[bad] = tlo + (thi - tlo) * torch.rand(int(bad.sum()), 2, device=self.device)
        self.glass_start[ids], self.target_xy[ids] = start, tgt
        pos = torch.zeros(len(ids), 3, device=self.device)
        pos[:, :2] = self.glass_start[ids]
        pos[:, 2] = GLASS_Z
        pos = pos + self.scene.env_origins[ids]
        quat = torch.tensor([[0.0, 0.0, 0.0, 1.0]], device=self.device).expand(len(ids), -1)
        self.glass.write_root_pose_to_sim_index(root_pose=torch.cat([pos, quat], -1), env_ids=ids)
        self.glass.write_root_velocity_to_sim_index(root_velocity=torch.zeros(len(ids), 6, device=self.device), env_ids=ids)
        self.scene.write_data_to_sim()
        self.sim.forward()
        self.robot.update(0.0)
        self.glass.update(0.0)
        gp, gq = self.glass_pose_w()
        self.water_sim.reset(ids, gp.clone(), gq.clone())
