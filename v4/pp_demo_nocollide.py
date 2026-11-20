"""Scripted pick-and-place demonstrator for the G1 wine glass (v4, 2026-11-19).

All envs run the same phase schedule (seconds); the grip point (centre of the Dex3 ring) follows minimum-jerk segments, the palm z
axis is held vertical (tilt correction), joint rates by damped least squares on [grip-point Jacobian; palm angular x/y rows]:
  pregrasp 1.5 grip point 8 cm back from the glass at grasp height  approach 1.0 into the glass (it enters the open ring)
  close   0.4  hand closes; weld on at the end                    lift    1.0  up 6 cm
  carry   T_C  to 6 cm above the target's stem-middle height     lower   1.0  to the standing height (+3 mm)
  release 0.4  weld off, hand opens                               retreat 1.0  8 cm back towards the palm side (the stem leaves
               the open ring), rise 0.8 up 5 cm, then hold
Success: glass within 3 cm of the target, upright (< 5 deg), released, hand > 4 cm away; spill reported.
--video_dir: Isaac Lab VideoRecorder (headless Newton GL) of env 0 with the water and the target drawn.
"""
import argparse
import os

from isaaclab.app import add_launcher_args, launch_simulation

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--num_envs", type=int, default=8)
parser.add_argument("--t_carry", type=float, default=2.5)
parser.add_argument("--video_dir", default=None)
parser.add_argument("--eye", type=float, nargs=3, default=(1.25, -0.75, 1.30))
parser.add_argument("--lookat", type=float, nargs=3, default=(0.33, -0.12, 0.82))
parser.add_argument("--focal", type=float, default=22.0)
parser.add_argument("--seed", type=int, default=0)
add_launcher_args(parser)
args = parser.parse_args()
if args.video_dir:
    os.environ["G1P_SHOW_WATER"] = "1"

from g1_pp_env import GLASS_Z, GRASP_H, G1PPEnv, G1PPEnvCfg  # noqa: E402

PHASES = [("pregrasp", 1.5), ("approach", 1.0), ("close", 0.4), ("lift", 1.0), ("carry", args.t_carry), ("lower", 1.0),
          ("release", 0.4), ("retreat", 1.0), ("rise", 0.8), ("hold", 0.6)]
cfg = G1PPEnvCfg()
cfg.scene.num_envs = args.num_envs
cfg.sim.device = args.device
cfg.seed = args.seed
cfg.episode_length_s = sum(t for _, t in PHASES) + 2.0
if args.video_dir:
    from isaaclab.envs.utils.video_recorder_cfg import VideoRecorderCfg
    from isaaclab.visualizers import VisualizerCfg, WindowCfg
    from isaaclab_visualizers.newton import NewtonGLBackendCfg

    cfg.sim.visualizer_cfgs = [VisualizerCfg(eye=tuple(args.eye), lookat=tuple(args.lookat), focal_length=args.focal, window=WindowCfg(size=(1280, 720)),
                                             backend=NewtonGLBackendCfg(show_particles=False))]
    steps = int(sum(t for _, t in PHASES) * 15) + 2
    cfg.video_recorders = [VideoRecorderCfg(source="viz:newton_gl", output_dir=args.video_dir, output_filename_prefix="pick_place",
                                            video_length=steps, fps=15)]
with launch_simulation(cfg, args):
    import torch
    from isaaclab.utils import math as mu

    env = G1PPEnv(cfg=cfg)
    env.reset()
    n, d, dt = env.num_envs, env.device, env.step_dt
    term = env.action_manager.get_term("right_arm")
    ids_arm, vmax = term._joint_ids, term._max_vel[0]
    all_ids = torch.arange(n, device=d)
    zw = torch.tensor([0.0, 0.0, 1.0], device=d).expand(n, -1)
    stem_z = GLASS_Z + GRASP_H  # grasp point height when the glass stands on the table (env frame)
    origin = env.scene.env_origins
    max_spill = torch.zeros(n, device=d)
    max_tilt = torch.zeros(n, device=d)

    def waypoint(name, start):
        stem = env.stem_mid_w()
        tgt = env.target_pos_w()
        if name == "pregrasp":  # 8 cm back from the glass at grasp height: the glass then enters the open ring from the side
            u = env.grip_point_w() - env.palm_pose_w()[0]
            u[:, 2] = 0.0
            return stem - 0.08 * u / u.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        if name == "approach":
            return stem
        if name == "lift":
            return start + torch.tensor([0.0, 0.0, 0.06], device=d)
        if name == "carry":
            p = start.clone()
            p[:, :2] = tgt[:, :2]
            p[:, 2] = origin[:, 2] + stem_z + 0.06
            return p
        if name == "lower":
            p = start.clone()
            p[:, 2] = origin[:, 2] + stem_z + 0.003
            return p
        if name == "retreat":  # horizontally away from the stem, towards the palm side
            back = env.palm_pose_w()[0] - start
            back[:, 2] = 0.0
            return start + 0.08 * back / back.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        if name == "rise":
            return start + torch.tensor([0.0, 0.0, 0.05], device=d)
        return start  # close, release, hold

    for name, T in PHASES:
        start = env.grip_point_w().clone()
        goal = waypoint(name, start)
        if name == "close":
            env.set_hand(all_ids, True)
        if name == "release":
            env.set_weld(all_ids, False)
            env.set_hand(all_ids, False)
        steps = max(1, int(round(T / dt)))
        for k in range(steps):
            tau = min(1.0, (k + 1) * dt / T)
            s = 10 * tau**3 - 15 * tau**4 + 6 * tau**5
            ds = (30 * tau**2 - 60 * tau**3 + 30 * tau**4) / T if tau < 1 else 0.0
            p_ref, v_ref = start + (goal - start) * s, (goal - start) * ds
            g = env.grip_point_w()
            v = v_ref + 3.0 * (p_ref - g)
            pp, pq = env.palm_pose_w()
            zp = mu.quat_apply(pq, zw)
            w = 4.0 * torch.cross(zp, zw, dim=-1)
            J = env.robot.data.body_link_jacobian_w.torch[:, env.palm - 1][:, :, ids_arm]  # fixed base: row = body - 1
            r = g - pp
            rx = torch.zeros(n, 3, 3, device=d)
            rx[:, 0, 1], rx[:, 0, 2], rx[:, 1, 0], rx[:, 1, 2], rx[:, 2, 0], rx[:, 2, 1] = -r[:, 2], r[:, 1], r[:, 2], -r[:, 0], -r[:, 1], r[:, 0]
            Jg = J[:, :3] - rx @ J[:, 3:]
            A = torch.cat([Jg, J[:, 3:5]], 1)
            x = torch.cat([v, w[:, :2]], -1)
            qd = (A.transpose(1, 2) @ torch.linalg.inv(A @ A.transpose(1, 2) + 0.05**2 * torch.eye(5, device=d)) @ x[..., None])[..., 0]
            env.step((qd / vmax).clamp(-1.0, 1.0))
            max_spill = torch.maximum(max_spill, env.spilled_fraction())
            max_tilt = torch.maximum(max_tilt, env.glass_tilt())
        if name == "close":
            env.set_weld(all_ids, True)
        if name in ("approach", "lower", "retreat"):
            err = (env.grip_point_w() - goal).norm(dim=-1)
            print(f"PHASE {name:8s}: grip error mean {float(err.mean()) * 1000:.1f} mm, max {float(err.max()) * 1000:.1f} mm", flush=True)
    gp, _ = env.glass_pose_w()
    foot_xy = gp[:, :2] - origin[:, :2]
    place_err = (foot_xy - env.target_xy).norm(dim=-1)
    tilt = env.glass_tilt()
    away = (env.grip_point_w() - env.stem_mid_w()).norm(dim=-1)
    spill = env.spilled_fraction()
    ok = (place_err < 0.03) & (tilt < 5 / 57.3) & (away > 0.04) & ~env.grasped
    for i in range(n):
        print(f"ENV {i}: place error {float(place_err[i]) * 100:.1f} cm, tilt {float(tilt[i]) * 57.3:.1f} deg (max during {float(max_tilt[i]) * 57.3:.1f}), "
              f"spill {float(spill[i]) * 100:.2f} % (max {float(max_spill[i]) * 100:.2f}), hand away {float(away[i]) * 100:.1f} cm -> {'SUCCESS' if ok[i] else 'fail'}", flush=True)
    print(f"PICKPLACE {int(ok.sum())}/{n} placed; spill mean {float(spill.mean()) * 100:.2f} %", flush=True)
    for _ in range(4):
        env.step(torch.zeros(n, 7, device=d))
    env.close()
