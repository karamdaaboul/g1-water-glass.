"""Back-of-hand collision test (2026-11-19, user request): the G1 sweeps the BACK of its open right hand through the glass.

The controlled point is the middle of the back of the hand, B = palm frame (0.04, -0.03, 0) (the fingers curl towards +y, so -y
is the back). Path: up above the glass -> over to 12 cm from the glass axis on the hand's +y side (back of the hand facing the
glass) -> down to grasp height -> sweep 22 cm along the hand's -y axis, through where the glass stands (1.2 s) -> hold.
With hand-glass contact on, the glass must be pushed or knocked over, not passed through. Prints the glass displacement, tilt
and spill; --video_dir records it (Isaac Lab VideoRecorder, Newton GL)."""
import argparse
import os

from isaaclab.app import add_launcher_args, launch_simulation

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--video_dir", default=None)
parser.add_argument("--eye", type=float, nargs=3, default=(0.85, -0.55, 1.02))
parser.add_argument("--lookat", type=float, nargs=3, default=(0.33, -0.12, 0.82))
parser.add_argument("--focal", type=float, default=30.0)
parser.add_argument("--seed", type=int, default=3)
add_launcher_args(parser)
args = parser.parse_args()
if args.video_dir:
    os.environ["G1P_SHOW_WATER"] = "1"

from g1_pp_env import GLASS_Z, GRASP_H, G1PPEnv, G1PPEnvCfg  # noqa: E402

PHASES = [("up", 1.0), ("over", 1.5), ("down", 1.0), ("sweep", 1.2), ("hold", 1.5)]
B_LOCAL = (0.04, -0.03, 0.0)
cfg = G1PPEnvCfg()
cfg.scene.num_envs = 1
cfg.sim.device = args.device
cfg.seed = args.seed
cfg.episode_length_s = sum(t for _, t in PHASES) + 2.0
if args.video_dir:
    from isaaclab.envs.utils.video_recorder_cfg import VideoRecorderCfg
    from isaaclab.visualizers import VisualizerCfg, WindowCfg
    from isaaclab_visualizers.newton import NewtonGLBackendCfg

    cfg.sim.visualizer_cfgs = [VisualizerCfg(eye=tuple(args.eye), lookat=tuple(args.lookat), focal_length=args.focal, window=WindowCfg(size=(1280, 720)),
                                             backend=NewtonGLBackendCfg(show_particles=False))]
    cfg.video_recorders = [VideoRecorderCfg(source="viz:newton_gl", output_dir=args.video_dir, output_filename_prefix="backhand",
                                            video_length=int(sum(t for _, t in PHASES) * 15) + 2, fps=15)]
with launch_simulation(cfg, args):
    import torch
    from isaaclab.utils import math as mu

    env = G1PPEnv(cfg=cfg)
    env.reset()
    d, dt, n = env.device, env.step_dt, env.num_envs
    term = env.action_manager.get_term("right_arm")
    ids_arm, vmax = term._joint_ids, term._max_vel[0]
    zw = torch.tensor([0.0, 0.0, 1.0], device=d).expand(n, -1)
    origin = env.scene.env_origins
    b_loc = torch.tensor(B_LOCAL, device=d).expand(n, -1)

    def point_b():
        pp, pq = env.palm_pose_w()
        return pp + mu.quat_apply(pq, b_loc)

    def y_hand():
        _, pq = env.palm_pose_w()
        y = mu.quat_apply(pq, torch.tensor([0.0, 1.0, 0.0], device=d).expand(n, -1))
        y[:, 2] = 0.0
        return y / y.norm(dim=-1, keepdim=True).clamp_min(1e-6)

    g0 = env.glass_pose_w()[0].clone()
    axis = g0.clone()
    axis[:, 2] = origin[:, 2] + GLASS_Z + GRASP_H
    clear_z = origin[:, 2] + GLASS_Z + 0.12 + 0.06
    plan = {}
    for name, T in PHASES:
        start = point_b().clone()
        if name == "up":
            goal = start.clone(); goal[:, 2] = clear_z
        elif name == "over":
            plan["s"] = axis + 0.12 * y_hand()
            goal = plan["s"].clone(); goal[:, 2] = clear_z
        elif name == "down":
            goal = plan["s"]
        elif name == "sweep":
            goal = axis - 0.10 * y_hand()
        else:
            goal = start
        for k in range(max(1, int(round(T / dt)))):
            tau = min(1.0, (k + 1) * dt / T)
            s = 10 * tau**3 - 15 * tau**4 + 6 * tau**5
            ds = (30 * tau**2 - 60 * tau**3 + 30 * tau**4) / T if tau < 1 else 0.0
            p_ref, v_ref = start + (goal - start) * s, (goal - start) * ds
            b = point_b()
            v = v_ref + 3.0 * (p_ref - b)
            pp, pq = env.palm_pose_w()
            w = 4.0 * torch.cross(mu.quat_apply(pq, zw), zw, dim=-1)
            J = env.robot.data.body_link_jacobian_w.torch[:, env.palm - 1][:, :, ids_arm]
            r = b - pp
            rx = torch.zeros(n, 3, 3, device=d)
            rx[:, 0, 1], rx[:, 0, 2], rx[:, 1, 0], rx[:, 1, 2], rx[:, 2, 0], rx[:, 2, 1] = -r[:, 2], r[:, 1], r[:, 2], -r[:, 0], -r[:, 1], r[:, 0]
            A = torch.cat([J[:, :3] - rx @ J[:, 3:], J[:, 3:5]], 1)
            x = torch.cat([v, w[:, :2]], -1)
            qd = (A.transpose(1, 2) @ torch.linalg.inv(A @ A.transpose(1, 2) + 0.05**2 * torch.eye(5, device=d)) @ x[..., None])[..., 0]
            env.step((qd / vmax).clamp(-1.0, 1.0))
        g = env.glass_pose_w()[0]
        print(f"PHASE {name:6s}: glass moved {float((g - g0)[0, :2].norm()) * 100:.1f} cm, tilt {float(env.glass_tilt()[0]) * 57.3:.1f} deg, "
              f"spill {float(env.spilled_fraction()[0]) * 100:.1f} %", flush=True)
    for _ in range(4):
        env.step(torch.zeros(n, 7, device=d))
    env.close()
