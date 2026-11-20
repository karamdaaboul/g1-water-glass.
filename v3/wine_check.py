"""v3 wine-glass spill check (2026-11-19), from v2/grip_video.py: hold 10 steps, then home -> +15 cm forward -> +10 cm left -> home,
each leg a --T s minimum-jerk path (upright resolved-rate law). Original doc: Grip video (2026-11-18) for the user's review of the in-hand glass (g1_inhand_env.py): the arm carries the glass along
home -> +12 cm forward -> +10 cm to the left -> +8 cm up -> home, each leg a 2.5 s minimum-jerk path, with the demonstrator's
resolved-rate law (glass Jacobian, tilt correction keeps it upright). One camera per run; prints spill and max tilt."""
import argparse
from isaaclab.app import add_launcher_args, launch_simulation
parser = argparse.ArgumentParser(); parser.add_argument("--out", required=True)
parser.add_argument("--eye", type=float, nargs=3, default=(0.85, -0.55, 1.15)); parser.add_argument("--lookat", type=float, nargs=3, default=(0.33, -0.08, 0.85))
parser.add_argument("--focal", type=float, default=26.0); parser.add_argument("--T", type=float, default=2.5); add_launcher_args(parser); args = parser.parse_args()
from g1_wine_env import G1InHandEnv, G1InHandEnvCfg, set_mpm_capacity  # noqa: E402
from isaaclab.envs.utils.video_recorder_cfg import VideoRecorderCfg  # noqa: E402
from isaaclab.visualizers import VisualizerCfg, WindowCfg  # noqa: E402
from isaaclab_visualizers.newton import NewtonGLBackendCfg  # noqa: E402
LEGS = [(0.15, 0.0, 0.0), (0.15, 0.10, 0.0), (0.0, 0.0, 0.0)]
HOLD = 6
T = args.T
STEPS = 10 + len(LEGS) * (int(T * 15) + HOLD) + 4
cfg = G1InHandEnvCfg(); cfg.scene.num_envs = 1; cfg.sim.device = args.device; set_mpm_capacity(cfg); cfg.episode_length_s = 60.0
cfg.sim.visualizer_cfgs = [VisualizerCfg(eye=tuple(args.eye), lookat=tuple(args.lookat), focal_length=args.focal, window=WindowCfg(size=(1280, 720)),
                                         backend=NewtonGLBackendCfg(show_particles=True, particle_color=(0.15, 0.45, 0.95)))]
cfg.video_recorders = [VideoRecorderCfg(source="viz:newton_gl", output_dir=args.out, output_filename_prefix="grip", video_length=STEPS, fps=15)]
with launch_simulation(cfg, args):
    import torch
    from isaaclab.utils import math as mu
    env = G1InHandEnv(cfg=cfg); env.reset()
    d, dt = env.device, env.step_dt
    term = env.action_manager.get_term("right_arm"); ids = term._joint_ids; vmax = term._max_vel[0]
    z = torch.tensor([[0.0, 0.0, 1.0]], device=d)
    pos = lambda: env.cup_pose_w()[0] - env.scene.env_origins  # noqa: E731
    for _ in range(10):
        env.step(torch.zeros(1, 7, device=d))
    print(f"HOLD spill {env.spilled_fraction()[0]:.4f}", flush=True)
    home = pos().clone(); start = home.clone(); max_tilt = 0.0; k = 0
    for leg in LEGS:
        goal = home + torch.tensor([leg], device=d)
        for s in range(int(T * 15) + HOLD):
            tau = min(1.0, s * dt / T)
            sm = 10 * tau**3 - 15 * tau**4 + 6 * tau**5; ds = (30 * tau**2 - 60 * tau**3 + 30 * tau**4) / T if tau < 1 else 0.0
            p_ref, v_ref = start + (goal - start) * sm, (goal - start) * ds
            v = v_ref + 3.0 * (p_ref - pos())
            up = mu.quat_apply(env.cup_pose_w()[1], z); w = 4.0 * torch.cross(up, z, dim=-1)
            J = env.robot.data.body_link_jacobian_w.torch[:, env.cup_body - 1][:, :5][:, :, ids]
            x = torch.cat([v, w[:, :2]], -1)
            qd = (J.transpose(1, 2) @ torch.linalg.inv(J @ J.transpose(1, 2) + 0.05**2 * torch.eye(5, device=d)) @ x[..., None])[..., 0]
            env.step((qd / vmax).clamp(-1, 1))
            max_tilt = max(max_tilt, float(env.cup_tilt()[0]) * 57.3); k += 1
        start = goal
    for _ in range(4):
        env.step(torch.zeros(1, 7, device=d))
    print(f"T {T} s: GRIP VIDEO {k} steps: spill {env.spilled_fraction()[0]:.4f}, max tilt {max_tilt:.2f} deg, end error {float((pos() - home).norm()) * 1000:.1f} mm", flush=True)
    env.close()
