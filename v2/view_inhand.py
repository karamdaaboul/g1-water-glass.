"""View (2026-11-18) of the in-hand glass (g1_inhand_env.py): arm at home, still for --steps; one camera (--eye/--lookat/--focal).
Prints spill and glass tilt at the end."""
import argparse
from isaaclab.app import add_launcher_args, launch_simulation
parser = argparse.ArgumentParser(); parser.add_argument("--out", required=True); parser.add_argument("--steps", type=int, default=20)
parser.add_argument("--eye", type=float, nargs=3, default=(0.62, -0.42, 1.08)); parser.add_argument("--lookat", type=float, nargs=3, default=(0.30, -0.12, 0.84))
parser.add_argument("--focal", type=float, default=35.0); add_launcher_args(parser); args = parser.parse_args()
from g1_inhand_env import G1InHandEnv, G1InHandEnvCfg, set_mpm_capacity  # noqa: E402
from isaaclab.envs.utils.video_recorder_cfg import VideoRecorderCfg  # noqa: E402
from isaaclab.visualizers import VisualizerCfg, WindowCfg  # noqa: E402
from isaaclab_visualizers.newton import NewtonGLBackendCfg  # noqa: E402
cfg = G1InHandEnvCfg(); cfg.scene.num_envs = 1; cfg.sim.device = args.device; set_mpm_capacity(cfg)
cfg.sim.visualizer_cfgs = [VisualizerCfg(eye=tuple(args.eye), lookat=tuple(args.lookat), focal_length=args.focal, window=WindowCfg(size=(1280, 720)),
                                         backend=NewtonGLBackendCfg(show_particles=True, particle_color=(0.15, 0.45, 0.95)))]
cfg.video_recorders = [VideoRecorderCfg(source="viz:newton_gl", output_dir=args.out, output_filename_prefix="view", video_length=args.steps, fps=15)]
with launch_simulation(cfg, args):
    import torch
    env = G1InHandEnv(cfg=cfg); env.reset()
    for _ in range(args.steps + 3):
        env.step(torch.zeros(1, 7, device=env.device))
    pos, _ = env.cup_pose_w()
    print(f"STILL spill {env.spilled_fraction()[0]:.4f} tilt {env.cup_tilt()[0] * 57.3:.2f} deg glass base {[round(v, 4) for v in (pos[0] - env.scene.env_origins[0]).tolist()]} "
          f"fingers {[round(v, 2) for v in env.robot.data.joint_pos.torch[0, env.hand_ids].tolist()]}", flush=True)
    env.close()
