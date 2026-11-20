"""Check the partial-progress starts (option 3): every reset from the start bank; the arm holds still 1 s with the gripper
command matching the hand (held -> close). Per env: start type, held, ring-to-grasp-point distance, glass tilt and height
change over the second, spill. --video_dir films env 0 over several resets."""
import argparse
import os

from isaaclab.app import add_launcher_args, launch_simulation

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=8)
parser.add_argument("--resets", type=int, default=2)
parser.add_argument("--video_dir", default=None)
add_launcher_args(parser)
args = parser.parse_args()
os.environ.setdefault("G1P_START_BANK", os.path.join(os.path.dirname(os.path.abspath(__file__)), "demos", "start_bank_v3.npz"))
os.environ["G1P_START_FRACTION"] = "1.0"
if args.video_dir:
    os.environ["G1P_SHOW_WATER"] = "1"

from g1_pp_task import G1PPTaskEnv, G1PPTaskEnvCfg  # noqa: E402

cfg = G1PPTaskEnvCfg()
cfg.scene.num_envs = args.num_envs
cfg.sim.device = args.device
if args.video_dir:
    from isaaclab.envs.utils.video_recorder_cfg import VideoRecorderCfg
    from isaaclab.visualizers import VisualizerCfg, WindowCfg
    from isaaclab_visualizers.newton import NewtonGLBackendCfg

    cfg.sim.visualizer_cfgs = [VisualizerCfg(eye=(1.35, -0.75, 1.35), lookat=(0.28, -0.12, 0.88), focal_length=22.0, window=WindowCfg(size=(1280, 720)),
                                             backend=NewtonGLBackendCfg(show_particles=False))]
    cfg.video_recorders = [VideoRecorderCfg(source="viz:newton_gl", output_dir=args.video_dir, output_filename_prefix="bank",
                                            video_length=args.resets * 17, fps=15)]
with launch_simulation(cfg, args):
    import torch

    env = G1PPTaskEnv(cfg=cfg)
    n, d = env.num_envs, env.device
    types = ["pregrasp", "grasped", "carry", "lower"]
    for r in range(args.resets):
        env.reset()
        held0 = env.grasped.clone()
        g0 = env.glass_base_e().clone()
        ring = (env.grip_point_w() - env.stem_mid_w()).norm(dim=-1)
        grip = torch.where(env.hand_closed, 1.0, -1.0)
        for _ in range(15):
            a = torch.zeros(n, 8, device=d)
            a[:, 7] = grip
            env.step(a)
        g1 = env.glass_base_e()
        for i in range(n):
            print(f"RESET {r} env {i}: held {bool(held0[i])} closed {bool(env.hand_closed[i])}, ring-grasp-point {float(ring[i]) * 1000:.1f} mm, "
                  f"glass moved {float((g1[i] - g0[i]).norm()) * 1000:.1f} mm, z {float(g1[i, 2]):.3f}, tilt {float(env.glass_tilt()[i]) * 57.3:.1f} deg, "
                  f"spill {float(env.spilled_fraction()[i]) * 100:.2f} %, still held {bool(env.grasped[i])}", flush=True)
    env.close()
